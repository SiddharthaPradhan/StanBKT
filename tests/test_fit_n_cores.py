"""Tests for fit(n_kcs_workers=...) concurrent KC fitting (mock based, no CmdStan)."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stanbkt.models.core.base as base_module
from stanbkt.fits.fit_options import (
    MCMCFitOptions,
    MLEFitOptions,
    PFFitOptions,
    VBFitOptions,
)
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.base import BKTModelBase
from stanbkt.models.core.multi import MultiBKT
from stanbkt.models.core.standard import StandardBKT
from stanbkt.models.model_types import InitKnowledgeStrategy
from stanbkt.utils.verbose import VerbosityLevel

KCS = ("kc_a", "kc_b", "kc_c")


@pytest.fixture(autouse=True)
def _plenty_of_cpus(monkeypatch):
    # deterministic cpu count so explicit worker counts never depend on the machine
    monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 64))


def _df(kc_students=None) -> pd.DataFrame:
    # each KC has a different number of students so per-KC fits are distinguishable
    kc_students = kc_students or {
        "kc_a": ("s1", "s2"),
        "kc_b": ("s1", "s2", "s3"),
        "kc_c": ("s1", "s2", "s3", "s4"),
    }
    rows = []
    for kc, students in kc_students.items():
        for s in students:
            for t, (p, c) in enumerate([("p1", 1), ("p2", 0), ("p3", 1)], start=1):
                rows.append(
                    {
                        "student_id": s,
                        "problem_id": p,
                        "correct": c,
                        "timestamp": t,
                        "kc_id": kc,
                        "group_id": "g1" if s in ("s1", "s3") else "g2",
                    }
                )
    return pd.DataFrame(rows)


def _covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["s1", "s2", "s3", "s4"],
            "pretest": [0.1, 0.2, 0.3, 0.4],
            "age": [10.0, 11.0, 12.0, 13.0],
        }
    )


class _DummySavedFit:
    """Fake fit that can be persisted by release_fit_from_memory."""

    def __init__(self, n_students: int):
        self.n_students = n_students

    def save_csvfiles(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "mock_chain.csv"), "w", encoding="utf-8") as f:
            f.write("lp__\n0\n")


def _mock_fit(model, monkeypatch, fake=None):
    """Patch compile and Stan fitting; returns the list of captured data dicts."""
    monkeypatch.setattr(
        model, "_compile_model", lambda _: setattr(model, "_stan_model", object())
    )
    captured: list[dict] = []
    lock = threading.Lock()

    def _default_fake(data_dict, fit_options):
        with lock:
            captured.append(data_dict)
        return _DummySavedFit(int(data_dict["nStudents"]))

    def _recording_fake(data_dict, fit_options):
        with lock:
            captured.append(data_dict)
        return fake(data_dict, fit_options)

    monkeypatch.setattr(
        model,
        "_fit_stan_model_using_method",
        _default_fake if fake is None else _recording_fake,
    )
    monkeypatch.setattr(
        FitMethod,
        "infer_fit_method_from_stan_fit",
        staticmethod(lambda _: FitMethod.MCMC),
    )
    return captured


def _fitted_state(model) -> dict:
    return {
        "order": list(model.fits.stan_fits.keys()),
        "saves": list(model.fits._fit_metadata.fit_saves.keys()),
        "n_students": {kc: fit.n_students for kc, fit in model.fits.stan_fits.items()},
    }


class TestSequentialEquivalence:
    def test_default_is_auto_and_matches_sequential(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        state = _fitted_state(model)
        assert state["order"] == list(KCS)
        assert state["n_students"] == {"kc_a": 2, "kc_b": 3, "kc_c": 4}
        assert model._is_fitted

    def test_concurrent_matches_sequential(self, monkeypatch):
        seq = StandardBKT()
        _mock_fit(seq, monkeypatch)
        seq.fit(_df(), n_kcs_workers=1)

        conc = StandardBKT()
        _mock_fit(conc, monkeypatch)
        conc.fit(_df(), n_kcs_workers=3)

        assert _fitted_state(conc) == _fitted_state(seq)
        assert conc.fits.num_fitted_kcs == seq.fits.num_fitted_kcs == 3

    def test_insertion_order_follows_kcs_when_completion_order_differs(
        self, monkeypatch
    ):
        # earliest KC finishes last, so completion order is the reverse of KC order
        delays = {2: 0.3, 3: 0.15, 4: 0.0}

        def _slow_fit(data_dict, fit_options):
            time.sleep(delays[int(data_dict["nStudents"])])
            return _DummySavedFit(int(data_dict["nStudents"]))

        model = StandardBKT()
        _mock_fit(model, monkeypatch, fake=_slow_fit)
        model.fit(_df(), n_kcs_workers=3)
        state = _fitted_state(model)
        assert state["order"] == list(KCS)
        assert state["saves"] == list(KCS)

    def test_more_cores_than_kcs(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=16)
        assert _fitted_state(model)["order"] == list(KCS)

    def test_single_kc_data_with_many_cores(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        df = _df({"kc_a": ("s1", "s2")}).drop(columns=["kc_id"])
        model.fit(df, n_kcs_workers=4)
        assert _fitted_state(model)["order"] == ["default_kc"]

    def test_group_metadata_recorded_per_kc(self, monkeypatch):
        seq = MultiBKT()
        _mock_fit(seq, monkeypatch)
        seq.fit(_df(), n_kcs_workers=1)
        conc = MultiBKT()
        _mock_fit(conc, monkeypatch)
        conc.fit(_df(), n_kcs_workers=3)
        for kc in KCS:
            a = seq.fits.get_fit_save_entry(kc)
            b = conc.fits.get_fit_save_entry(kc)
            assert a.group2index == b.group2index
            assert a.groups == b.groups


class TestConcurrency:
    def test_kcs_really_run_concurrently(self, monkeypatch):
        # all three fits must be in flight at once for the barrier to release
        barrier = threading.Barrier(3, timeout=10)

        def _barrier_fit(data_dict, fit_options):
            barrier.wait()
            return _DummySavedFit(int(data_dict["nStudents"]))

        model = StandardBKT()
        _mock_fit(model, monkeypatch, fake=_barrier_fit)
        model.fit(_df(), n_kcs_workers=3)
        assert _fitted_state(model)["order"] == list(KCS)

    def test_single_core_never_overlaps(self, monkeypatch):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def _tracking_fit(data_dict, fit_options):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return _DummySavedFit(int(data_dict["nStudents"]))

        model = StandardBKT()
        _mock_fit(model, monkeypatch, fake=_tracking_fit)
        model.fit(_df(), n_kcs_workers=1)
        assert max_active == 1

    def test_in_flight_fits_are_bounded_by_n_kcs_workers(self, monkeypatch):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def _tracking_fit(data_dict, fit_options):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return _DummySavedFit(int(data_dict["nStudents"]))

        df = _df({f"kc_{i}": ("s1", "s2") for i in range(8)})
        model = StandardBKT()
        _mock_fit(model, monkeypatch, fake=_tracking_fit)
        model.fit(df, n_kcs_workers=2)
        assert max_active <= 2
        assert model.fits.num_fitted_kcs == 8

    def test_auto_worker_count_reaches_the_executor(self, monkeypatch):
        workers: list[int] = []
        real_executor = base_module.ThreadPoolExecutor

        class _RecordingExecutor(real_executor):
            def __init__(self, max_workers=None, *args, **kwargs):
                workers.append(max_workers)
                super().__init__(max_workers, *args, **kwargs)

        monkeypatch.setattr(base_module, "ThreadPoolExecutor", _RecordingExecutor)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 8))
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=-1)
        assert workers == [2]
        assert _fitted_state(model)["order"] == list(KCS)


class TestAvailableCpus:
    def test_prefers_process_cpu_count(self, monkeypatch):
        monkeypatch.undo()
        monkeypatch.setattr(base_module.os, "process_cpu_count", lambda: 6, raising=False)
        monkeypatch.setattr(base_module.os, "cpu_count", lambda: 32)
        assert BKTModelBase._available_cpus() == 6

    def test_falls_back_to_affinity_then_cpu_count(self, monkeypatch):
        monkeypatch.undo()
        monkeypatch.delattr(base_module.os, "process_cpu_count", raising=False)
        monkeypatch.setattr(
            base_module.os, "sched_getaffinity", lambda _: {0, 1, 2}, raising=False
        )
        monkeypatch.setattr(base_module.os, "cpu_count", lambda: 32)
        assert BKTModelBase._available_cpus() == 3
        monkeypatch.delattr(base_module.os, "sched_getaffinity", raising=False)
        assert BKTModelBase._available_cpus() == 32

    def test_unknown_count_falls_back_to_one(self, monkeypatch):
        monkeypatch.undo()
        monkeypatch.delattr(base_module.os, "process_cpu_count", raising=False)
        monkeypatch.delattr(base_module.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(base_module.os, "cpu_count", lambda: None)
        assert BKTModelBase._available_cpus() == 1


class TestStanCoresPerFit:
    def test_mcmc_defaults_are_four_chains_one_thread(self):
        assert BKTModelBase._stan_cores_per_fit(MCMCFitOptions()) == 4

    def test_mcmc_multiplies_parallel_chains_and_threads(self):
        options = MCMCFitOptions(chains=4, parallel_chains=4, threads_per_chain=2)
        assert BKTModelBase._stan_cores_per_fit(options) == 8

    def test_mcmc_parallel_chains_are_capped_by_chains(self):
        options = MCMCFitOptions(chains=2, parallel_chains=8, threads_per_chain=3)
        assert BKTModelBase._stan_cores_per_fit(options) == 6

    def test_mcmc_serial_chains(self):
        options = MCMCFitOptions(chains=4, parallel_chains=1)
        assert BKTModelBase._stan_cores_per_fit(options) == 1

    def test_pathfinder_uses_num_threads(self):
        assert BKTModelBase._stan_cores_per_fit(PFFitOptions()) == 1
        assert BKTModelBase._stan_cores_per_fit(PFFitOptions(num_threads=3)) == 3

    @pytest.mark.parametrize("options", [MLEFitOptions(), VBFitOptions()])
    def test_mle_and_vb_use_one_core(self, options):
        assert BKTModelBase._stan_cores_per_fit(options) == 1


def _resolve(monkeypatch, workers, stan_cores, n_kcs, cpus):
    model = StandardBKT()
    monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: cpus))
    return model._resolve_n_kcs_workers(workers, stan_cores, n_kcs)


class TestAutoResolution:
    @pytest.mark.parametrize(
        "cpus, stan_cores, n_kcs, expected",
        [
            (16, 4, 10, 4),  # MCMC defaults on 16 cpus
            (24, 4, 100, 6),
            (8, 4, 10, 2),
            (8, 8, 10, 1),  # Stan settings use every cpu
            (8, 16, 10, 1),  # Stan settings alone oversubscribe
            (6, 4, 10, 1),  # leftover cores cannot host another fit
            (16, 8, 10, 2),  # threads_per_chain=2
            (16, 1, 10, 10),  # MLE style single core fits, capped by KCs
            (64, 1, 3, 3),
            (16, 4, 1, 1),  # single KC
            (1, 1, 5, 1),
            (16, 4, 0, 1),  # no KCs still resolves to a valid count
        ],
    )
    def test_auto_resolution(self, monkeypatch, cpus, stan_cores, n_kcs, expected):
        assert _resolve(monkeypatch, -1, stan_cores, n_kcs, cpus) == expected

    def test_auto_never_changes_the_stan_options(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 16))
        options = MCMCFitOptions(chains=4, parallel_chains=4, threads_per_chain=1)
        before = options.to_dict()
        model.fit(_df(), stan_fit_options=options, n_kcs_workers=-1)
        assert options.to_dict() == before

    def test_default_argument_is_auto(self, monkeypatch):
        workers: list[int] = []
        real_executor = base_module.ThreadPoolExecutor

        class _RecordingExecutor(real_executor):
            def __init__(self, max_workers=None, *args, **kwargs):
                workers.append(max_workers)
                super().__init__(max_workers, *args, **kwargs)

        monkeypatch.setattr(base_module, "ThreadPoolExecutor", _RecordingExecutor)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 8))
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        assert workers == [2]

    def test_mle_fits_use_all_cpus_up_to_the_kc_count(self, monkeypatch):
        workers: list[int] = []
        real_executor = base_module.ThreadPoolExecutor

        class _RecordingExecutor(real_executor):
            def __init__(self, max_workers=None, *args, **kwargs):
                workers.append(max_workers)
                super().__init__(max_workers, *args, **kwargs)

        monkeypatch.setattr(base_module, "ThreadPoolExecutor", _RecordingExecutor)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 8))
        model = StandardBKT(fit_method=FitMethod.MLE)
        _mock_fit(model, monkeypatch)
        monkeypatch.setattr(
            FitMethod,
            "infer_fit_method_from_stan_fit",
            staticmethod(lambda _: FitMethod.MLE),
        )
        model.fit(_df())
        assert workers == [3]


class TestExplicitWorkers:
    def test_within_budget_is_used_as_given(self, monkeypatch):
        assert _resolve(monkeypatch, 3, 4, 10, 16) == 3

    def test_exactly_all_cpus_is_allowed(self, monkeypatch):
        assert _resolve(monkeypatch, 4, 4, 10, 16) == 4

    def test_capped_by_kc_count(self, monkeypatch):
        assert _resolve(monkeypatch, 16, 1, 3, 16) == 3

    def test_oversubscription_raises_with_the_numbers(self, monkeypatch):
        with pytest.raises(ValueError) as err:
            _resolve(monkeypatch, 5, 4, 10, 16)
        message = str(err.value)
        assert "n_kcs_workers=5" in message
        assert "4 Stan cores per fit" in message
        assert "20 cores" in message
        assert "16 CPUs" in message
        assert "chains" in message and "threads_per_chain" in message

    def test_oversubscription_uses_the_effective_worker_count(self, monkeypatch):
        # 16 requested but only 2 KCs, so the real load is 2 x 4 = 8 cores
        assert _resolve(monkeypatch, 16, 4, 2, 8) == 2

    @pytest.mark.parametrize("cpus, stan_cores", [(1, 1), (2, 4), (4, 64)])
    def test_explicit_one_never_raises(self, monkeypatch, cpus, stan_cores):
        assert _resolve(monkeypatch, 1, stan_cores, 10, cpus) == 1

    def test_single_kc_never_raises(self, monkeypatch):
        assert _resolve(monkeypatch, 8, 64, 1, 4) == 1

    def test_oversubscription_raises_from_fit_before_any_fitting(self, monkeypatch):
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 8))
        model = StandardBKT()
        captured = _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="only 8 CPUs are available"):
            model.fit(_df(), n_kcs_workers=3)
        assert captured == []
        assert model.fits.num_fitted_kcs == 0
        assert not model._is_fitted

    def test_stan_options_are_part_of_the_budget(self, monkeypatch):
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 8))
        model = StandardBKT(cpp_compile_kwargs={"STAN_THREADS": True})
        _mock_fit(model, monkeypatch)
        threaded = MCMCFitOptions(chains=4, parallel_chains=4, threads_per_chain=2)
        with pytest.raises(ValueError, match="16 cores"):
            model.fit(_df(), stan_fit_options=threaded, n_kcs_workers=2)
        serial = MCMCFitOptions(chains=4, parallel_chains=1)
        model.fit(_df(), stan_fit_options=serial, n_kcs_workers=3)
        assert model.fits.num_fitted_kcs == 3


class TestAutoLogging:
    @staticmethod
    def _logged(model):
        messages: list[tuple[str, VerbosityLevel]] = []
        model.log = lambda message, level=VerbosityLevel.INFO: messages.append(
            (message, level)
        )
        return messages

    def test_warns_when_stan_settings_use_all_cpus(self, monkeypatch):
        model = StandardBKT()
        messages = self._logged(model)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 4))
        assert model._resolve_n_kcs_workers(-1, 4, 10) == 1
        warnings_ = [m for m, level in messages if level == VerbosityLevel.WARN]
        assert len(warnings_) == 1
        assert "resolved to 1" in warnings_[0]
        assert "oversubscribe" not in warnings_[0]

    def test_warning_notes_stan_oversubscription(self, monkeypatch):
        model = StandardBKT()
        messages = self._logged(model)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 4))
        assert model._resolve_n_kcs_workers(-1, 8, 10) == 1
        warnings_ = [m for m, level in messages if level == VerbosityLevel.WARN]
        assert len(warnings_) == 1
        assert "oversubscribe" in warnings_[0]

    def test_no_warning_for_a_single_kc(self, monkeypatch):
        model = StandardBKT()
        messages = self._logged(model)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 2))
        assert model._resolve_n_kcs_workers(-1, 4, 1) == 1
        assert all(level != VerbosityLevel.WARN for _, level in messages)

    def test_no_warning_when_leftover_cores_cannot_host_another_fit(self, monkeypatch):
        model = StandardBKT()
        messages = self._logged(model)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 6))
        assert model._resolve_n_kcs_workers(-1, 4, 10) == 1
        assert all(level != VerbosityLevel.WARN for _, level in messages)

    def test_info_when_concurrent_and_debug_when_sequential(self, monkeypatch):
        model = StandardBKT()
        messages = self._logged(model)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 16))
        assert model._resolve_n_kcs_workers(-1, 4, 10) == 4
        assert messages[-1][1] == VerbosityLevel.INFO
        assert "4 concurrent worker(s)" in messages[-1][0]
        assert model._resolve_n_kcs_workers(-1, 4, 1) == 1
        assert messages[-1][1] == VerbosityLevel.DEBUG

    def test_explicit_values_do_not_warn(self, monkeypatch):
        model = StandardBKT()
        messages = self._logged(model)
        monkeypatch.setattr(BKTModelBase, "_available_cpus", staticmethod(lambda: 4))
        assert model._resolve_n_kcs_workers(1, 8, 10) == 1
        assert messages == []


class TestInvalidNKcsWorkers:
    @pytest.mark.parametrize("bad", [0, -2, -10])
    def test_invalid_counts_raise_before_fitting(self, monkeypatch, bad):
        model = StandardBKT()
        captured = _mock_fit(model, monkeypatch)
        with pytest.raises(
            ValueError, match="'n_kcs_workers' must be -1 .auto. or a positive integer"
        ):
            model.fit(_df(), n_kcs_workers=bad)
        assert captured == []
        assert model.fits.num_fitted_kcs == 0
        assert not model._is_fitted

    @pytest.mark.parametrize("bad", ["2", 2.0, None, True, False, [2]])
    def test_non_int_raises_type_error(self, monkeypatch, bad):
        model = StandardBKT()
        captured = _mock_fit(model, monkeypatch)
        with pytest.raises(TypeError, match="'n_kcs_workers'"):
            model.fit(_df(), n_kcs_workers=bad)  # type: ignore[arg-type]
        assert captured == []

    def test_numpy_integers_are_accepted(self, monkeypatch):
        assert _resolve(monkeypatch, np.int64(2), 1, 5, 8) == 2

    def test_prediction_n_cores_validation_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(base_module.os, "cpu_count", lambda: 5)
        assert BKTModelBase._resolve_n_cores(-1) == 5
        assert BKTModelBase._resolve_n_cores(3) == 3
        with pytest.raises(ValueError, match="'n_cores' must be -1 or at least 1"):
            BKTModelBase._resolve_n_cores(0)


class TestFailures:
    @staticmethod
    def _failing_fit(failing_n_students):
        def _fit(data_dict, fit_options):
            n = int(data_dict["nStudents"])
            if n == failing_n_students:
                raise RuntimeError(f"boom {n}")
            return _DummySavedFit(n)

        return _fit

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_error_propagates_and_only_finished_prefix_is_recorded(
        self, monkeypatch, n_workers
    ):
        model = StandardBKT()
        # kc_b (3 students) fails
        _mock_fit(model, monkeypatch, fake=self._failing_fit(3))
        with pytest.raises(RuntimeError, match="boom 3"):
            model.fit(_df(), n_kcs_workers=n_workers)
        state = _fitted_state(model)
        assert state["order"] == ["kc_a"]
        assert state["saves"] == ["kc_a"]
        assert "kc_b" not in model.fits.get_fitted_kcs()
        assert model.fits.num_fitted_kcs == 1

    def test_failure_state_is_same_for_sequential_and_concurrent(self, monkeypatch):
        seq = StandardBKT()
        _mock_fit(seq, monkeypatch, fake=self._failing_fit(4))
        with pytest.raises(RuntimeError):
            seq.fit(_df(), n_kcs_workers=1)
        conc = StandardBKT()
        _mock_fit(conc, monkeypatch, fake=self._failing_fit(4))
        with pytest.raises(RuntimeError):
            conc.fit(_df(), n_kcs_workers=3)
        assert _fitted_state(seq) == _fitted_state(conc)

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_retry_with_overwrite_succeeds_after_failure(self, monkeypatch, n_workers):
        model = StandardBKT()
        _mock_fit(model, monkeypatch, fake=self._failing_fit(3))
        with pytest.raises(RuntimeError):
            model.fit(_df(), n_kcs_workers=n_workers)

        # a retry with a working fit function only needs overwrite for the KC that finished
        _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="already exists"):
            model.fit(_df(), n_kcs_workers=n_workers)
        model.fit(_df(), n_kcs_workers=n_workers, overwrite_kcs=True)
        assert _fitted_state(model)["order"] == list(KCS)
        assert model.fits.num_fitted_kcs == 3

    def test_executor_is_shut_down_after_failure(self, monkeypatch):
        before = threading.active_count()
        model = StandardBKT()
        _mock_fit(model, monkeypatch, fake=self._failing_fit(2))
        with pytest.raises(RuntimeError):
            model.fit(_df(), n_kcs_workers=3)
        time.sleep(0.1)
        assert threading.active_count() <= before

    def test_data_validation_error_before_any_fit(self, monkeypatch):
        model = StandardBKT()
        captured = _mock_fit(model, monkeypatch)
        bad = _df().drop(columns=["problem_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(bad, n_kcs_workers=3)
        assert captured == []


class TestOverwriteCheck:
    @staticmethod
    def _prefit(monkeypatch, kc_students):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        model.fit(_df(kc_students))
        return model

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_existing_kc_raises_before_any_fitting(self, monkeypatch, n_workers):
        # kc_b already fitted but sits in the middle of the data
        model = self._prefit(monkeypatch, {"kc_b": ("s1", "s2", "s3")})
        captured = _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="'kc_b' already exists"):
            model.fit(_df(), n_kcs_workers=n_workers)
        assert captured == []
        assert model.fits.num_fitted_kcs == 1
        assert list(model.fits.stan_fits.keys()) == ["kc_b"]

    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_overwrite_refits_all_data_kcs(self, monkeypatch, n_workers):
        model = self._prefit(monkeypatch, {"kc_b": ("s1", "s2")})
        _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=n_workers, overwrite_kcs=True)
        state = _fitted_state(model)
        assert set(state["order"]) == set(KCS)
        assert state["n_students"]["kc_b"] == 3

    def test_new_kcs_can_be_added_without_overwrite(self, monkeypatch):
        model = self._prefit(monkeypatch, {"kc_a": ("s1", "s2")})
        _mock_fit(model, monkeypatch)
        model.fit(_df({"kc_c": ("s1", "s2", "s3", "s4")}), n_kcs_workers=2)
        assert set(model.fits.get_fitted_kcs()) == {"kc_a", "kc_c"}

    def test_default_kc_without_kc_column(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        df = _df({"kc_a": ("s1", "s2")}).drop(columns=["kc_id"])
        model.fit(df)
        captured = _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="'default_kc' already exists"):
            model.fit(df, n_kcs_workers=2)
        assert captured == []


class TestLowMemory:
    @pytest.mark.parametrize("n_workers", [1, 3])
    def test_each_kc_is_evicted_and_metadata_consistent(self, monkeypatch, n_workers):
        model = StandardBKT(low_memory=True)
        _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=n_workers)
        assert model.fits.stan_fits == {}
        assert list(model.fits._fit_metadata.fit_saves.keys()) == list(KCS)
        assert model.fits.num_fitted_kcs == 3
        assert model.fits.get_fitted_kcs() == set(KCS)
        assert model._is_fitted

    def test_evicted_fits_reload_lazily_after_concurrent_fit(self, monkeypatch):
        import stanbkt.fits.core.base as fit_base_module

        model = StandardBKT(low_memory=True)
        _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=3)
        reloaded: list[str] = []

        def _fake_from_csv(path: str):
            reloaded.append(os.path.basename(path.rstrip("/")))
            return SimpleNamespace(path=path)

        monkeypatch.setattr(fit_base_module, "cmdstan_from_csv", _fake_from_csv)
        for kc in KCS:
            assert model.fits.get_fit(kc) is not None
        assert len(reloaded) == 3
        assert len(set(reloaded)) == 3


class TestLogging:
    def test_debug_logging_from_worker_threads_does_not_crash(
        self, monkeypatch, capsys
    ):
        model = StandardBKT(verbose=VerbosityLevel.DEBUG)

        def _logging_fit(data_dict, fit_options):
            model.log(f"in worker {data_dict['nStudents']}", VerbosityLevel.DEBUG)
            return _DummySavedFit(int(data_dict["nStudents"]))

        _mock_fit(model, monkeypatch, fake=_logging_fit)
        model.fit(_df(), n_kcs_workers=3)
        out = capsys.readouterr().out
        for kc in KCS:
            assert f"Fitting KC: {kc}" in out
            assert f"Finished fitting KC: {kc}" in out
        for n in (2, 3, 4):
            assert f"in worker {n}" in out

    def test_warn_level_stays_quiet(self, monkeypatch, capsys):
        model = StandardBKT(verbose=VerbosityLevel.WARN)
        _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=3)
        assert "Finished fitting" not in capsys.readouterr().out


class TestIndividualInitialKnowledge:
    @staticmethod
    def _individual(monkeypatch, cls, n_workers, strategy, **fit_kwargs):
        model = cls(
            individual_initial_knowledge=True, init_knowledge_strategy=strategy
        )
        captured = _mock_fit(model, monkeypatch)
        model.fit(_df(), n_kcs_workers=n_workers, **fit_kwargs)
        return model, captured

    @pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
    def test_correctness_only_individual_student_index(self, monkeypatch, cls):
        seq, _ = self._individual(
            monkeypatch, cls, 1, InitKnowledgeStrategy.CORRECTNESS_ONLY
        )
        conc, _ = self._individual(
            monkeypatch, cls, 3, InitKnowledgeStrategy.CORRECTNESS_ONLY
        )
        for kc in KCS:
            a = seq.fits.get_fit_save_entry(kc)
            b = conc.fits.get_fit_save_entry(kc)
            assert a.student2index == b.student2index
            assert a.student2index is not None
        assert conc.fits.get_fit_save_entry("kc_c").student2index == {
            "s1": 1,
            "s2": 2,
            "s3": 3,
            "s4": 4,
        }

    @pytest.mark.parametrize("cls", [StandardBKT, MultiBKT])
    def test_joint_covariates_per_kc(self, monkeypatch, cls):
        seq, seq_data = self._individual(
            monkeypatch,
            cls,
            1,
            InitKnowledgeStrategy.JOINT,
            student_covariates=_covariates(),
        )
        conc, conc_data = self._individual(
            monkeypatch,
            cls,
            3,
            InitKnowledgeStrategy.JOINT,
            student_covariates=_covariates(),
        )
        seq_by_n = {int(d["nStudents"]): d for d in seq_data}
        conc_by_n = {int(d["nStudents"]): d for d in conc_data}
        assert set(seq_by_n) == set(conc_by_n) == {2, 3, 4}
        for n in (2, 3, 4):
            np.testing.assert_array_equal(
                seq_by_n[n]["covariates"], conc_by_n[n]["covariates"]
            )
            assert conc_by_n[n]["covariates"].shape == (n, 2)
            assert conc_by_n[n]["nCovariates"] == 2
        for kc in KCS:
            a = seq.fits.get_fit_save_entry(kc)
            b = conc.fits.get_fit_save_entry(kc)
            assert a.student2index == b.student2index
            assert a.covariate_columns == b.covariate_columns == ("pretest", "age")

    def test_joint_missing_covariates_raises_before_any_fit(self, monkeypatch):
        model = StandardBKT(
            individual_initial_knowledge=True,
            init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
        )
        captured = _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="student_covariates"):
            model.fit(_df(), n_kcs_workers=3)
        assert captured == []


@pytest.mark.slow
@pytest.mark.parametrize("method", ["mle", "mcmc"])
def test_concurrent_fit_matches_sequential_with_real_stan(method):
    import logging

    from stanbkt.fits.fit_options import MCMCFitOptions, MLEFitOptions
    from stanbkt.utils.sim import sim_simple_BKT

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data = sim_simple_BKT(n_students=12, n_problems=6, n_kcs=4, rng_seed=3)
    if method == "mle":
        options = MLEFitOptions(seed=11)
    else:
        options = MCMCFitOptions(
            chains=1, iter_warmup=100, iter_sampling=100, seed=11, show_progress=False
        )

    def _fit(n_workers):
        model = StandardBKT(fit_method=FitMethod(method), verbose=VerbosityLevel.WARN)
        model.fit(data, stan_fit_options=options, n_kcs_workers=n_workers)
        return model

    seq = _fit(1)
    conc = _fit(-1)

    # KCs are recorded in order of appearance in the data
    kcs = list(dict.fromkeys(data["kc_id"].astype(str)))
    assert list(seq.fits.stan_fits) == list(conc.fits.stan_fits) == kcs
    from cmdstanpy import CmdStanMCMC, CmdStanMLE

    expected_type = CmdStanMLE if method == "mle" else CmdStanMCMC
    for kc in kcs:
        assert isinstance(seq.fits.get_fit(kc), expected_type)
        assert isinstance(conc.fits.get_fit(kc), expected_type)
        for name in ("pi_know", "learn", "forget", "guess", "slip"):
            np.testing.assert_array_equal(
                seq.fits.get_fit(kc).stan_variable(name),
                conc.fits.get_fit(kc).stan_variable(name),
            )

    # ESS per second depends on wall time, everything else is deterministic for a fixed seed
    def _summary(model):
        summary = model.summary()
        return summary.loc[:, [c for c in summary.columns if "/s" not in str(c)]]

    pd.testing.assert_frame_equal(_summary(seq), _summary(conc))
