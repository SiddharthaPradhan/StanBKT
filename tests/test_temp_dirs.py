"""Tests for temp file locations, work dirs and low memory eviction (mock based, no CmdStan)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_factory import FitFactory
from stanbkt.fits.fit_options import MCMCFitOptions
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils import temp_dirs
from stanbkt.utils.model_io import load_model
from stanbkt.utils.posterior_utils import release_gq_draws
from stanbkt.utils.sim import sim_simple_BKT
from stanbkt.utils.temp_dirs import (
    LOCK_FILE,
    WORK_DIR_PREFIX,
    WorkDir,
    resolve_temp_root,
    run_dir_of,
    sweep_stale_work_dirs,
    _filesystem_type,
    ram_temp_notice,
)


def _df(kcs=("kc_a",)) -> pd.DataFrame:
    rows = []
    for kc in kcs:
        for s in ("s1", "s2"):
            for t, (p, c) in enumerate([("p1", 1), ("p2", 0)], start=1):
                rows.append(
                    {
                        "student_id": s,
                        "problem_id": p,
                        "correct": c,
                        "timestamp": t,
                        "kc_id": kc,
                    }
                )
    return pd.DataFrame(rows)


class _FakeFit:
    """Holds one CSV in a run folder, moved like a CmdStan fit moves its files."""

    def __init__(self, csv_file: str):
        self.runset = SimpleNamespace(csv_files=[csv_file])

    def save_csvfiles(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, os.path.basename(self.runset.csv_files[0]))
        shutil.move(self.runset.csv_files[0], target)
        self.runset.csv_files[0] = target


def _mock_fit(model, monkeypatch):
    """Patch compile and Stan fitting so each fit writes one uniquely named CSV to a run dir."""
    calls = {"n": 0}

    def _fake(data_dict, fit_options):
        calls["n"] += 1
        run_dir = model._get_work_dir().new_run_dir("fit")
        csv_file = os.path.join(run_dir, f"run_{calls['n']}.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("lp__\n0\n")
        with open(os.path.join(run_dir, "stdout.txt"), "w", encoding="utf-8") as f:
            f.write("log\n")
        return _FakeFit(csv_file)

    monkeypatch.setattr(
        model, "_compile_model", lambda _: setattr(model, "_stan_model", object())
    )
    monkeypatch.setattr(model, "_fit_stan_model_using_method", _fake)
    monkeypatch.setattr(
        FitMethod,
        "infer_fit_method_from_stan_fit",
        staticmethod(lambda _: FitMethod.MCMC),
    )


def _files_under(path) -> list[str]:
    return [
        os.path.relpath(os.path.join(root, name), path)
        for root, _, names in os.walk(path)
        for name in names
    ]


def _age(work_dir_path, seconds=temp_dirs.SWEEP_GRACE_SECONDS + 10):
    past = time.time() - seconds
    os.utime(os.path.join(work_dir_path, LOCK_FILE), (past, past))


def _dead_work_dir(root) -> str:
    """A work dir whose owner is gone: it has an unlocked lock file past the grace period."""
    path = os.path.join(root, f"{WORK_DIR_PREFIX}dead")
    os.makedirs(os.path.join(path, "fit_x"))
    with open(os.path.join(path, LOCK_FILE), "w", encoding="utf-8"):
        pass
    with open(os.path.join(path, "fit_x", "a.csv"), "w", encoding="utf-8"):
        pass
    _age(path)
    return path


class TestResolveTempRoot:
    def test_explicit_temp_dir_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TMPDIR", str(tmp_path / "env"))
        target = tmp_path / "chosen" / "nested"
        assert resolve_temp_root(target, low_memory=True) == str(target)
        assert target.is_dir()

    def test_tmpdir_used_before_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TMPDIR", str(tmp_path / "env"))
        assert resolve_temp_root(None, low_memory=True) == str(tmp_path / "env")

    def test_low_memory_uses_cache_without_tmpdir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TMPDIR")
        cache = tmp_path / "cache" / "tmp"
        monkeypatch.setattr(temp_dirs, "default_cache_root", lambda: cache)
        assert resolve_temp_root(None, low_memory=True) == str(cache)

    def test_regular_model_uses_system_temp_without_tmpdir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TMPDIR")
        monkeypatch.setattr(temp_dirs.tempfile, "gettempdir", lambda: str(tmp_path))
        assert resolve_temp_root(None, low_memory=False) == str(tmp_path)

    def test_unusable_path_raises(self, tmp_path):
        blocker = tmp_path / "a_file"
        blocker.write_text("x")
        with pytest.raises(ValueError, match="Cannot write temporary files"):
            resolve_temp_root(blocker / "sub")


class TestWorkDir:
    def test_run_dirs_are_unique_under_concurrent_calls(self, tmp_path):
        work_dir = WorkDir(tmp_path)
        with ThreadPoolExecutor(max_workers=8) as executor:
            paths = list(executor.map(lambda _: work_dir.new_run_dir("fit"), range(64)))
        assert len(set(paths)) == 64

    def test_remove_run_dir_ignores_paths_outside(self, tmp_path):
        (tmp_path / "root").mkdir()
        work_dir = WorkDir(tmp_path / "root")
        outside = tmp_path / "outside"
        outside.mkdir()
        work_dir.remove_run_dir(str(outside))
        work_dir.remove_run_dir(work_dir.path)
        work_dir.remove_run_dir(None)
        assert outside.is_dir()
        assert os.path.isdir(work_dir.path)

        run_dir = work_dir.new_run_dir("gq")
        work_dir.remove_run_dir(run_dir)
        assert not os.path.exists(run_dir)

    def test_folder_removed_on_cleanup_and_collection(self, tmp_path):
        first = WorkDir(tmp_path)
        first.cleanup()
        assert not os.path.exists(first.path)

        second = WorkDir(tmp_path)
        path = second.path
        del second
        gc.collect()
        assert not os.path.exists(path)

    def test_keep_alive_outlives_the_owner(self, tmp_path):
        work_dir = WorkDir(tmp_path)
        path = work_dir.path

        class _Output:
            pass

        output = _Output()
        work_dir.keep_alive(output)
        del work_dir
        gc.collect()
        assert os.path.isdir(path)

        del output
        gc.collect()
        assert not os.path.exists(path)

    def test_run_dir_of_tolerates_objects_without_a_runset(self, tmp_path):
        assert run_dir_of(None) is None
        assert run_dir_of(object()) is None
        csv_file = str(tmp_path / "x.csv")
        assert run_dir_of(_FakeFit(csv_file)) == str(tmp_path)


class TestSweep:
    def test_removes_folder_with_dead_owner(self, tmp_path):
        path = _dead_work_dir(tmp_path)
        assert sweep_stale_work_dirs(tmp_path) == [path]
        assert not os.path.exists(path)

    def test_keeps_live_owner(self, tmp_path):
        work_dir = WorkDir(tmp_path)
        _age(work_dir.path)
        assert sweep_stale_work_dirs(tmp_path) == []
        assert os.path.isdir(work_dir.path)

    def test_keeps_folder_in_grace_period(self, tmp_path):
        path = _dead_work_dir(tmp_path)
        _age(path, seconds=0)
        assert sweep_stale_work_dirs(tmp_path) == []
        assert os.path.isdir(path)

    def test_keeps_everything_that_is_not_a_work_dir(self, tmp_path):
        no_lock = tmp_path / f"{WORK_DIR_PREFIX}nolock"
        no_lock.mkdir()
        compiled = tmp_path / "compiled_stan"
        compiled.mkdir()
        (compiled / LOCK_FILE).write_text("")
        _age(compiled)
        assert sweep_stale_work_dirs(tmp_path) == []
        assert no_lock.is_dir()
        assert compiled.is_dir()

    def test_missing_root_is_fine(self, tmp_path):
        assert sweep_stale_work_dirs(tmp_path / "missing") == []

    def test_unlockable_folder_is_left_alone(self, monkeypatch, tmp_path):
        path = _dead_work_dir(tmp_path)
        monkeypatch.setattr(temp_dirs, "_try_lock", lambda fd: False)
        assert sweep_stale_work_dirs(tmp_path) == []
        assert os.path.isdir(path)

    def test_creating_a_work_dir_sweeps_its_root(self, tmp_path):
        path = _dead_work_dir(tmp_path)
        WorkDir(tmp_path)
        assert not os.path.exists(path)

    @pytest.mark.skipif(os.name == "nt", reason="SIGKILL is POSIX only")
    def test_killed_process_is_swept(self, tmp_path):
        script = textwrap.dedent(
            f"""
            import time
            from stanbkt.utils.temp_dirs import WorkDir
            work_dir = WorkDir({str(tmp_path)!r})
            print(work_dir.path, flush=True)
            time.sleep(120)
            """
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )
        assert process.stdout is not None
        path = process.stdout.readline().strip()
        try:
            _age(path)
            assert sweep_stale_work_dirs(tmp_path) == []
        finally:
            process.send_signal(signal.SIGKILL)
            process.wait()
            process.stdout.close()

        assert os.path.isdir(path)
        assert sweep_stale_work_dirs(tmp_path) == [path]
        assert not os.path.exists(path)


class TestImportSweep:
    def _run_import(self, env_cache: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "TMPDIR"}
        env["XDG_CACHE_HOME"] = env_cache
        code = (
            "import threading, stanbkt\n"
            "[t.join() for t in threading.enumerate() if t.name == 'stanbkt-temp-sweep']\n"
        )
        return subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )

    @pytest.mark.skipif(sys.platform != "linux", reason="XDG cache paths are Linux only")
    def test_import_removes_stale_folder_but_not_compiled_models(self, tmp_path):
        tmp_root = tmp_path / "stanbkt" / "tmp"
        tmp_root.mkdir(parents=True)
        path = _dead_work_dir(tmp_root)
        compiled = tmp_path / "stanbkt" / "compiled_stan" / "m"
        compiled.mkdir(parents=True)

        result = self._run_import(str(tmp_path))

        assert result.returncode == 0, result.stderr
        assert not os.path.exists(path)
        assert compiled.is_dir()

    @pytest.mark.skipif(sys.platform != "linux", reason="XDG cache paths are Linux only")
    def test_import_works_when_cache_root_is_missing(self, tmp_path):
        result = self._run_import(str(tmp_path / "never_created"))
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "never_created").exists()

    @pytest.mark.skipif(
        os.name == "nt" or os.geteuid() == 0, reason="needs enforced file permissions"
    )
    def test_import_works_when_cache_root_is_read_only(self, tmp_path):
        tmp_root = tmp_path / "stanbkt" / "tmp"
        tmp_root.mkdir(parents=True)
        path = _dead_work_dir(tmp_root)
        os.chmod(path, 0o500)
        os.chmod(tmp_root, 0o500)
        try:
            result = self._run_import(str(tmp_path))
        finally:
            os.chmod(tmp_root, 0o700)
            os.chmod(path, 0o700)
        assert result.returncode == 0, result.stderr


class TestReservedFitOptions:
    @pytest.mark.parametrize("key", ["output_dir", "gq_output_dir"])
    def test_rejected_in_dataclass(self, key):
        with pytest.raises(ValueError, match="temp_dir"):
            FitFactory.verify_fit_options_compatibility(
                MCMCFitOptions(extra_kwargs={key: "/x"}), FitMethod.MCMC
            )

    def test_rejected_in_dict_through_fit(self, monkeypatch):
        model = StandardBKT()
        _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="temp_dir"):
            model.fit(_df(), stan_fit_options={"output_dir": "/x"})


class TestModelTempDir:
    def test_temp_dir_is_used_and_not_saved(self, monkeypatch, tmp_path):
        model = StandardBKT(low_memory=True, temp_dir=tmp_path / "chosen")
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        assert os.path.dirname(model._get_work_dir().path) == str(tmp_path / "chosen")
        assert "temp_dir" not in model._get_model_init_kwargs()

    def test_stan_runs_get_their_own_output_folder(self, tmp_path):
        model = StandardBKT(temp_dir=tmp_path)
        seen = []

        class _StanModel:
            def sample(self, data, output_dir, **kwargs):
                seen.append(output_dir)
                return SimpleNamespace()

        model._stan_model = _StanModel()
        for _ in range(2):
            model._fit_stan_model_using_method({}, MCMCFitOptions())
        assert len(set(seen)) == 2
        assert all(os.path.dirname(p) == model._get_work_dir().path for p in seen)

    def test_generated_quantities_get_their_own_output_folder(self, monkeypatch, tmp_path):
        model = StandardBKT(temp_dir=tmp_path)
        monkeypatch.setattr(model, "_build_stan_data_dict", lambda *a: {})
        monkeypatch.setattr(model, "_augment_stan_data_for_predict", lambda *a: {})
        seen = []

        class _GQModel:
            def generate_quantities(self, data, previous_fit, gq_output_dir):
                seen.append(gq_output_dir)
                return SimpleNamespace()

        for _ in range(2):
            model._generate_single_kc_quantities("kc", None, None, None, _GQModel())
        assert len(set(seen)) == 2
        assert all(os.path.dirname(p) == model._get_work_dir().path for p in seen)

    def test_remove_gq_files_only_touches_own_runs(self, tmp_path):
        model = StandardBKT(temp_dir=tmp_path / "root")
        run_dir = model._get_work_dir().new_run_dir("gq")
        csv_file = os.path.join(run_dir, "gq.csv")
        open(csv_file, "w", encoding="utf-8").close()
        model._remove_gq_files(_FakeFit(csv_file))
        assert not os.path.exists(run_dir)

        kept = tmp_path / "user_saved"
        kept.mkdir()
        (kept / "gq.csv").write_text("")
        model._remove_gq_files(_FakeFit(str(kept / "gq.csv")))
        assert (kept / "gq.csv").exists()

    def test_refit_without_low_memory_removes_the_old_run(self, monkeypatch, tmp_path):
        model = StandardBKT(temp_dir=tmp_path)
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        old_run = run_dir_of(model.fits.stan_fits["kc_a"])
        model.fit(_df(), overwrite_kcs=True)
        assert not os.path.exists(old_run)
        assert os.path.exists(model.fits.stan_fits["kc_a"].runset.csv_files[0])


class TestLowMemoryEviction:
    def test_eviction_keeps_one_copy_and_removes_the_run_folder(self, monkeypatch, tmp_path):
        model = StandardBKT(low_memory=True, temp_dir=tmp_path)
        _mock_fit(model, monkeypatch)
        model.fit(_df(("kc_a", "kc_b")))

        csvs = [f for f in _files_under(model._get_work_dir().path) if f.endswith(".csv")]
        assert len(csvs) == 2
        assert len({os.path.basename(f) for f in csvs}) == 2
        assert not [f for f in _files_under(model._get_work_dir().path) if "stdout" in f]
        assert model.fits.stan_fits == {}

    def test_refit_replaces_the_evicted_csvs(self, monkeypatch, tmp_path):
        model = StandardBKT(low_memory=True, temp_dir=tmp_path)
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        model.fit(_df(), overwrite_kcs=True)

        folder = model.fits._persisted_fit_folder("kc_a")
        assert sorted(os.listdir(folder)) == ["run_2.csv"]

    def test_get_fit_then_release_is_harmless(self, monkeypatch, tmp_path):
        import stanbkt.fits.core.base as fit_base_module

        model = StandardBKT(low_memory=True, temp_dir=tmp_path)
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        folder = model.fits._persisted_fit_folder("kc_a")
        reloaded = _FakeFit(os.path.join(folder, "run_1.csv"))
        monkeypatch.setattr(fit_base_module, "cmdstan_from_csv", lambda _: reloaded)

        model.fits.get_fit("kc_a")
        model.fits.release_fit_from_memory("kc_a")
        assert os.listdir(folder) == ["run_1.csv"]
        assert model.fits.stan_fits == {}

    def test_releasing_a_resident_reloaded_fit_keeps_its_csvs(self, monkeypatch, tmp_path):
        model = StandardBKT(low_memory=True, temp_dir=tmp_path)
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        folder = model.fits._persisted_fit_folder("kc_a")
        model.fits.stan_fits["kc_a"] = _FakeFit(os.path.join(folder, "run_1.csv"))

        model.fits.release_fit_from_memory("kc_a")
        assert os.listdir(folder) == ["run_1.csv"]
        assert model.fits.stan_fits == {}

    def test_save_leaves_evicted_csvs_in_place(self, monkeypatch, tmp_path):
        model = StandardBKT(low_memory=True, temp_dir=tmp_path / "root")
        _mock_fit(model, monkeypatch)
        model.fit(_df())
        folder = model.fits._persisted_fit_folder("kc_a")
        model.save(tmp_path / "model.stanbktmod")
        assert os.listdir(folder) == ["run_1.csv"]
        # staging folder is gone
        assert not [
            d for d in os.listdir(model._get_work_dir().path) if d.startswith("stanbkt_save_")
        ]


class TestLoadedModel:
    def _saved_archive(self, monkeypatch, tmp_path, low_memory=True):
        model = StandardBKT(low_memory=low_memory)
        _mock_fit(model, monkeypatch)
        model.fit(_df(("kc_a",)))
        archive = tmp_path / "model.stanbktmod"
        model.save(archive)
        return archive

    def test_fit_on_loaded_model_keeps_earlier_kcs_reachable(self, monkeypatch, tmp_path):
        archive = self._saved_archive(monkeypatch, tmp_path)

        loaded = load_model(archive)
        _mock_fit(loaded, monkeypatch)
        loaded.fit(_df(("kc_b",)))

        assert loaded.fits.has_kc("kc_a")
        assert os.path.isdir(loaded.fits._persisted_fit_folder("kc_a"))
        assert os.path.isdir(loaded.fits._persisted_fit_folder("kc_b"))

    def test_load_extracts_into_temp_dir(self, monkeypatch, tmp_path):
        archive = self._saved_archive(monkeypatch, tmp_path)
        loaded = load_model(archive, temp_dir=tmp_path / "chosen")
        assert os.path.dirname(loaded._get_work_dir().path) == str(tmp_path / "chosen")
        assert loaded.temp_dir == str(tmp_path / "chosen")
        assert loaded.fits._persisted_fit_folder("kc_a").startswith(loaded._get_work_dir().path)

    def test_low_memory_can_be_overridden(self, monkeypatch, tmp_path):
        archive = self._saved_archive(monkeypatch, tmp_path)
        assert load_model(archive).low_memory is True
        # the fake CSV cannot be parsed for an eager load, which load_model warns about and skips
        with pytest.warns(UserWarning, match="Skipping this KC"):
            assert load_model(archive, low_memory=False).low_memory is False

    def test_override_enables_lazy_load_of_a_regular_save(self, monkeypatch, tmp_path):
        archive = self._saved_archive(monkeypatch, tmp_path, low_memory=False)
        loaded = load_model(archive, low_memory=True)
        assert loaded.low_memory is True
        assert loaded.fits.stan_fits == {}
        assert loaded.fits.has_kc("kc_a")

    def test_failed_load_removes_its_folder(self, tmp_path):
        import zipfile

        archive = tmp_path / "bad.stanbktmod"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("junk.txt", "x")
        root = tmp_path / "root"
        with pytest.raises(FileNotFoundError):
            load_model(archive, temp_dir=root)
        assert os.listdir(root) == []


class TestGQFilesFollowTheirObject:
    def _model_and_gq_model(self, monkeypatch, tmp_path):
        model = StandardBKT(temp_dir=tmp_path)
        monkeypatch.setattr(model, "_build_stan_data_dict", lambda *a: {})
        monkeypatch.setattr(model, "_augment_stan_data_for_predict", lambda *a: {})

        class _GQ:
            pass

        class _GQModel:
            def generate_quantities(self, data, previous_fit, gq_output_dir):
                gq = _GQ()
                gq.csv_file = os.path.join(gq_output_dir, "gq.csv")
                with open(gq.csv_file, "w", encoding="utf-8"):
                    pass
                return gq

        return model, _GQModel()

    def test_dropping_the_object_removes_its_files(self, monkeypatch, tmp_path):
        model, gq_model = self._model_and_gq_model(monkeypatch, tmp_path)
        _, kept, _ = model._generate_single_kc_quantities("a", None, None, None, gq_model)
        _, dropped, _ = model._generate_single_kc_quantities("b", None, None, None, gq_model)
        dropped_file = dropped.csv_file
        del dropped
        gc.collect()
        assert not os.path.exists(dropped_file)
        assert os.path.exists(dropped_file.replace(os.path.dirname(dropped_file), os.path.dirname(kept.csv_file)))

    def test_files_moved_out_survive(self, monkeypatch, tmp_path):
        model, gq_model = self._model_and_gq_model(monkeypatch, tmp_path / "root")
        _, gq, _ = model._generate_single_kc_quantities("a", None, None, None, gq_model)
        saved = tmp_path / "saved.csv"
        shutil.move(gq.csv_file, saved)
        run_dir = os.path.dirname(gq.csv_file)
        del gq
        gc.collect()
        assert saved.exists()
        assert not os.path.exists(run_dir)


class TestFilesystemType:
    @pytest.mark.skipif(sys.platform != "linux", reason="the mount table is Linux only")
    def test_uses_the_longest_mount_that_contains_the_path(self, tmp_path):
        mounts = tmp_path / "mounts"
        mounts.write_text(
            "/dev/sda1 / ext4 rw 0 0\n"
            "tmpfs /mnt/ram tmpfs rw 0 0\n"
            "/dev/sdb1 /mnt/ram/data xfs rw 0 0\n"
        )
        assert _filesystem_type("/mnt/ram/work", str(mounts)) == "tmpfs"
        assert _filesystem_type("/mnt/ram/data/work", str(mounts)) == "xfs"
        assert _filesystem_type("/mnt/ramfoo", str(mounts)) == "ext4"

    def test_missing_mount_table(self, tmp_path):
        assert _filesystem_type("/mnt/ram", str(tmp_path / "missing")) is None


class TestRamTempNotice:
    @pytest.fixture
    def on_tmpfs(self, monkeypatch):
        monkeypatch.setattr(temp_dirs, "_filesystem_type", lambda path: "tmpfs")
        monkeypatch.setattr(
            temp_dirs.shutil,
            "disk_usage",
            lambda _: SimpleNamespace(total=16e9, used=4e9, free=12e9),
        )

    def test_reports_folder_and_space(self, on_tmpfs):
        notice = ram_temp_notice("/tmp", None)
        assert notice is not None
        assert "'/tmp'" in notice
        assert "held in RAM" in notice
        assert "12.0 GB free of 16.0 GB" in notice
        assert "temp_dir" in notice

    def test_ramfs_counts(self, monkeypatch, on_tmpfs):
        monkeypatch.setattr(temp_dirs, "_filesystem_type", lambda path: "ramfs")
        assert ram_temp_notice("/x", None) is not None

    def test_quiet_when_chosen_with_temp_dir(self, on_tmpfs):
        assert ram_temp_notice("/tmp", "/tmp") is None

    @pytest.mark.parametrize("filesystem", ["ext4", "btrfs", None])
    def test_quiet_when_on_disk_or_unknown(self, monkeypatch, filesystem):
        monkeypatch.setattr(temp_dirs, "_filesystem_type", lambda path: filesystem)
        assert ram_temp_notice("/tmp", None) is None

    def test_failed_space_check_is_not_an_error(self, monkeypatch, on_tmpfs):
        def _raise(_):
            raise OSError

        monkeypatch.setattr(temp_dirs.shutil, "disk_usage", _raise)
        assert ram_temp_notice("/tmp", None) is None

    @pytest.mark.parametrize("verbose, printed", [(2, True), (1, False), (0, False)])
    def test_model_logs_once_at_info(self, capsys, on_tmpfs, verbose, printed):
        model = StandardBKT(verbose=verbose)
        model._get_work_dir()
        model._get_work_dir()
        assert capsys.readouterr().out.count("held in RAM") == (1 if printed else 0)

    def test_low_memory_on_tmpfs_is_reported_too(self, capsys, on_tmpfs):
        StandardBKT(low_memory=True)._get_work_dir()
        assert "held in RAM" in capsys.readouterr().out

    def test_quiet_for_model_with_temp_dir(self, capsys, on_tmpfs, tmp_path):
        StandardBKT(temp_dir=tmp_path)._get_work_dir()
        assert capsys.readouterr().out == ""


class TestGetFitUnderLowMemory:
    def _evicted_model(self, monkeypatch, low_memory=True):
        import stanbkt.fits.core.base as fit_base_module

        model = StandardBKT(low_memory=low_memory)
        _mock_fit(model, monkeypatch)
        model.fit(_df(("kc_a", "kc_b")))
        reads = []

        def _fake_from_csv(path):
            reads.append(path)
            return SimpleNamespace(path=path)

        monkeypatch.setattr(fit_base_module, "cmdstan_from_csv", _fake_from_csv)
        return model, reads

    def test_not_kept_by_the_model(self, monkeypatch):
        model, reads = self._evicted_model(monkeypatch)
        for kc in ("kc_a", "kc_b"):
            assert model.fits.get_fit(kc) is not None
        assert model.fits.stan_fits == {}
        assert len(reads) == 2

    def test_each_call_reads_from_disk(self, monkeypatch):
        model, reads = self._evicted_model(monkeypatch)
        model.fits.get_fit("kc_a")
        model.fits.get_fit("kc_a")
        assert len(reads) == 2

    def test_unknown_kc_raises(self, monkeypatch):
        model, _ = self._evicted_model(monkeypatch)
        with pytest.raises(KeyError, match="No fit found"):
            model.fits.get_fit("missing")

    def test_regular_loaded_fit_is_still_kept(self, tmp_path, monkeypatch):
        import stanbkt.fits.core.base as fit_base_module

        fits = StandardBKT().fits
        fits.set_fit_artifact_base_location(tmp_path)
        fits._fit_metadata.fit_saves["kc_a"] = SimpleNamespace(save_folder="kc_a")
        monkeypatch.setattr(fit_base_module, "cmdstan_from_csv", lambda _: SimpleNamespace())
        assert fits.get_fit("kc_a") is fits.get_fit("kc_a")
        assert "kc_a" in fits.stan_fits

    def test_transient_read_still_holds_one_fit_for_the_pass(self, monkeypatch):
        model, _ = self._evicted_model(monkeypatch)
        model.fits._get_fit_transient("kc_a")
        assert set(model.fits.stan_fits) == {"kc_a"}
        model.fits._get_fit_transient("kc_b")
        assert set(model.fits.stan_fits) == {"kc_b"}
        model.fits._release_transient_fits()
        assert model.fits.stan_fits == {}


class TestReleaseGQDraws:
    def test_clears_when_csvs_exist(self, tmp_path):
        csv_file = tmp_path / "gq.csv"
        csv_file.write_text("")
        gq = SimpleNamespace(
            runset=SimpleNamespace(csv_files=[str(csv_file)]), _draws=np.ones((2, 2))
        )
        release_gq_draws(gq)
        assert gq._draws.shape == (0,)

    def test_kept_when_csvs_are_gone(self, tmp_path):
        gq = SimpleNamespace(
            runset=SimpleNamespace(csv_files=[str(tmp_path / "gone.csv")]),
            _draws=np.ones((2, 2)),
        )
        release_gq_draws(gq)
        assert gq._draws.shape == (2, 2)

    def test_object_without_a_runset_is_left_alone(self):
        gq = SimpleNamespace(_draws=np.ones((2, 2)))
        release_gq_draws(gq)
        assert gq._draws.shape == (2, 2)


@pytest.mark.slow
class TestLowMemoryWithCmdStan:
    """Real CmdStan checks, including the CmdStanGQ._draws cache that is private to CmdStanPy."""

    @pytest.fixture(scope="class")
    def data(self):
        return sim_simple_BKT(n_students=20, n_problems=10, n_kcs=2, rng_seed=1)

    @pytest.fixture(scope="class")
    def model(self, data, tmp_path_factory):
        model = StandardBKT(low_memory=True, temp_dir=tmp_path_factory.mktemp("lm"))
        model.fit(
            data,
            stan_fit_options=MCMCFitOptions(
                chains=2,
                parallel_chains=2,
                iter_warmup=100,
                iter_sampling=50,
                seed=3,
                show_progress=False,
            ),
        )
        return model

    def test_get_fit_is_usable_and_not_kept(self, model):
        for kc in sorted(model.fits.get_fitted_kcs()):
            assert model.fits.get_fit(kc).stan_variable("guess").shape[0] == 100
        assert model.fits.stan_fits == {}

    def test_summary_releases_cached_draws_and_they_reread_the_same(self, model, data):
        gq = model.predict_posterior_stan(data)
        before = {kc: g.stan_variable("pKnow").copy() for kc, g in gq.items()}
        assert all(g._draws.size > 0 for g in gq.values())

        first = model.predict_posterior_summary(data, stan_output=gq)
        assert all(g._draws.size == 0 for g in gq.values())

        for kc, g in gq.items():
            np.testing.assert_array_equal(g.stan_variable("pKnow"), before[kc])
        second = model.predict_posterior_summary(data, stan_output=gq)
        pd.testing.assert_frame_equal(first, second)

    def test_draws_output_releases_cached_draws(self, model, data):
        gq = model.predict_posterior_stan(data)
        model.predict_posterior_draws(data, stan_output=gq)
        assert all(g._draws.size == 0 for g in gq.values())


def test_unreadable_fit_folder_raises_a_clear_error(monkeypatch, tmp_path):
    import stanbkt.fits.core.base as fit_base_module

    model = StandardBKT(low_memory=True, temp_dir=tmp_path)
    _mock_fit(model, monkeypatch)
    model.fit(_df())
    monkeypatch.setattr(fit_base_module, "cmdstan_from_csv", lambda _: None)
    with pytest.raises(ValueError, match="Could not load the fit for KC 'kc_a'"):
        model.fits.get_fit("kc_a")
