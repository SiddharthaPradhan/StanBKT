from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_options import (
    MCMCFitOptions,
    MLEFitOptions,
    PFFitOptions,
    VBFitOptions,
)
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.multi import MultiBKT
from stanbkt.models.core.standard import StandardBKT


def _df():
    # s1 has 3 interactions, s2 has 1 and s3 has 2
    return pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s1", "s2", "s3", "s3"],
            "problem_id": ["p1", "p2", "p3", "p1", "p1", "p2"],
            "correct": [1, 0, 1, 1, 0, 0],
            "group_id": ["g1", "g1", "g1", "g2", "g2", "g2"],
            "timestamp": [1, 2, 3, 1, 1, 2],
        }
    )


def _capture_fit_data(model, monkeypatch):
    monkeypatch.setattr(
        model, "_compile_model", lambda _: setattr(model, "_stan_model", object())
    )
    monkeypatch.setattr(
        FitMethod, "infer_fit_method_from_stan_fit", staticmethod(lambda _: FitMethod.MCMC)
    )
    captured: list[dict] = []

    def _fake_fit(data_dict, fit_options):
        captured.append(data_dict)
        return MagicMock()

    monkeypatch.setattr(model, "_fit_stan_model_using_method", _fake_fit)
    return captured


def test_grainsize_defaults_to_automatic():
    for options_cls in (MCMCFitOptions, VBFitOptions, MLEFitOptions, PFFitOptions):
        assert options_cls().grainsize == -1


def test_automatic_grainsize_is_one_slice_on_a_single_thread():
    assert MCMCFitOptions().resolve_grainsize(1234) == 1234
    assert MLEFitOptions().resolve_grainsize(57) == 57
    assert VBFitOptions().resolve_grainsize(57) == 57
    assert PFFitOptions().resolve_grainsize(57) == 57


def test_automatic_grainsize_gives_about_four_slices_per_thread():
    assert MCMCFitOptions(threads_per_chain=4).resolve_grainsize(1000) == 63
    assert MCMCFitOptions(threads_per_chain=4).resolve_grainsize(1600) == 100
    assert PFFitOptions(num_threads=2).resolve_grainsize(80) == 10
    # never below one student per slice
    assert MCMCFitOptions(threads_per_chain=8).resolve_grainsize(3) == 1


def test_user_grainsize_is_used_as_given():
    assert MCMCFitOptions(grainsize=5, threads_per_chain=4).resolve_grainsize(1000) == 5
    assert MLEFitOptions(grainsize=1).resolve_grainsize(1000) == 1


@pytest.mark.parametrize("grainsize", [0, -2, 1.5, True, "4"])
def test_invalid_grainsize_raises(grainsize):
    with pytest.raises(ValueError, match="grainsize"):
        MCMCFitOptions(grainsize=grainsize)


def test_grainsize_is_not_forwarded_to_cmdstanpy():
    assert "grainsize" not in MCMCFitOptions(grainsize=3).to_dict()
    assert "grainsize" not in PFFitOptions().to_dict()


def test_grainsize_is_read_from_a_dict():
    options = MCMCFitOptions.from_dict({"grainsize": 7, "chains": 2})
    assert options.grainsize == 7
    assert "grainsize" not in options.extra_kwargs


@pytest.mark.parametrize("model_cls", [StandardBKT, MultiBKT])
def test_stan_data_has_total_observations_and_single_slice(model_cls):
    from stanbkt.utils.data_utils import ColumnNames, iter_kc_data

    model = model_cls()
    mapping = ColumnNames.apply_default_mapping(None)
    _, kc_data = next(iter_kc_data(_df(), col_mapping=mapping, return_groups=model._use_groups))
    data_dict = model._build_stan_data_dict(kc_data, model._default_priors())
    assert data_dict["nObs"] == 6
    assert data_dict["nObs"] == int(np.sum(data_dict["interaction_lengths"]))
    assert data_dict["grainsize"] == data_dict["nStudents"] == 3


@pytest.mark.parametrize(
    "options, expected",
    [
        (None, 3),
        ({"threads_per_chain": 1}, 3),
        ({"grainsize": 2}, 2),
    ],
)
def test_fit_passes_resolved_grainsize_to_stan(monkeypatch, options, expected):
    model = StandardBKT()
    captured = _capture_fit_data(model, monkeypatch)
    model.fit(_df(), stan_fit_options=options, n_kcs_workers=1)
    assert captured[0]["grainsize"] == expected


def test_fit_uses_threads_for_automatic_grainsize(monkeypatch):
    model = StandardBKT(cpp_compile_kwargs={"STAN_THREADS": True})
    captured = _capture_fit_data(model, monkeypatch)
    model.fit(
        _df(),
        stan_fit_options=MCMCFitOptions(threads_per_chain=2, chains=1, parallel_chains=1),
        n_kcs_workers=1,
    )
    # 3 students, 4 slices per thread over 2 threads
    assert captured[0]["grainsize"] == 1
