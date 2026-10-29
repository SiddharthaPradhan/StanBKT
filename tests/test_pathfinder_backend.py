"""Pathfinder fits are rejected by the Stan backend with a clear message."""

from __future__ import annotations

import pandas as pd
import pytest

import stanbkt.models.predictions as predictions_module
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils.data_utils import ColumnNames


@pytest.fixture
def pathfinder_model(monkeypatch):
    model = StandardBKT(fit_method=FitMethod.PATHFINDER)
    monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
    monkeypatch.setattr(
        predictions_module,
        "_prepare_posterior_prediction_inputs",
        lambda model, data, column_mapping: (data, ColumnNames.apply_default_mapping(None)),
    )

    def _no_gq(*args, **kwargs):
        raise AssertionError("generate_quantities path should not be reached")

    monkeypatch.setattr(predictions_module, "_get_or_compile_gq_model", _no_gq)
    return model


@pytest.mark.parametrize(
    "method", ["predict_posterior_stan", "predict_smoothed_posterior_stan"]
)
def test_stan_output_points_to_draws_and_summary(pathfinder_model, method):
    with pytest.raises(ValueError, match="not available for Pathfinder fits.*backend='numba'"):
        getattr(pathfinder_model, method)(pd.DataFrame())


@pytest.mark.parametrize(
    "method",
    [
        "predict_posterior_draws",
        "predict_posterior_summary",
        "predict_smoothed_posterior_draws",
        "predict_smoothed_posterior_summary",
    ],
)
def test_stan_backend_points_to_numba(pathfinder_model, method):
    with pytest.raises(ValueError, match="does not support Pathfinder fits. Use backend='numba'"):
        getattr(pathfinder_model, method)(pd.DataFrame())
