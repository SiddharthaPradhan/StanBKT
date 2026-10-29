"""Prediction outputs use the column names given in column_mapping."""

from __future__ import annotations

import logging

import pytest

from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.base import BKTModelBase
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils import posterior_summary
from stanbkt.utils.data_utils import ColumnNames
from stanbkt.utils.sim import sim_simple_BKT

MAPPING = {
    ColumnNames.STUDENT_ID: "user_id",
    ColumnNames.PROBLEM_ID: "item_id",
    ColumnNames.CORRECTNESS: "is_correct",
    ColumnNames.ORDER: "order_id",
    ColumnNames.KC_ID: "skill_id",
}
MAPPED_ID_COLUMNS = ["skill_id", "user_id", "item_id", "is_correct"]


def test_empty_point_estimate_frame_uses_mapping():
    frame = BKTModelBase._empty_point_estimate_prediction_frame(MAPPING)
    assert [str(c) for c in frame.columns] == [
        "skill_id", "user_id", "item_id", "pKnow", "pCorrectness", "is_correct"
    ]


def test_empty_point_estimate_frame_defaults():
    frame = BKTModelBase._empty_point_estimate_prediction_frame()
    assert [str(c) for c in frame.columns][:3] == ["kc_id", "student_id", "problem_id"]


@pytest.mark.slow
def test_every_prediction_output_uses_mapped_names():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data = sim_simple_BKT(n_students=8, n_problems=6, n_kcs=2, rng_seed=1).rename(
        columns={
            "student_id": "user_id", "problem_id": "item_id", "correct": "is_correct",
            "timestamp": "order_id", "kc_id": "skill_id",
        }
    )
    model = StandardBKT(fit_method=FitMethod.MLE)
    model.fit(data, column_mapping=MAPPING, stan_fit_options={"show_console": False})

    frames = {
        "predict": model.predict(data, column_mapping=MAPPING),
        "predict_smoothed": model.predict_smoothed(data, column_mapping=MAPPING),
    }
    for backend in ("stan", "numba"):
        frames[f"summary_{backend}"] = model.predict_posterior_summary(
            data, column_mapping=MAPPING, backend=backend
        )
        draws = model.predict_posterior_draws(data, column_mapping=MAPPING, backend=backend)
        frames[f"draws_{backend}"] = next(iter(draws.values()))
        frames[f"posterior_summary_{backend}"] = posterior_summary(draws, col_mapping=MAPPING)

    for name, frame in frames.items():
        columns = [str(c) for c in frame.columns]
        for col in MAPPED_ID_COLUMNS:
            assert col in columns, f"{name} is missing '{col}': {columns}"
        for default in ("kc_id", "student_id", "problem_id", "correct"):
            assert default not in columns, f"{name} still has default '{default}'"
