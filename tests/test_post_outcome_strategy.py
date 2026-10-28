"""Structural tests for the post-test outcome regression feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils.data_utils import KCData, StudentInteraction


def _simple_data(n_students=4, n_problems=3):
    rows = []
    for s in range(n_students):
        for p in range(n_problems):
            rows.append(
                {"student_id": s, "problem_id": p, "correct": (s + p) % 2, "timestamp": p}
            )
    return pd.DataFrame(rows)


def _mock_fit(model, monkeypatch):
    monkeypatch.setattr(
        model, "_compile_model", lambda _: setattr(model, "_stan_model", object())
    )
    captured: list[dict] = []

    def _fake_fit(data_dict, fit_options):
        captured.append(data_dict)
        return MagicMock()

    monkeypatch.setattr(model, "_fit_stan_model_using_method", _fake_fit)
    monkeypatch.setattr(
        FitMethod,
        "infer_fit_method_from_stan_fit",
        staticmethod(lambda _: FitMethod.MCMC),
    )
    return captured


class TestFitValidation:
    def test_missing_post_outcome_raises(self, monkeypatch):
        model = StandardBKT(post_outcome_regression=True)
        _mock_fit(model, monkeypatch)
        with pytest.raises(ValueError, match="post_outcome"):
            model.fit(_simple_data())

    def test_post_outcome_missing_student_raises(self, monkeypatch):
        model = StandardBKT(post_outcome_regression=True)
        _mock_fit(model, monkeypatch)
        post_outcome = pd.DataFrame({"student_id": [0, 1, 2], "score": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="missing rows"):
            model.fit(_simple_data(), post_outcome=post_outcome)

    def test_post_outcome_multiple_outcome_columns_raises(self, monkeypatch):
        model = StandardBKT(post_outcome_regression=True)
        _mock_fit(model, monkeypatch)
        post_outcome = pd.DataFrame(
            {"student_id": [0, 1, 2, 3], "a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]}
        )
        with pytest.raises(ValueError, match="exactly one outcome column"):
            model.fit(_simple_data(), post_outcome=post_outcome)

    def test_disabled_by_default_ignores_post_outcome_kwarg(self, monkeypatch):
        # post_outcome_regression=False (default): fit succeeds even without post_outcome.
        model = StandardBKT()
        captured = _mock_fit(model, monkeypatch)
        model.fit(_simple_data())
        assert model.fits.has_kc("default_kc")
        assert captured[0]["post_outcome_regression"] == 0


class TestStanDataDict:
    def _kc_data(self, n_students=3, n_problems=2, post_outcome=None, post_covariates=None):
        correctness = np.ones((n_students, n_problems), dtype=np.int8)
        lengths = np.full(n_students, n_problems, dtype=np.int32)
        student_ids = [str(i) for i in range(n_students)]
        return KCData(
            correctness=correctness,
            student_inter_dict={
                sid: StudentInteraction(problem_ids=["0", "1"], length=n_problems)
                for sid in student_ids
            },
            lengths=lengths,
            student_ids=student_ids,
            problem_ids=["0", "1"],
            post_outcome=post_outcome,
            post_covariates=post_covariates,
            post_covariate_columns=(["cov"] if post_covariates is not None else None),
        )

    def test_disabled_flag_zero_sized(self):
        model = StandardBKT(post_outcome_regression=False)
        data_dict = model._build_stan_data_dict(self._kc_data(), model._default_priors())
        assert data_dict["post_outcome_regression"] == 0
        assert data_dict["nPostCovariates"] == 0
        assert data_dict["post_covariates"].shape == (3, 0)
        assert data_dict["unif_prior_post_b0"] == 1

    def test_enabled_flag_populates_fields(self):
        model = StandardBKT(post_outcome_regression=True)
        post_outcome = np.array([1.0, 2.0, 3.0])
        post_covariates = np.array([[0.1], [0.2], [0.3]])
        data_dict = model._build_stan_data_dict(
            self._kc_data(post_outcome=post_outcome, post_covariates=post_covariates),
            model._default_priors(),
        )
        assert data_dict["post_outcome_regression"] == 1
        assert data_dict["nPostCovariates"] == 1
        np.testing.assert_array_equal(data_dict["post_outcome"], post_outcome)
        np.testing.assert_array_equal(data_dict["post_covariates"], post_covariates)


class TestMetadataRoundTrip:
    def test_model_init_kwargs_include_flag(self):
        model = StandardBKT(post_outcome_regression=True)
        assert model._get_model_init_kwargs()["post_outcome_regression"] is True
