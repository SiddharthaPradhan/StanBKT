"""Tests for MultiBKT — grouped Bayesian Knowledge Tracing model."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.multi import MultiBKT, _is_all_none
from stanbkt.models.priors import MultiPriors
from stanbkt.utils.data_utils import KCData
from stanbkt.utils.verbose import VerbosityLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grouped_df(n_groups: int = 2) -> pd.DataFrame:
    """Minimal long-format DataFrame with a group_id column."""
    rows = []
    for g in range(1, n_groups + 1):
        for s in range(1, 3):
            for t, p in enumerate(["p1", "p2"], start=1):
                rows.append(
                    {
                        "student_id": f"g{g}_s{s}",
                        "problem_id": p,
                        "correct": (s + t) % 2,
                        "group_id": f"group{g}",
                        "timestamp": t,
                    }
                )
    return pd.DataFrame(rows)


def _kc_data_with_groups(
    n_students: int = 4,
    n_problems: int = 3,
    n_groups: int = 2,
) -> KCData:
    """KCData with groups and group_2_index populated."""
    rng = np.random.default_rng(42)
    correctness = rng.integers(0, 2, size=(n_students, n_problems), dtype=np.int8)
    # 1-based group array: alternate students between groups
    groups = np.array([(i % n_groups) + 1 for i in range(n_students)], dtype=np.int32)
    group_2_index = {f"group{g}": g for g in range(1, n_groups + 1)}
    return KCData(
        correctness=correctness,
        student_inter_dict={},
        lengths=np.full(n_students, n_problems, dtype=np.int32),
        student_ids=[f"s{i}" for i in range(n_students)],
        problem_ids=[f"p{j}" for j in range(n_problems)],
        groups=groups,
        group_2_index=group_2_index,
    )


def _kc_data_no_groups(n_students: int = 3, n_problems: int = 4) -> KCData:
    """KCData without groups (simulates StandardBKT data)."""
    rng = np.random.default_rng(7)
    correctness = rng.integers(0, 2, size=(n_students, n_problems), dtype=np.int8)
    return KCData(
        correctness=correctness,
        student_inter_dict={},
        lengths=np.full(n_students, n_problems, dtype=np.int32),
        student_ids=[f"s{i}" for i in range(n_students)],
        problem_ids=[f"p{j}" for j in range(n_problems)],
    )


def _mock_fit_mcmc(n_groups: int = 2) -> MagicMock:
    """Mock MCMC fit returning shape (n_samples, n_groups) for each param."""
    rng = np.random.default_rng(0)
    n_samples = 50

    def stan_variable(name: str) -> np.ndarray:
        values = {
            "pi_know": rng.uniform(0.1, 0.9, size=(n_samples, n_groups)),
            "learn": rng.uniform(0.1, 0.5, size=(n_samples, n_groups)),
            "forget": rng.uniform(0.0, 0.2, size=(n_samples, n_groups)),
            "guess": rng.uniform(0.1, 0.4, size=(n_samples, n_groups)),
            "slip": rng.uniform(0.0, 0.3, size=(n_samples, n_groups)),
        }
        if name not in values:
            raise ValueError(f"Unknown param: {name}")
        return values[name]

    fit = MagicMock()
    fit.stan_variable = stan_variable
    return fit


def _mock_fit_mle(n_groups: int = 2) -> MagicMock:
    """Mock MLE fit returning shape (n_groups,) for each param."""

    def stan_variable(name: str) -> np.ndarray:
        values = {
            "pi_know": np.array([0.3, 0.6]),
            "learn": np.array([0.2, 0.4]),
            "forget": np.array([0.05, 0.1]),
            "guess": np.array([0.25, 0.15]),
            "slip": np.array([0.1, 0.2]),
        }
        if name not in values:
            raise ValueError(f"Unknown param: {name}")
        return values[name][:n_groups]

    fit = MagicMock()
    fit.stan_variable = stan_variable
    return fit


# ---------------------------------------------------------------------------
# _is_all_none helper
# ---------------------------------------------------------------------------


class TestIsAllNone:
    def test_none_is_all_none(self):
        assert _is_all_none(None) is True

    def test_list_of_nones_is_all_none(self):
        assert _is_all_none([None, None]) is True

    def test_list_with_float_is_not_all_none(self):
        assert _is_all_none([None, 1.0]) is False

    def test_float_is_not_all_none(self):
        assert _is_all_none(0.0) is False

    def test_empty_list_is_all_none(self):
        # vacuously true — no element is non-None
        assert _is_all_none([]) is True


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestMultiBKTInit:
    def test_default_fit_method_is_mcmc(self):
        model = MultiBKT()
        assert model._fit_method == FitMethod.MCMC

    def test_custom_fit_method(self):
        model = MultiBKT(fit_method=FitMethod.MLE)
        assert model._fit_method == FitMethod.MLE

    def test_default_verbose(self):
        assert MultiBKT().verbose == VerbosityLevel.INFO

    def test_custom_verbose(self):
        assert MultiBKT(verbose=VerbosityLevel.DEBUG).verbose == VerbosityLevel.DEBUG

    def test_is_fitted_false_on_init(self):
        assert MultiBKT()._is_fitted is False

    def test_fits_is_none_on_init(self):
        assert MultiBKT().fits is None

    def test_stan_model_is_none_on_init(self):
        assert MultiBKT()._stan_model is None

    def test_stan_compile_kwargs_default_empty(self):
        assert MultiBKT().stan_compile_kwargs == {}

    def test_cpp_compile_kwargs_default_empty(self):
        assert MultiBKT().cpp_compile_kwargs == {}


# ---------------------------------------------------------------------------
# Stan filename properties
# ---------------------------------------------------------------------------


class TestStanFilenames:
    def test_model_filename_ends_with_bkt_model_stan(self):
        assert MultiBKT()._stan_model_filename.endswith("BKT_model.stan")

    def test_hidden_filename_ends_with_hidden_states_stan(self):
        assert MultiBKT()._stan_hidden_filename.endswith("hidden_states.stan")

    def test_smoothed_filename_ends_with_smoothed_hidden_states_stan(self):
        assert MultiBKT()._stan_smoothed_hidden_filename.endswith(
            "smoothed_hidden_states.stan"
        )

    def test_all_filenames_are_strings(self):
        model = MultiBKT()
        assert isinstance(model._stan_model_filename, str)
        assert isinstance(model._stan_hidden_filename, str)
        assert isinstance(model._stan_smoothed_hidden_filename, str)


# ---------------------------------------------------------------------------
# _default_priors
# ---------------------------------------------------------------------------


class TestDefaultPriors:
    def test_returns_multi_priors_instance(self):
        assert isinstance(MultiBKT()._default_priors(), MultiPriors)

    def test_returned_priors_use_defaults(self):
        priors = MultiBKT()._default_priors()
        assert priors.learn_mu is not None
        assert isinstance(priors.learn_mu, float)


# ---------------------------------------------------------------------------
# _build_stan_data_dict
# ---------------------------------------------------------------------------


class TestBuildStanDataDict:
    def test_raises_without_groups(self):
        model = MultiBKT()
        kc_data = _kc_data_no_groups()
        with pytest.raises(ValueError, match="groups populated"):
            model._build_stan_data_dict(kc_data)

    def test_required_keys_present(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        result = model._build_stan_data_dict(kc_data)
        for key in (
            "nStudents",
            "nProblems",
            "correctness",
            "interaction_lengths",
            "nGroups",
            "groups",
            "individual_pi_know",
        ):
            assert key in result, f"Missing key: {key}"

    def test_individual_pi_know_matches_model_setting(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        result = model._build_stan_data_dict(kc_data)
        assert result["individual_pi_know"] == int(model.individual_initial_knowledge)

    def test_n_groups_matches_group_2_index(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(6, 4, 3)
        result = model._build_stan_data_dict(kc_data)
        assert result["nGroups"] == 3

    def test_n_students_correct(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(5, 2, 2)
        result = model._build_stan_data_dict(kc_data)
        assert result["nStudents"] == 5

    def test_n_problems_correct(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 7, 2)
        result = model._build_stan_data_dict(kc_data)
        assert result["nProblems"] == 7

    def test_groups_array_passed_through(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        result = model._build_stan_data_dict(kc_data)
        np.testing.assert_array_equal(result["groups"], kc_data.groups)

    def test_prior_lists_have_length_n_groups(self):
        model = MultiBKT()
        n_groups = 3
        kc_data = _kc_data_with_groups(6, 4, n_groups)
        result = model._build_stan_data_dict(kc_data)
        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            assert len(result[f"prior_{param}_mu"]) == n_groups
            assert len(result[f"prior_{param}_std"]) == n_groups

    def test_unif_prior_flags_present_and_binary(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        result = model._build_stan_data_dict(kc_data)
        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            flag = result[f"unif_prior_{param}"]
            assert flag in (0, 1), f"unif_prior_{param} must be 0 or 1, got {flag}"

    def test_informative_priors_set_unif_flag_to_zero(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        priors = MultiPriors(use_defaults=True)
        result = model._build_stan_data_dict(kc_data, priors=priors)
        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            assert result[f"unif_prior_{param}"] == 0

    def test_none_priors_set_unif_flag_to_one(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        priors = MultiPriors(use_defaults=False)  # all None
        result = model._build_stan_data_dict(kc_data, priors=priors)
        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            assert result[f"unif_prior_{param}"] == 1

    def test_dummy_std_is_positive_for_uniform_priors(self):
        """Dummy std placeholder must be > 0 (Stan constraint on prior_*_std)."""
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        priors = MultiPriors(use_defaults=False)
        result = model._build_stan_data_dict(kc_data, priors=priors)
        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            for val in result[f"prior_{param}_std"]:
                assert val > 0

    def test_scalar_priors_are_expanded_to_lists(self):
        """A single scalar prior should be broadcast to all groups."""
        model = MultiBKT()
        n_groups = 3
        kc_data = _kc_data_with_groups(6, 4, n_groups)
        priors = MultiPriors(learn_mu=0.5, learn_std=2.0, use_defaults=True)
        result = model._build_stan_data_dict(kc_data, priors=priors)
        # The scalar 0.5 should appear n_groups times
        assert result["prior_learn_mu"] == [0.5] * n_groups

    def test_per_group_prior_lists_passed_through(self):
        """Explicit per-group lists should be preserved."""
        model = MultiBKT()
        n_groups = 2
        kc_data = _kc_data_with_groups(4, 3, n_groups)
        priors = MultiPriors(
            learn_mu=[0.1, 0.9], learn_std=[1.0, 2.0], use_defaults=True
        )
        result = model._build_stan_data_dict(kc_data, priors=priors)
        assert result["prior_learn_mu"] == [0.1, 0.9]
        assert result["prior_learn_std"] == [1.0, 2.0]

    def test_correctness_passed_through(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        result = model._build_stan_data_dict(kc_data)
        np.testing.assert_array_equal(result["correctness"], kc_data.correctness)

    def test_interaction_lengths_passed_through(self):
        model = MultiBKT()
        kc_data = _kc_data_with_groups(4, 3, 2)
        result = model._build_stan_data_dict(kc_data)
        np.testing.assert_array_equal(result["interaction_lengths"], kc_data.lengths)


# ---------------------------------------------------------------------------
# _extract_group_param_estimates
# ---------------------------------------------------------------------------


class TestExtractGroupParamEstimates:
    def test_mcmc_mean_has_correct_shape(self):
        fit = _mock_fit_mcmc(n_groups=2)
        result = MultiBKT._extract_group_param_estimates(fit, "pi_know", 2, "mean")
        assert result.shape == (2,)

    def test_mcmc_median_has_correct_shape(self):
        fit = _mock_fit_mcmc(n_groups=2)
        result = MultiBKT._extract_group_param_estimates(fit, "learn", 2, "median")
        assert result.shape == (2,)

    def test_mle_returns_correct_values(self):
        fit = _mock_fit_mle(n_groups=2)
        result = MultiBKT._extract_group_param_estimates(fit, "pi_know", 2, "mean")
        np.testing.assert_allclose(result, [0.3, 0.6])

    def test_mcmc_mean_is_in_valid_range(self):
        fit = _mock_fit_mcmc(n_groups=3)
        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            result = MultiBKT._extract_group_param_estimates(fit, param, 3, "mean")
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_three_groups_mcmc(self):
        fit = _mock_fit_mcmc(n_groups=3)
        result = MultiBKT._extract_group_param_estimates(fit, "forget", 3, "mean")
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# _extract_bkt_params_from_fit
# ---------------------------------------------------------------------------


class TestExtractBktParamsFromFit:
    def test_returns_five_arrays(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        groups = np.array([1, 1, 2, 2], dtype=np.int32)
        result = model._extract_bkt_params_from_fit(fit, n_students=4, groups=groups)
        assert len(result) == 5

    def test_output_shape_matches_n_students(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        groups = np.array([1, 2, 1, 2, 1], dtype=np.int32)
        for arr in model._extract_bkt_params_from_fit(fit, n_students=5, groups=groups):
            assert arr.shape == (5,)

    def test_group1_students_get_group1_params(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        # All students in group 1
        groups = np.array([1, 1, 1], dtype=np.int32)
        prior, *_ = model._extract_bkt_params_from_fit(fit, n_students=3, groups=groups)
        np.testing.assert_allclose(prior, [0.3, 0.3, 0.3])

    def test_group2_students_get_group2_params(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        groups = np.array([2, 2], dtype=np.int32)
        prior, *_ = model._extract_bkt_params_from_fit(fit, n_students=2, groups=groups)
        np.testing.assert_allclose(prior, [0.6, 0.6])

    def test_mixed_groups_map_correctly(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        groups = np.array([1, 2, 1], dtype=np.int32)
        prior, *_ = model._extract_bkt_params_from_fit(fit, n_students=3, groups=groups)
        np.testing.assert_allclose(prior, [0.3, 0.6, 0.3])

    def test_no_groups_broadcasts_first_group(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        prior, *_ = model._extract_bkt_params_from_fit(fit, n_students=3, groups=None)
        # Should broadcast group-1 (index 0) value to all students
        np.testing.assert_allclose(prior, [0.3, 0.3, 0.3])

    def test_all_five_params_differ_between_groups(self):
        model = MultiBKT()
        fit = _mock_fit_mle(n_groups=2)
        groups = np.array([1, 2], dtype=np.int32)
        results = model._extract_bkt_params_from_fit(fit, n_students=2, groups=groups)
        param_names = ("pi_know", "learn", "forget", "guess", "slip")
        expected_g1 = [0.3, 0.2, 0.05, 0.25, 0.1]
        expected_g2 = [0.6, 0.4, 0.1, 0.15, 0.2]
        for arr, name, v1, v2 in zip(results, param_names, expected_g1, expected_g2):
            assert arr[0] == pytest.approx(v1), f"{name} group-1 mismatch"
            assert arr[1] == pytest.approx(v2), f"{name} group-2 mismatch"


# ---------------------------------------------------------------------------
# fit — guards (no Stan compilation required)
# ---------------------------------------------------------------------------


class TestFitGuards:
    def test_fit_raises_without_group_id_column(self):
        model = MultiBKT()
        df_no_group = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2", "s2"],
                "problem_id": ["p1", "p2", "p1", "p2"],
                "correct": [1, 0, 0, 1],
                "timestamp": [1, 2, 1, 2],
            }
        )
        # iter_kc_data(return_groups=True) will raise because group_id is absent
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(df_no_group)

    def test_fit_raises_for_unexpected_kwarg(self):
        model = MultiBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            model.fit(_grouped_df(), method=FitMethod.VB)

    def test_fit_raises_if_stan_model_compilation_fails(self, monkeypatch):
        model = MultiBKT()

        def _bad_compile(path):
            model._stan_model = None

        monkeypatch.setattr(model, "_compile_model", _bad_compile)
        with pytest.raises(RuntimeError, match="compilation failed"):
            model.fit(_grouped_df())


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            MultiBKT().evaluate()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_raises_when_not_fitted(self):
        model = MultiBKT()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(_grouped_df())

    def test_raises_without_data(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        with pytest.raises(ValueError, match="'data' must be provided"):
            model.predict()

    def test_raises_invalid_point_estimate(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        with pytest.raises(ValueError, match="'point_estimate' must be"):
            model.predict(data=_grouped_df(), point_estimate="garbage")

    def test_returns_dataframe_with_correct_columns(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        model.fits = MagicMock()
        model.fits.get_fit.return_value = object()
        monkeypatch.setattr(
            model,
            "_extract_bkt_params_from_fit",
            lambda fit, n_students, point_estimate="mean", groups=None: (
                np.full(n_students, 0.3),
                np.full(n_students, 0.2),
                np.full(n_students, 0.05),
                np.full(n_students, 0.25),
                np.full(n_students, 0.1),
            ),
        )
        out = model.predict(data=_grouped_df())
        assert isinstance(out, pd.DataFrame)
        assert {
            "kc_id",
            "student_id",
            "problem_id",
            "pKnow",
            "pCorrectness",
            "correct",
        }.issubset(set(out.columns))

    def test_returns_empty_df_for_no_overlapping_kcs(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: set())
        model.fits = MagicMock()
        out = model.predict(data=_grouped_df())
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_group_params_are_used_per_student(self, monkeypatch):
        """Verify that different groups produce different pKnow values."""
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        model.fits = MagicMock()
        model.fits.get_fit.return_value = _mock_fit_mle(n_groups=2)

        out = model.predict(data=_grouped_df(n_groups=2))
        # Students from different groups should have different pKnow values
        # because their pi_know differs (0.3 vs 0.6)
        unique_pknow = out.groupby("student_id")["pKnow"].first()
        assert unique_pknow.nunique() > 1


# ---------------------------------------------------------------------------
# predict_smoothed
# ---------------------------------------------------------------------------


class TestPredictSmoothedStates:
    def test_raises_when_not_fitted(self):
        model = MultiBKT()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_smoothed(_grouped_df())

    def test_raises_without_data(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        with pytest.raises(ValueError, match="'data' must be provided"):
            model.predict_smoothed()

    def test_raises_invalid_point_estimate(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        with pytest.raises(ValueError, match="'point_estimate' must be"):
            model.predict_smoothed(data=_grouped_df(), point_estimate="garbage")

    def test_returns_dataframe_with_correct_columns(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)
        model.fits = MagicMock()
        model.fits.get_fit.return_value = object()
        monkeypatch.setattr(
            model,
            "_extract_bkt_params_from_fit",
            lambda fit, n_students, point_estimate="mean", groups=None: (
                np.full(n_students, 0.3),
                np.full(n_students, 0.2),
                np.full(n_students, 0.05),
                np.full(n_students, 0.25),
                np.full(n_students, 0.1),
            ),
        )
        out = model.predict_smoothed(data=_grouped_df())
        assert isinstance(out, pd.DataFrame)
        assert {
            "kc_id",
            "student_id",
            "problem_id",
            "pKnow",
            "pCorrectness",
            "correct",
        }.issubset(set(out.columns))

    def test_returns_empty_df_for_no_overlapping_kcs(self, monkeypatch):
        model = MultiBKT()
        monkeypatch.setattr(model, "_fit_check", lambda **kwargs: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: set())
        model.fits = MagicMock()
        out = model.predict_smoothed(data=_grouped_df())
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0
