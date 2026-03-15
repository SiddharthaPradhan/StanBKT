import os

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.standard import StandardBKT
from stanbkt.utils.verbose import VerbosityLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_df() -> pd.DataFrame:
    """Smallest valid long-format DataFrame (single KC inferred)."""
    return pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 0, 1],
        }
    )


def _correctness_array(n_students=3, n_problems=4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(n_students, n_problems), dtype=np.int8)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestStandardBKTInit:
    def test_hidden_states_model_is_none(self):
        model = StandardBKT()
        assert model._hidden_states_model is None

    def test_smoothed_hidden_states_model_is_none(self):
        model = StandardBKT()
        assert model._smoothed_hidden_states_model is None

    def test_is_fitted_is_false(self):
        model = StandardBKT()
        assert model._is_fitted is False

    def test_fits_is_none(self):
        model = StandardBKT()
        assert model.fits is None

    def test_default_verbose(self):
        model = StandardBKT()
        assert model.verbose == VerbosityLevel.INFO

    def test_custom_verbose(self):
        model = StandardBKT(verbose=VerbosityLevel.DEBUG)
        assert model.verbose == VerbosityLevel.DEBUG

    def test_stan_compile_kwargs_default_empty(self):
        model = StandardBKT()
        assert model.stan_compile_kwargs == {}

    def test_cpp_compile_kwargs_default_empty(self):
        model = StandardBKT()
        assert model.cpp_compile_kwargs == {}


# ---------------------------------------------------------------------------
# _build_stan_data_dict
# ---------------------------------------------------------------------------


class TestBuildStanDataDict:
    def test_contains_required_keys(self):
        model = StandardBKT()
        arr = _correctness_array(3, 4)
        result = model._build_stan_data_dict(arr)
        assert "nStudents" in result
        assert "nProblems" in result
        assert "correctness" in result
        assert "nGroups" in result
        assert "groups" in result

    def test_n_students_matches_array_rows(self):
        model = StandardBKT()
        arr = _correctness_array(5, 6)
        result = model._build_stan_data_dict(arr)
        assert result["nStudents"] == 5

    def test_n_problems_matches_array_cols(self):
        model = StandardBKT()
        arr = _correctness_array(5, 6)
        result = model._build_stan_data_dict(arr)
        assert result["nProblems"] == 6

    def test_correctness_is_passed_through(self):
        model = StandardBKT()
        arr = _correctness_array(3, 4)
        result = model._build_stan_data_dict(arr)
        np.testing.assert_array_equal(result["correctness"], arr)

    def test_n_groups_is_one(self):
        model = StandardBKT()
        result = model._build_stan_data_dict(_correctness_array(3, 4))
        assert result["nGroups"] == 1

    def test_groups_is_all_ones(self):
        model = StandardBKT()
        arr = _correctness_array(5, 3)
        result = model._build_stan_data_dict(arr)
        np.testing.assert_array_equal(result["groups"], np.ones(5, dtype=np.int32))

    def test_groups_length_matches_n_students(self):
        model = StandardBKT()
        arr = _correctness_array(7, 2)
        result = model._build_stan_data_dict(arr)
        assert len(result["groups"]) == 7

    def test_n_students_and_n_problems_are_ints(self):
        model = StandardBKT()
        result = model._build_stan_data_dict(_correctness_array(3, 4))
        assert isinstance(result["nStudents"], int)
        assert isinstance(result["nProblems"], int)


# ---------------------------------------------------------------------------
# fit — method-switching guard and unsupported-method guard
# (both checks occur before any Stan call)
# ---------------------------------------------------------------------------


class TestFitMethodGuards:
    def test_raises_for_variational_method(self):
        model = StandardBKT()
        with pytest.raises(ValueError, match="Only method='sample'"):
            model.fit(_minimal_df(), method=FitMethod.VB)

    def test_raises_for_optimize_method(self):
        model = StandardBKT()
        with pytest.raises(ValueError, match="Only method='sample'"):
            model.fit(_minimal_df(), method=FitMethod.MLE)

    def test_raises_for_pathfinder_method(self):
        model = StandardBKT()
        with pytest.raises(ValueError, match="Only method='sample'"):
            model.fit(_minimal_df(), method=FitMethod.PATHFINDER)

    def test_method_switch_raises_before_stan(self):
        """If a model was previously fitted (simulated), refit with a different method raises."""
        model = StandardBKT()
        # Simulate a previous fit with MCMC without actually running Stan
        model._previous_fit_method = FitMethod.MCMC
        with pytest.raises(ValueError, match="Refitting with a different method"):
            model.fit(_minimal_df(), method=FitMethod.VB)

    def test_method_switch_error_message_contains_previous_method(self):
        model = StandardBKT()
        model._previous_fit_method = FitMethod.MCMC
        with pytest.raises(ValueError) as exc_info:
            model.fit(_minimal_df(), method=FitMethod.VB)
        assert "previously fitted" in str(exc_info.value)

    def test_same_method_does_not_trigger_switch_guard(self):
        """Calling fit again with the same method should not raise the switch error."""
        model = StandardBKT()
        model._previous_fit_method = FitMethod.MCMC
        # Raises the "only sample" error, not the switch error, confirming the guard passed.
        # Since MCMC == "sample", it should proceed past both guards and attempt Stan —
        # at that point it will try to compile the Stan model, which we DON'T want.
        # Instead, confirm no ValueError about "Refitting with a different method" is raised.
        try:
            model.fit(_minimal_df(), method=FitMethod.MCMC)
        except ValueError as e:
            assert "Refitting with a different method" not in str(e)
        except Exception:
            pass  # Stan compilation or runtime error — acceptable, guards passed


# ---------------------------------------------------------------------------
# _fit_using_method — raises for unimplemented methods
# ---------------------------------------------------------------------------


class TestFitUsingMethod:
    def test_calling_causes_run_time_error(self):
        model = StandardBKT()
        with pytest.raises(RuntimeError):
            model._fit_using_method(FitMethod.VB, {})

    # TODO add more tests after


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_raises_not_implemented(self):
        model = StandardBKT()
        with pytest.raises(NotImplementedError):
            model.evaluate()


# ---------------------------------------------------------------------------
# predict / predict_smoothed_states — known bug: missing `self` parameter
# ---------------------------------------------------------------------------


class TestPredictMissingSelfBug:
    def test_predict_raises_typeerror_on_instance_call(self):
        """predict() is defined without `self`, so calling it on an instance raises TypeError.

        This is a known bug. The test documents current (broken) behaviour.
        It should be fixed by adding `self` to the function signature.
        """
        model = StandardBKT()
        with pytest.raises(TypeError):
            model.predict()

    def test_predict_smoothed_states_raises_typeerror_on_instance_call(self):
        """predict_smoothed_states() is defined without `self`, same bug as predict()."""
        model = StandardBKT()
        with pytest.raises(TypeError):
            model.predict_smoothed_states()


# ---------------------------------------------------------------------------
# predict_posterior / predict_smoothed_states_posterior — fit_check
# ---------------------------------------------------------------------------


class TestPredictPosteriorUnfitted:
    def test_predict_posterior_raises_when_not_fitted(self):
        model = StandardBKT()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_posterior(pd.DataFrame())

    def test_predict_smoothed_states_posterior_raises_when_not_fitted(self):
        model = StandardBKT()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_smoothed_states_posterior(pd.DataFrame())


# ---------------------------------------------------------------------------
# Stan file path properties — verify files exist on disk
# ---------------------------------------------------------------------------


class TestStanFilePaths:
    def test_stan_model_filename_is_string(self):
        model = StandardBKT()
        assert isinstance(model._stan_model_filename, str)

    def test_stan_model_file_exists(self):
        model = StandardBKT()
        assert os.path.isfile(
            model._stan_model_filename
        ), f"Stan model file not found: {model._stan_model_filename}"

    def test_stan_hidden_filename_is_string(self):
        model = StandardBKT()
        assert isinstance(model._stan_hidden_filename, str)

    def test_stan_hidden_file_exists(self):
        model = StandardBKT()
        assert os.path.isfile(
            model._stan_hidden_filename
        ), f"Hidden states Stan file not found: {model._stan_hidden_filename}"

    def test_stan_smoothed_hidden_filename_is_string(self):
        model = StandardBKT()
        assert isinstance(model._stan_smoothed_hidden_filename, str)

    def test_stan_smoothed_hidden_file_exists(self):
        model = StandardBKT()
        assert os.path.isfile(
            model._stan_smoothed_hidden_filename
        ), f"Smoothed hidden states Stan file not found: {model._stan_smoothed_hidden_filename}"
