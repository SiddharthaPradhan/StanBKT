import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.standard import StandardBKT
from stanbkt.utils.data_utils import KCData, format_data
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


def _kc_data(n_students=3, n_problems=4) -> KCData:
    correctness = _correctness_array(n_students, n_problems)
    return KCData(
        correctness=correctness,
        student_inter_dict={},
        lengths=np.full(n_students, n_problems, dtype=np.int32),
        problem_ids=[f"p{problem_index + 1}" for problem_index in range(n_problems)],
    )


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
        kc_data = _kc_data(3, 4)
        result = model._build_stan_data_dict(kc_data)
        assert "nStudents" in result
        assert "nProblems" in result
        assert "correctness" in result
        assert "interaction_lengths" in result
        assert "nGroups" in result
        assert "groups" in result

    def test_n_students_matches_array_rows(self):
        model = StandardBKT()
        kc_data = _kc_data(5, 6)
        result = model._build_stan_data_dict(kc_data)
        assert result["nStudents"] == 5

    def test_n_problems_matches_array_cols(self):
        model = StandardBKT()
        kc_data = _kc_data(5, 6)
        result = model._build_stan_data_dict(kc_data)
        assert result["nProblems"] == 6

    def test_correctness_is_passed_through(self):
        model = StandardBKT()
        kc_data = _kc_data(3, 4)
        result = model._build_stan_data_dict(kc_data)
        np.testing.assert_array_equal(result["correctness"], kc_data.correctness)

    def test_interaction_lengths_are_passed_through(self):
        model = StandardBKT()
        kc_data = _kc_data(3, 4)
        result = model._build_stan_data_dict(kc_data)
        np.testing.assert_array_equal(result["interaction_lengths"], kc_data.lengths)

    def test_n_groups_is_one(self):
        model = StandardBKT()
        result = model._build_stan_data_dict(_kc_data(3, 4))
        assert result["nGroups"] == 1

    def test_groups_is_all_ones(self):
        model = StandardBKT()
        kc_data = _kc_data(5, 3)
        result = model._build_stan_data_dict(kc_data)
        np.testing.assert_array_equal(result["groups"], np.ones(5, dtype=np.int32))

    def test_groups_length_matches_n_students(self):
        model = StandardBKT()
        kc_data = _kc_data(7, 2)
        result = model._build_stan_data_dict(kc_data)
        assert len(result["groups"]) == 7

    def test_n_students_and_n_problems_are_ints(self):
        model = StandardBKT()
        result = model._build_stan_data_dict(_kc_data(3, 4))
        assert isinstance(result["nStudents"], int)
        assert isinstance(result["nProblems"], int)


# ---------------------------------------------------------------------------
# fit — method-switching guard and unsupported-method guard
# (both checks occur before any Stan call)
# ---------------------------------------------------------------------------


class TestFitMethodGuards:
    def test_raises_for_variational_method(self):
        model = StandardBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument 'method'"):
            model.fit(_minimal_df(), method=FitMethod.VB)

    def test_raises_for_optimize_method(self):
        model = StandardBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument 'method'"):
            model.fit(_minimal_df(), method=FitMethod.MLE)

    def test_raises_for_pathfinder_method(self):
        model = StandardBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument 'method'"):
            model.fit(_minimal_df(), method=FitMethod.PATHFINDER)

    def test_method_switch_raises_before_stan(self):
        """Legacy method kwarg path no longer exists; fit method is constructor-driven."""
        model = StandardBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument 'method'"):
            model.fit(_minimal_df(), method=FitMethod.VB)

    def test_method_switch_error_message_contains_previous_method(self):
        model = StandardBKT()
        with pytest.raises(TypeError) as exc_info:
            model.fit(_minimal_df(), method=FitMethod.VB)
        assert "unexpected keyword argument 'method'" in str(exc_info.value)

    def test_same_method_does_not_trigger_switch_guard(self):
        """fit() no longer accepts per-call method overrides."""
        model = StandardBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument 'method'"):
            model.fit(_minimal_df(), method=FitMethod.MCMC)


# ---------------------------------------------------------------------------
# _fit_using_method — raises for unimplemented methods
# ---------------------------------------------------------------------------


class TestFitUsingMethod:
    def test_calling_causes_run_time_error(self):
        model = StandardBKT()
        with pytest.raises(RuntimeError):
            model._fit_stan_model_using_method({}, MagicMock())

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
# predict / predict_smoothed_states
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_raises_when_not_fitted(self):
        model = StandardBKT()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(_minimal_df())

    def test_predict_rejects_output_kwarg(self):
        model = StandardBKT()
        with pytest.raises(TypeError, match="unexpected keyword argument 'output'"):
            model.predict(data=_minimal_df(), output="summary")

    def test_predict_rejects_invalid_point_estimate(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        with pytest.raises(ValueError, match="'point_estimate' must be either"):
            model.predict(data=_minimal_df(), point_estimate="mode")

    def test_predict_returns_default_output(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)

        model.fits = MagicMock()
        model.fits.get_fit.return_value = object()

        monkeypatch.setattr(
            model,
            "_extract_bkt_params_from_fit",
            lambda fit, n_students, point_estimate="mean": (
                np.full(n_students, 0.2),
                np.full(n_students, 0.3),
                np.full(n_students, 0.1),
                np.full(n_students, 0.2),
                np.full(n_students, 0.1),
            ),
        )

        out = model.predict(data=_minimal_df())
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == {
            "kc_id",
            "parameter",
            "student_id",
            "problem_id",
            "value",
        }
        assert set(out["kc_id"]) == {"default_kc"}
        assert set(out["parameter"]) == {
            "pKnow",
            "pCorrectness",
            "true_correctness",
        }

    def test_predict_smoothed_states_raises_when_not_fitted(self):
        model = StandardBKT()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_smoothed_states(_minimal_df())

    def test_predict_smoothed_states_requires_data(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        with pytest.raises(ValueError, match="'data' must be provided"):
            model.predict_smoothed_states()

    def test_predict_smoothed_states_rejects_invalid_point_estimate(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        with pytest.raises(ValueError, match="'point_estimate' must be either"):
            model.predict_smoothed_states(data=_minimal_df(), point_estimate="mode")

    def test_predict_smoothed_states_returns_dataframe(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)

        model.fits = MagicMock()
        model.fits.get_fit.return_value = object()

        monkeypatch.setattr(
            model,
            "_extract_bkt_params_from_fit",
            lambda fit, n_students, point_estimate="mean": (
                np.full(n_students, 0.2),
                np.full(n_students, 0.3),
                np.full(n_students, 0.1),
                np.full(n_students, 0.2),
                np.full(n_students, 0.1),
            ),
        )

        out = model.predict_smoothed_states(data=_minimal_df())
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == {
            "kc_id",
            "parameter",
            "student_id",
            "problem_id",
            "value",
        }
        assert set(out["kc_id"]) == {"default_kc"}
        assert set(out["parameter"]) == {"pKnow", "true_correctness"}

    def test_predict_smoothed_states_concatenates_multiple_kcs(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)

        model.fits = MagicMock()
        model.fits.get_fit.return_value = object()

        monkeypatch.setattr(
            model,
            "_extract_bkt_params_from_fit",
            lambda fit, n_students, point_estimate="mean": (
                np.full(n_students, 0.2),
                np.full(n_students, 0.3),
                np.full(n_students, 0.1),
                np.full(n_students, 0.2),
                np.full(n_students, 0.1),
            ),
        )

        multi_kc_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s1", "s1"],
                "problem_id": ["p1", "p2", "p1", "p2"],
                "correct": [1, 0, 0, 1],
                "kc_id": ["kc_a", "kc_a", "kc_b", "kc_b"],
            }
        )

        out = model.predict_smoothed_states(data=multi_kc_df)
        assert set(out["kc_id"]) == {"kc_a", "kc_b"}

    def test_predict_uses_original_ids_and_masks_padded_positions(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        monkeypatch.setattr(model, "check_data_contains_fitted_kcs", lambda kcs: None)
        monkeypatch.setattr(model, "get_kcs_in_fitted_kcs", lambda kcs: kcs)

        model.fits = MagicMock()
        model.fits.get_fit.return_value = object()

        monkeypatch.setattr(
            model,
            "_extract_bkt_params_from_fit",
            lambda fit, n_students, point_estimate="mean": (
                np.full(n_students, 0.2),
                np.full(n_students, 0.3),
                np.full(n_students, 0.1),
                np.full(n_students, 0.2),
                np.full(n_students, 0.1),
            ),
        )

        sparse_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "problem_id": ["p10", "p30", "p20"],
                "correct": [1, 0, 1],
            }
        )

        out = model.predict(data=sparse_df)
        p_know = out[out["parameter"] == "pKnow"].reset_index(drop=True)
        true_correctness = out[
            out["parameter"] == "true_correctness"
        ].reset_index(drop=True)

        assert p_know["student_id"].tolist() == ["s1", "s1", "s1", "s2", "s2", "s2"]
        assert p_know["problem_id"].tolist() == ["p10", "p30", "-1", "p20", "-1", "-1"]
        assert p_know["value"].tolist()[2] == -1.0
        assert p_know["value"].tolist()[4] == -1.0
        assert p_know["value"].tolist()[5] == -1.0
        assert true_correctness["value"].tolist() == [1.0, 0.0, -1.0, 1.0, -1.0, -1.0]


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


class TestPredictSmoothedStatesPosteriorApi:
    def test_raises_when_data_and_posterior_draws_missing(self):
        model = StandardBKT()
        with pytest.raises(ValueError, match="Either 'data' or 'posterior_draws'"):
            model.predict_smoothed_states_posterior()

    def test_returns_posterior_draws_for_default_output(self):
        model = StandardBKT()
        posterior_draws = {"kc_1": pd.DataFrame({"draw": [0.1, 0.2]})}
        out = model.predict_smoothed_states_posterior(posterior_draws=posterior_draws)
        assert out == posterior_draws

    def test_returns_summary_when_output_summary(self, monkeypatch):
        model = StandardBKT()
        posterior_draws = {"kc_1": pd.DataFrame({"draw": [0.1, 0.2]})}
        expected_summary = pd.DataFrame({"kc_id": ["kc_1"], "mean": [0.15]})

        def _fake_summarize(draws, quantiles):
            assert draws == posterior_draws
            assert quantiles == [0.05, 0.95]
            return expected_summary

        monkeypatch.setattr(
            StandardBKT, "_summarize_state_predictions", _fake_summarize
        )

        out = model.predict_smoothed_states_posterior(
            posterior_draws=posterior_draws,
            output="summary",
            summary_quantiles=[0.05, 0.95],
        )
        pd.testing.assert_frame_equal(out, expected_summary)

    def test_raises_when_output_stan_and_posterior_draws_provided(self):
        model = StandardBKT()
        posterior_draws = {"kc_1": pd.DataFrame({"draw": [0.1, 0.2]})}
        with pytest.raises(TypeError, match="cannot be used when 'output' is 'stan'"):
            model.predict_smoothed_states_posterior(
                posterior_draws=posterior_draws,
                output="stan",
            )

    def test_prefers_posterior_draws_when_both_inputs_are_provided(self):
        model = StandardBKT()
        posterior_draws = {"kc_1": pd.DataFrame({"draw": [0.1, 0.2]})}

        out = model.predict_smoothed_states_posterior(
            data=_minimal_df(),
            posterior_draws=posterior_draws,
        )

        assert out == posterior_draws


class TestPredictPosteriorDataPath:
    def test_predict_posterior_uses_process_helper_for_default_output(
        self, monkeypatch
    ):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._hidden_states_model = object()

        fake_stan = {"kc_1": MagicMock()}
        expected = {
            "kc_1": pd.DataFrame(
                {
                    "parameter": ["pred"],
                    "student_id": [1],
                    "problem_id": [1],
                    "value": [0.5],
                }
            )
        }

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (fake_stan, {}),
        )
        monkeypatch.setattr(StandardBKT, "_process_predict_gq", lambda _: expected)
        monkeypatch.setattr(
            StandardBKT,
            "_apply_kc_data_to_posterior_draws",
            lambda posterior_draws, kc_data_by_kc, drop_index_cols=True: posterior_draws,
        )

        out = model.predict_posterior(data=_minimal_df(), output="default")
        assert out == expected

    def test_predict_posterior_uses_summary_helper(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._hidden_states_model = object()

        fake_stan = {"kc_1": MagicMock()}
        processed = {
            "kc_1": pd.DataFrame(
                {
                    "parameter": ["pred"],
                    "student_id": [1],
                    "problem_id": [1],
                    "value": [0.5],
                }
            )
        }
        expected_summary = pd.DataFrame({"kc_id": ["kc_1"], "mean": [0.5]})

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (fake_stan, {}),
        )
        monkeypatch.setattr(StandardBKT, "_process_predict_gq", lambda _: processed)
        monkeypatch.setattr(
            StandardBKT,
            "_apply_kc_data_to_posterior_draws",
            lambda posterior_draws, kc_data_by_kc, drop_index_cols=True: posterior_draws,
        )

        def _fake_summary(draws, quantiles):
            assert draws == processed
            assert quantiles == [0.1, 0.9]
            return expected_summary

        monkeypatch.setattr(StandardBKT, "_summarize_state_predictions", _fake_summary)

        out = model.predict_posterior(
            data=_minimal_df(),
            output="summary",
            summary_quantiles=[0.1, 0.9],
        )
        pd.testing.assert_frame_equal(out, expected_summary)

    def test_predict_posterior_uses_original_ids_and_masks_padded_positions(
        self, monkeypatch
    ):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._hidden_states_model = object()

        class _FakeGQ:
            def __init__(self, draws_df: pd.DataFrame):
                self._draws_df = draws_df

            def draws_pd(self) -> pd.DataFrame:
                return self._draws_df

        gq_draws = pd.DataFrame(
            {
                "lp__": [0.0],
                "pKnow[1,1]": [0.1],
                "pKnow[1,2]": [0.2],
                "pKnow[1,3]": [0.3],
                "pKnow[2,1]": [0.4],
                "pKnow[2,2]": [0.5],
                "pKnow[2,3]": [0.6],
            }
        )

        sparse_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "problem_id": ["p10", "p30", "p20"],
                "correct": [1, 0, 1],
            }
        )
        kc_data_by_kc = format_data(sparse_df)

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (
                {"default_kc": _FakeGQ(gq_draws)},
                kc_data_by_kc,
            ),
        )

        out = model.predict_posterior(data=sparse_df, output="default")
        p_know = out["default_kc"]
        p_know = p_know[p_know["parameter"] == "pKnow"].reset_index(drop=True)
        true_correctness = out["default_kc"]
        true_correctness = true_correctness[
            true_correctness["parameter"] == "true_correctness"
        ].reset_index(drop=True)

        assert p_know["student_id"].tolist() == ["s1", "s1", "s1", "s2", "s2", "s2"]
        assert p_know["problem_id"].tolist() == ["p10", "p30", "-1", "p20", "-1", "-1"]
        assert p_know["value"].tolist()[2] == -1.0
        assert p_know["value"].tolist()[4] == -1.0
        assert p_know["value"].tolist()[5] == -1.0
        assert true_correctness["value"].tolist() == [1.0, 0.0, -1.0, 1.0, -1.0, -1.0]

    def test_predict_posterior_summary_includes_true_correctness(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._hidden_states_model = object()

        class _FakeGQ:
            def __init__(self, draws_df: pd.DataFrame):
                self._draws_df = draws_df

            def draws_pd(self) -> pd.DataFrame:
                return self._draws_df

        gq_draws = pd.DataFrame(
            {
                "lp__": [0.0, 1.0],
                "pKnow[1,1]": [0.1, 0.3],
                "pKnow[1,2]": [0.2, 0.4],
                "pKnow[1,3]": [0.3, 0.5],
                "pKnow[2,1]": [0.4, 0.6],
                "pKnow[2,2]": [0.5, 0.7],
                "pKnow[2,3]": [0.6, 0.8],
            }
        )

        sparse_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "problem_id": ["p10", "p30", "p20"],
                "correct": [1, 0, 1],
            }
        )
        kc_data_by_kc = format_data(sparse_df)

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (
                {"default_kc": _FakeGQ(gq_draws)},
                kc_data_by_kc,
            ),
        )

        out = model.predict_posterior(data=sparse_df, output="summary")
        true_correctness = out[
            out["parameter"] == "true_correctness"
        ].reset_index(drop=True)

        assert set(true_correctness["kc_id"]) == {"default_kc"}
        assert true_correctness["student_id"].tolist() == [
            "s1",
            "s1",
            "s1",
            "s2",
            "s2",
            "s2",
        ]
        assert true_correctness["problem_id"].tolist() == [
            "p10",
            "p30",
            "-1",
            "p20",
            "-1",
            "-1",
        ]
        assert true_correctness["mean"].tolist() == [1.0, 0.0, -1.0, 1.0, -1.0, -1.0]


class TestPredictSmoothedPosteriorDataPath:
    def test_predict_smoothed_uses_process_helper_for_default_output(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._smoothed_hidden_states_model = object()

        fake_stan = {"kc_1": MagicMock()}
        expected = {
            "kc_1": pd.DataFrame(
                {
                    "parameter": ["pred"],
                    "student_id": [1],
                    "problem_id": [1],
                    "value": [0.4],
                }
            )
        }

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (fake_stan, {}),
        )
        monkeypatch.setattr(StandardBKT, "_process_predict_gq", lambda _: expected)
        monkeypatch.setattr(
            StandardBKT,
            "_apply_kc_data_to_posterior_draws",
            lambda posterior_draws, kc_data_by_kc, drop_index_cols=True: posterior_draws,
        )

        out = model.predict_smoothed_states_posterior(
            data=_minimal_df(), output="default"
        )
        assert out == expected

    def test_predict_smoothed_uses_summary_helper(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._smoothed_hidden_states_model = object()

        fake_stan = {"kc_1": MagicMock()}
        processed = {
            "kc_1": pd.DataFrame(
                {
                    "parameter": ["pred"],
                    "student_id": [1],
                    "problem_id": [1],
                    "value": [0.4],
                }
            )
        }
        expected_summary = pd.DataFrame({"kc_id": ["kc_1"], "mean": [0.4]})

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (fake_stan, {}),
        )
        monkeypatch.setattr(StandardBKT, "_process_predict_gq", lambda _: processed)
        monkeypatch.setattr(
            StandardBKT,
            "_apply_kc_data_to_posterior_draws",
            lambda posterior_draws, kc_data_by_kc, drop_index_cols=True: posterior_draws,
        )

        def _fake_summary(draws, quantiles):
            assert draws == processed
            assert quantiles == [0.2, 0.8]
            return expected_summary

        monkeypatch.setattr(StandardBKT, "_summarize_state_predictions", _fake_summary)

        out = model.predict_smoothed_states_posterior(
            data=_minimal_df(),
            output="summary",
            summary_quantiles=[0.2, 0.8],
        )
        pd.testing.assert_frame_equal(out, expected_summary)

    def test_predict_smoothed_summary_includes_true_correctness(self, monkeypatch):
        model = StandardBKT()
        monkeypatch.setattr(model, "_fit_check", lambda: None)
        model._smoothed_hidden_states_model = object()

        class _FakeGQ:
            def __init__(self, draws_df: pd.DataFrame):
                self._draws_df = draws_df

            def draws_pd(self) -> pd.DataFrame:
                return self._draws_df

        gq_draws = pd.DataFrame(
            {
                "lp__": [0.0, 1.0],
                "pKnow[1,1]": [0.1, 0.3],
                "pKnow[1,2]": [0.2, 0.4],
                "pKnow[1,3]": [0.3, 0.5],
                "pKnow[2,1]": [0.4, 0.6],
                "pKnow[2,2]": [0.5, 0.7],
                "pKnow[2,3]": [0.6, 0.8],
            }
        )

        sparse_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "problem_id": ["p10", "p30", "p20"],
                "correct": [1, 0, 1],
            }
        )
        kc_data_by_kc = format_data(sparse_df)

        monkeypatch.setattr(
            model,
            "_predict_generated_quantities",
            lambda data, gq_model, column_mapping: (
                {"default_kc": _FakeGQ(gq_draws)},
                kc_data_by_kc,
            ),
        )

        out = model.predict_smoothed_states_posterior(
            data=sparse_df,
            output="summary",
        )
        true_correctness = out[
            out["parameter"] == "true_correctness"
        ].reset_index(drop=True)

        assert set(true_correctness["kc_id"]) == {"default_kc"}
        assert true_correctness["student_id"].tolist() == [
            "s1",
            "s1",
            "s1",
            "s2",
            "s2",
            "s2",
        ]
        assert true_correctness["problem_id"].tolist() == [
            "p10",
            "p30",
            "-1",
            "p20",
            "-1",
            "-1",
        ]
        assert true_correctness["mean"].tolist() == [1.0, 0.0, -1.0, 1.0, -1.0, -1.0]


class TestSummarizeStatePredictionsHelper:
    def test_summarize_state_predictions_includes_all_kcs(self):
        posterior_draws = {
            "kc_1": pd.DataFrame(
                {
                    "parameter": ["pred", "pred"],
                    "student_id": [1, 1],
                    "problem_id": [1, 1],
                    "value": [0.2, 0.4],
                }
            ),
            "kc_2": pd.DataFrame(
                {
                    "parameter": ["pred", "pred"],
                    "student_id": [1, 1],
                    "problem_id": [1, 1],
                    "value": [0.6, 0.8],
                }
            ),
        }

        out = StandardBKT._summarize_state_predictions(
            posterior_draws,
            quantiles=[0.025, 0.975],
        )

        assert set(out["kc_id"]) == {"kc_1", "kc_2"}
        assert {"2.50%", "97.50%"}.issubset(set(out.columns))


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
