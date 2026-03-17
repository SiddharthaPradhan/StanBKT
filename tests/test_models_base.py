from cmdstanpy import CmdStanMCMC
from stanbkt.fits.core.base import BaseFit
import json
import pytest
import numpy as np
import os
import zipfile
from unittest.mock import MagicMock

from stanbkt.models.core.base import BKTModelBase
from stanbkt.models.priors import BayesianPriors
from stanbkt.models.model_types import ModelType, PriorEstimationType
from stanbkt.fits.fit_types import FitMethod
from stanbkt.utils.model_archive import MODEL_ARCHIVE_SUFFIX
from stanbkt.utils.verbose import VerbosityLevel
from stanbkt.utils.model_io import MODEL_METADATA_SAVE_FILE


# ---------------------------------------------------------------------------
# Minimal concrete subclass — only satisfies the abstract interface
# ---------------------------------------------------------------------------


class _ConcreteModel(BKTModelBase):
    def fit(
        self, data, column_mapping=None, method=FitMethod.MCMC, stan_fit_kwargs=None
    ):
        return self

    def evaluate(self, **kwargs):
        return {}

    @property
    def _stan_model_filename(self):
        return "/nonexistent.stan"

    @property
    def _stan_hidden_filename(self):
        return "/nonexistent_hidden.stan"

    @property
    def _stan_smoothed_hidden_filename(self):
        return "/nonexistent_smoothed.stan"

    def _build_stan_data_dict(self, correctness):
        return {"correctness": correctness}

    def _extract_bkt_params_from_fit(self, fit, n_students, point_estimate="mean"):
        return (
            np.full(n_students, 0.2),
            np.full(n_students, 0.3),
            np.full(n_students, 0.1),
            np.full(n_students, 0.2),
            np.full(n_students, 0.1),
        )


class DummyFit(BaseFit):

    @property
    def _fit_method(self):
        return FitMethod.MCMC

    def _create_inits(self, *args, **kwargs):
        return None

    def summary(self, *args, **kwargs):
        return "summary"


def _make_fitted_model(kcs=("kc_a", "kc_b")):
    """Return a model whose _is_fitted and fits reflect a completed fit."""
    model = _ConcreteModel()
    model._is_fitted = True
    model.fits: BaseFit = DummyFit()
    # bypass add_fit since we don't have actual fit objects to add, but want the fitted KC keys to be present in the model's fit state
    model.fits.kc_fits = {kc: object() for kc in kcs}  # ty:ignore[invalid-assignment]
    model.fits.num_fitted_kcs = len(kcs)
    return model


# ---------------------------------------------------------------------------
# BKTModelBase.__init__
# ---------------------------------------------------------------------------


class TestBKTModelBaseInit:
    def test_default_verbose(self):
        m = _ConcreteModel()
        assert m.verbose == VerbosityLevel.INFO

    def test_stan_compile_kwargs_default_empty(self):
        m = _ConcreteModel()
        assert m.stan_compile_kwargs == {}

    def test_cpp_compile_kwargs_default_empty(self):
        m = _ConcreteModel()
        assert m.cpp_compile_kwargs == {}

    def test_stan_model_is_none(self):
        m = _ConcreteModel()
        assert m._stan_model is None

    def test_fitsis_none(self):
        m = _ConcreteModel()
        assert m.fits is None

    def test_is_fitted_is_false(self):
        m = _ConcreteModel()
        assert m._is_fitted is False

    def test_default_fit_method_is_mcmc(self):
        m = _ConcreteModel()
        assert m._fit_method == FitMethod.MCMC

    def test_custom_compile_kwargs_stored(self):
        m = _ConcreteModel(
            stan_compile_kwargs={"foo": "bar"}, cpp_compile_kwargs={"baz": 1}
        )
        assert m.stan_compile_kwargs == {"foo": "bar"}
        assert m.cpp_compile_kwargs == {"baz": 1}

    def test_string_fit_method_is_normalized(self):
        m = _ConcreteModel(fit_method="mcmc")
        assert m._fit_method == FitMethod.MCMC


# ---------------------------------------------------------------------------
# fit_check
# ---------------------------------------------------------------------------


class TestFitCheck:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m._fit_check()

    def test_raises_when_fitsis_none_even_if_is_fitted_true(self):
        m = _ConcreteModel()
        m._is_fitted = True
        m.fits = None
        with pytest.raises(RuntimeError, match="must be fitted"):
            m._fit_check()

    def test_passes_when_fitted(self):
        m = _make_fitted_model()
        m._fit_check()  # should not raise


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.summary()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict()


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_raises_when_not_fitted(self, tmp_path):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.save(tmp_path / "model")

    def test_save_delegates_to_fits_save(self, tmp_path):
        m = _make_fitted_model()
        m.fits._save = MagicMock()  # ty: ignore[method-assign]
        artifact_path = tmp_path / "model"

        m.save(artifact_path)

        m.fits._save.assert_called_once()
        save_call_path = m.fits._save.call_args.args[0]
        assert isinstance(save_call_path, str)
        assert str(artifact_path) not in save_call_path
        assert (tmp_path / f"model{MODEL_ARCHIVE_SUFFIX}").exists()

    def test_save_writes_model_metadata_for_reconstruction(self, tmp_path):
        m = _ConcreteModel(
            fit_method=FitMethod.VB,
            verbose=VerbosityLevel.DEBUG,
            stan_compile_kwargs={"stanc": True},
            cpp_compile_kwargs={"threads": 4},
        )
        m._is_fitted = True
        m.fits = DummyFit()
        m.fits.kc_fits = {"kc_a": object()}  # ty:ignore[invalid-assignment]
        m.fits.num_fitted_kcs = 1
        m.fits._save = MagicMock()  # ty: ignore[method-assign]
        artifact_path = tmp_path / "saved_model"

        m.save(artifact_path)

        archive_path = tmp_path / f"saved_model{MODEL_ARCHIVE_SUFFIX}"
        assert archive_path.exists()

        with zipfile.ZipFile(archive_path, "r") as archive_file:
            with archive_file.open(MODEL_METADATA_SAVE_FILE, "r") as metadata_file:
                model_metadata = json.load(metadata_file)

        assert model_metadata["model_module"] == m.__class__.__module__
        assert model_metadata["model_qualname"] == m.__class__.__qualname__
        assert model_metadata["model_init_kwargs"] == {
            "fit_method": "vb",
            "verbose": int(VerbosityLevel.DEBUG),
            "stan_compile_kwargs": {"stanc": True},
            "cpp_compile_kwargs": {"threads": 4},
        }


# ---------------------------------------------------------------------------
# check_data_contains_fitted_kcs
# ---------------------------------------------------------------------------


class TestCheckDataContainsFittedKCs:
    def test_passes_when_all_kcs_are_fitted(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        m.check_data_contains_fitted_kcs({"kc_a"})  # subset → no error

    def test_passes_for_exact_fitted_set(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        m.check_data_contains_fitted_kcs({"kc_a", "kc_b"})

    def test_raises_for_unknown_kc(self):
        m = _make_fitted_model(kcs=("kc_a",))
        with pytest.raises(ValueError, match="kc_z"):
            m.check_data_contains_fitted_kcs({"kc_z"})

    def test_warns_for_partially_overlapping_set(self, capsys):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        m.check_data_contains_fitted_kcs({"kc_a", "kc_z"})

        out = capsys.readouterr().out
        assert "WARNING: Data contains 1 KCs that were not fitted." in out
        assert "kc_z" in out

    def test_raises_fit_check_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError):
            m.check_data_contains_fitted_kcs({"kc_a"})


# ---------------------------------------------------------------------------
# get_kc_in_fitted_kcs
# ---------------------------------------------------------------------------


class TestGetKCInFittedKCs:
    def test_returns_exact_intersection(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        result = m.get_kcs_in_fitted_kcs({"kc_a", "kc_c"})
        assert result == {"kc_a"}

    def test_returns_all_when_full_overlap(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        result = m.get_kcs_in_fitted_kcs({"kc_a", "kc_b"})
        assert result == {"kc_a", "kc_b"}

    def test_returns_empty_when_no_overlap(self):
        m = _make_fitted_model(kcs=("kc_a",))
        result = m.get_kcs_in_fitted_kcs({"kc_z"})
        assert result == set()

    def test_raises_fit_check_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError):
            m.get_kcs_in_fitted_kcs({"kc_a"})


# ---------------------------------------------------------------------------
# BayesianPriors.get_default_priors
# ---------------------------------------------------------------------------


class TestBayesianPriorsGetDefaultPriors:
    def test_standard_returns_scalar_priors(self):
        priors = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert isinstance(priors, dict)
        assert all(isinstance(v, float) for v in priors.values())

    def test_standard_returns_all_10_prior_keys(self):
        priors = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert set(priors.keys()) == set(BayesianPriors)

    def test_nested_returns_same_scalar_priors_as_standard(self):
        standard = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        nested = BayesianPriors.get_default_priors(
            ModelType.NESTED, PriorEstimationType.DEFAULT
        )
        assert standard == nested

    def test_grouped_returns_list_priors(self):
        priors = BayesianPriors.get_default_priors(
            ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=3
        )
        assert all(isinstance(v, list) for v in priors.values())

    def test_grouped_list_length_matches_n_groups(self):
        n = 4
        priors = BayesianPriors.get_default_priors(
            ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=n
        )
        assert all(
            len(v) == n for v in priors.values()  # ty:ignore[invalid-argument-type]
        )

    def test_grouped_values_replicated_from_scalar(self):
        scalar = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        grouped = BayesianPriors.get_default_priors(
            ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=2
        )
        for key in scalar:
            assert grouped[key] == [scalar[key], scalar[key]]

    def test_grouped_raises_for_non_integer_n_groups(self):
        with pytest.raises(ValueError, match="n_groups must be an integer"):
            BayesianPriors.get_default_priors(
                ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=None
            )

    def test_grouped_raises_for_zero_n_groups(self):
        with pytest.raises(ValueError, match="n_groups must be > 0"):
            BayesianPriors.get_default_priors(
                ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=0
            )

    def test_grouped_raises_for_negative_n_groups(self):
        with pytest.raises(ValueError, match="n_groups must be > 0"):
            BayesianPriors.get_default_priors(
                ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=-1
            )

    def test_joint_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BayesianPriors.get_default_priors(
                ModelType.STANDARD, PriorEstimationType.JOINT
            )


# ---------------------------------------------------------------------------
# BayesianPriors.add_missing_priors
# ---------------------------------------------------------------------------


class TestBayesianPriorsAddMissingPriors:
    def test_empty_input_returns_all_defaults(self):
        result = BayesianPriors.add_missing_priors(
            {}, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        defaults = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert result == defaults

    def test_partial_override_merges_correctly(self):
        override: dict[BayesianPriors, int | float | list[int | float]] = {
            BayesianPriors.LEARN_MU: 1.5
        }
        result = BayesianPriors.add_missing_priors(
            override, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert result[BayesianPriors.LEARN_MU] == 1.5
        # Other keys remain as defaults
        defaults = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        for key in defaults:
            if key != BayesianPriors.LEARN_MU:
                assert result[key] == defaults[key]

    def test_string_key_accepted(self):
        result = BayesianPriors.add_missing_priors(
            {"learn_mu": 2.0},  # ty:ignore[invalid-argument-type]
            ModelType.STANDARD,
            PriorEstimationType.DEFAULT,
        )
        assert result[BayesianPriors.LEARN_MU] == 2.0

    def test_invalid_string_key_raises(self):
        with pytest.raises(ValueError, match="Unsupported prior key"):
            vals = {"nonexistent_param": 1.0}
            BayesianPriors.add_missing_priors(
                vals,  # ty:ignore[invalid-argument-type]
                ModelType.STANDARD,
                PriorEstimationType.DEFAULT,
            )

    def test_result_contains_all_prior_keys(self):
        result = BayesianPriors.add_missing_priors(
            {BayesianPriors.SLIP_MU: -0.5},
            ModelType.STANDARD,
            PriorEstimationType.DEFAULT,
        )
        assert set(result.keys()) == set(BayesianPriors)

    def test_grouped_partial_override(self):
        override: dict[BayesianPriors, int | float | list[float | int]] = {
            BayesianPriors.LEARN_MU: [1.5, 1.5, 1.5]
        }
        result = BayesianPriors.add_missing_priors(
            override, ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=3
        )
        assert result[BayesianPriors.LEARN_MU] == [1.5, 1.5, 1.5]

    def test_full_override_with_enum_keys(self):
        overrides: dict[BayesianPriors, int | float | list[int | float]] = {
            prior: 0.0 for prior in BayesianPriors
        }
        result = BayesianPriors.add_missing_priors(
            overrides, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert all(v == 0.0 for v in result.values())
