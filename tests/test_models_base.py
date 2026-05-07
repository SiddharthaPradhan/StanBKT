from cmdstanpy import CmdStanMCMC
from stanbkt.fits.core.base import FitBase
import json
import pytest
import numpy as np
import pandas as pd
import os
import zipfile
from unittest.mock import MagicMock

from stanbkt.models.core.base import BKTModelBase
from stanbkt.models.priors import StandardPriors
from stanbkt.models.model_types import ModelType, InitKnowledgeStrategy
from stanbkt.fits.fit_types import FitMethod
from stanbkt.utils.model_archive import MODEL_ARCHIVE_SUFFIX
from stanbkt.utils.verbose import VerbosityLevel
from stanbkt.utils.model_io import MODEL_METADATA_SAVE_FILE

# ---------------------------------------------------------------------------
# Minimal concrete subclass — only satisfies the abstract interface
# ---------------------------------------------------------------------------


class _ConcreteModel(BKTModelBase):
    def fit(
        self,
        data,
        priors=None,
        column_mapping=None,
        stan_fit_options=None,
        overwrite_kcs=False,
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

    def _build_stan_data_dict(self, kc_data, priors=None):
        return {"correctness": kc_data.correctness}

    def _default_priors(self):
        return StandardPriors()

    def _default_priors_class(self):
        return StandardPriors

    def _extract_bkt_params_from_fit(self, fit, n_students, point_estimate="mean"):
        return (
            np.full(n_students, 0.2),
            np.full(n_students, 0.3),
            np.full(n_students, 0.1),
            np.full(n_students, 0.2),
            np.full(n_students, 0.1),
        )


class DummyFit(FitBase):

    @property
    def _fit_method(self):
        return FitMethod.MCMC

    def _create_inits(self, *args, **kwargs):
        return None

    def _summary(self, kcs=None, kc_col_name="kc_id", percentiles=(2.5, 97.5)):
        return pd.DataFrame()


def _make_fitted_model(kcs=("kc_a", "kc_b")):
    """Return a model whose _is_fitted and fits reflect a completed fit."""
    model = _ConcreteModel()
    model._is_fitted = True
    model.fits: FitBase = DummyFit()
    # bypass add_fit since we don't have actual fit objects to add, but want the fitted KC keys to be present in the model's fit state
    model.fits.stan_fits = {kc: object() for kc in kcs}  # ty:ignore[invalid-assignment]
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

    def test_individual_initial_knowledge_default_false(self):
        m = _ConcreteModel()
        assert m.individual_initial_knowledge is False

    def test_individual_initial_knowledge_can_be_set_true(self):
        m = _ConcreteModel(individual_initial_knowledge=True)
        assert m.individual_initial_knowledge is True

    def test_init_knowledge_strategy_default_correctness_only(self):
        m = _ConcreteModel()
        assert m.init_knowledge_strategy == InitKnowledgeStrategy.CORRECTNESS_ONLY

    def test_init_knowledge_strategy_joint_allowed_with_individual_knowledge(self):
        m = _ConcreteModel(
            individual_initial_knowledge=True,
            init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
        )
        assert m.init_knowledge_strategy == InitKnowledgeStrategy.JOINT

    def test_init_knowledge_strategy_joint_without_individual_raises(self):
        with pytest.raises(ValueError, match="individual_initial_knowledge"):
            _ConcreteModel(
                individual_initial_knowledge=False,
                init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
            )

    def test_invalid_init_strategy_error_message_includes_values(self):
        with pytest.raises(ValueError, match="CORRECTNESS_ONLY"):
            _ConcreteModel(
                individual_initial_knowledge=False,
                init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
            )


# ---------------------------------------------------------------------------
# fit_check
# ---------------------------------------------------------------------------


class TestFitCheck:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m._fit_check()

    def test_raises_when_empty_fits_even_if_is_fitted_true(self):
        m = _ConcreteModel()
        m._is_fitted = True
        m.fits = DummyFit()
        m.fits.num_fitted_kcs = 0
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
        m.fits.stan_fits = {"kc_a": object()}  # ty:ignore[invalid-assignment]
        m.fits.num_fitted_kcs = 1
        m.fits._save = MagicMock()  # ty:ignore[invalid-assignment]
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
