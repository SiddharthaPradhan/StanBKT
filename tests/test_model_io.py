from __future__ import annotations

import os
import json
import zipfile
from typing import Any

import numpy as np
import pytest

from stanbkt.fits.core.base import BaseFit
from stanbkt.fits.persistence.fit_io import METADATA_SAVE_FILE
from stanbkt.fits.persistence.metadata import fit_metadata_to_json
from stanbkt.fits.fit_types import FitMetadata, FitMethod
from stanbkt.models.core.base import BKTModelBase
from stanbkt.utils.model_archive import MODEL_ARCHIVE_SUFFIX
from stanbkt.utils.verbose import VerbosityLevel
from stanbkt.utils.model_io import MODEL_METADATA_SAVE_FILE, load_model


class _LoadedFit(BaseFit):
    @property
    def _fit_method(self):
        return FitMethod.MCMC

    def _create_inits(self, *args, **kwargs):
        return None

    def _summary(self, kcs=None, kc_col_name="kc_id", percentiles=(2.5, 97.5)):
        return None


class _FakeFitClass:
    @classmethod
    def _load(cls, base_save_location: str):
        loaded = _LoadedFit()
        loaded.stan_fits = {"kc_1": object()}  # ty:ignore[invalid-assignment]
        loaded.num_fitted_kcs = 1
        return loaded


class _FakeModel(BKTModelBase):
    def __init__(self, fit_method: FitMethod = FitMethod.MCMC, **kwargs: Any):
        super().__init__(fit_method=fit_method, **kwargs)
        self.fit_class = _FakeFitClass  # ty:ignore[invalid-assignment]

    def fit(self, data, column_mapping=None, stan_fit_options=None):
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

    def _build_stan_data_dict(self, correctness: np.ndarray):
        return {"correctness": correctness}

    def _extract_bkt_params_from_fit(self, fit, n_students, point_estimate="mean"):
        return (
            np.full(n_students, 0.2),
            np.full(n_students, 0.3),
            np.full(n_students, 0.1),
            np.full(n_students, 0.2),
            np.full(n_students, 0.1),
        )


def _write_metadata(base_path: str, fit_method: FitMethod = FitMethod.MCMC) -> None:
    metadata = FitMetadata(fit_method=fit_method, fit_saves={})
    metadata_path = os.path.join(base_path, METADATA_SAVE_FILE)
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(fit_metadata_to_json(metadata))


def _write_model_metadata(
    base_path: str,
    model_class: type[BKTModelBase],
    *,
    fit_method: FitMethod = FitMethod.MCMC,
    verbose: VerbosityLevel = VerbosityLevel.INFO,
    stan_compile_kwargs: dict[str, object] | None = None,
    cpp_compile_kwargs: dict[str, object] | None = None,
) -> None:
    model_metadata = {
        "model_module": model_class.__module__,
        "model_qualname": model_class.__qualname__,
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
        "model_init_kwargs": {
            "fit_method": fit_method.value,
            "verbose": int(verbose),
            "stan_compile_kwargs": stan_compile_kwargs or {},
            "cpp_compile_kwargs": cpp_compile_kwargs or {},
        },
    }
    model_metadata_path = os.path.join(base_path, MODEL_METADATA_SAVE_FILE)
    with open(model_metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f)


def _pack_archive(directory: str, archive_path: str) -> None:
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                archive.write(file_path, os.path.relpath(file_path, directory))


class TestLoadModel:
    def test_load_model_infers_model_class_and_restores_fitted_state(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        _write_metadata(str(artifact_dir), fit_method=FitMethod.MCMC)
        _write_model_metadata(
            str(artifact_dir),
            _FakeModel,
            fit_method=FitMethod.MCMC,
            verbose=VerbosityLevel.DEBUG,
            stan_compile_kwargs={"a": 1},
            cpp_compile_kwargs={"b": 2},
        )
        artifact_path = tmp_path / f"model{MODEL_ARCHIVE_SUFFIX}"
        _pack_archive(str(artifact_dir), str(artifact_path))

        model = load_model(load_base_location=artifact_path)

        assert isinstance(model, _FakeModel)
        assert model._fit_method == FitMethod.MCMC
        assert model.verbose == VerbosityLevel.DEBUG
        assert model.stan_compile_kwargs == {"a": 1}
        assert model.cpp_compile_kwargs == {"b": 2}
        assert model._is_fitted is True
        assert model.fits is not None
        assert model.fits.num_fitted_kcs == 1
        assert hasattr(model, "_loaded_artifact_dir")

    def test_load_model_raises_for_missing_archive(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing.stanbktmod"):
            load_model(load_base_location=tmp_path / f"missing{MODEL_ARCHIVE_SUFFIX}")

    def test_load_model_raises_for_missing_fit_metadata(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        _write_model_metadata(str(artifact_dir), _FakeModel)
        artifact_path = tmp_path / f"model{MODEL_ARCHIVE_SUFFIX}"
        _pack_archive(str(artifact_dir), str(artifact_path))

        with pytest.raises(FileNotFoundError, match="fit_metadata.json"):
            load_model(load_base_location=artifact_path)

    def test_load_model_raises_for_missing_model_metadata(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        _write_metadata(str(artifact_dir), fit_method=FitMethod.MCMC)
        artifact_path = tmp_path / f"model{MODEL_ARCHIVE_SUFFIX}"
        _pack_archive(str(artifact_dir), str(artifact_path))

        with pytest.raises(FileNotFoundError, match="model_metadata.json"):
            load_model(load_base_location=artifact_path)

    def test_load_model_raises_for_mismatched_fit_method(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        _write_metadata(str(artifact_dir), fit_method=FitMethod.VB)
        _write_model_metadata(
            str(artifact_dir),
            _FakeModel,
            fit_method=FitMethod.MCMC,
        )
        artifact_path = tmp_path / f"model{MODEL_ARCHIVE_SUFFIX}"
        _pack_archive(str(artifact_dir), str(artifact_path))

        with pytest.raises(ValueError, match="disagree on fit_method"):
            load_model(load_base_location=artifact_path)
