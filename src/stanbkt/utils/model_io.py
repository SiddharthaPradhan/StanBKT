"""Model persistence helpers."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
from typing import Any

from stanbkt.fits.persistence.fit_io import METADATA_SAVE_FILE
from stanbkt.fits.persistence.metadata import fit_metadata_from_json
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.core.base import BKTModelBase
from stanbkt.utils.model_archive import MODEL_ARCHIVE_SUFFIX, unpack_model_archive
from stanbkt.utils.verbose import VerbosityLevel


MODEL_METADATA_SAVE_FILE = "model_metadata.json"


def _resolve_model_class(base_path: str) -> tuple[type[BKTModelBase], dict[str, Any]]:
    model_metadata_path = os.path.join(base_path, MODEL_METADATA_SAVE_FILE)
    if not os.path.exists(model_metadata_path):
        raise FileNotFoundError(
            f"Model metadata file '{model_metadata_path}' does not exist."
        )

    with open(model_metadata_path, "r", encoding="utf-8") as model_metadata_file:
        model_metadata = json.load(model_metadata_file)

    module_name = model_metadata.get("model_module")
    qualname = model_metadata.get("model_qualname")
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise ValueError(
            f"Model metadata file '{model_metadata_path}' is invalid: "
            "missing string fields 'model_module' and 'model_qualname'."
        )

    init_kwargs = model_metadata.get("model_init_kwargs")
    if not isinstance(init_kwargs, dict):
        raise ValueError(
            f"Model metadata file '{model_metadata_path}' is invalid: "
            "missing object field 'model_init_kwargs'."
        )

    module = importlib.import_module(module_name)
    model_class: object = module
    for attr in qualname.split("."):
        model_class = getattr(model_class, attr)

    if not isinstance(model_class, type) or not issubclass(model_class, BKTModelBase):
        raise TypeError(
            f"Resolved model class '{module_name}.{qualname}' is not a BKTModelBase subclass."
        )

    return model_class, dict(init_kwargs)


def _parse_model_init_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    fit_method_raw = raw_kwargs.get("fit_method")
    verbose_raw = raw_kwargs.get("verbose")
    stan_compile_kwargs_raw = raw_kwargs.get("stan_compile_kwargs")
    cpp_compile_kwargs_raw = raw_kwargs.get("cpp_compile_kwargs")

    if not isinstance(fit_method_raw, str):
        raise ValueError(
            "Saved model init kwargs must include string field 'fit_method'."
        )
    if not isinstance(verbose_raw, int):
        raise ValueError(
            "Saved model init kwargs must include integer field 'verbose'."
        )
    if not isinstance(stan_compile_kwargs_raw, dict):
        raise ValueError(
            "Saved model init kwargs must include object field 'stan_compile_kwargs'."
        )
    if not isinstance(cpp_compile_kwargs_raw, dict):
        raise ValueError(
            "Saved model init kwargs must include object field 'cpp_compile_kwargs'."
        )

    parsed_kwargs: dict[str, Any] = dict(raw_kwargs)
    parsed_kwargs["fit_method"] = FitMethod(fit_method_raw)
    parsed_kwargs["verbose"] = VerbosityLevel(verbose_raw)
    parsed_kwargs["stan_compile_kwargs"] = dict(stan_compile_kwargs_raw)
    parsed_kwargs["cpp_compile_kwargs"] = dict(cpp_compile_kwargs_raw)
    return parsed_kwargs


def load_model(
    load_base_location: str | os.PathLike[str],
) -> BKTModelBase:
    """Load a previously saved StanBKT model.

    Parameters
    ----------
    load_base_location : str | os.PathLike[str]
        Path to a saved StanBKT model archive.
    Returns
    -------
    BKTModelBase
        Loaded model instance with fitted state restored.

    Raises
    ------
    FileNotFoundError
        If fit metadata or model metadata is missing.
    ValueError
        If model metadata exists but is invalid.
    TypeError
        If resolved model class is not a ``BKTModelBase`` subclass.
    """
    archive_path = os.fspath(load_base_location)

    temp_dir = tempfile.mkdtemp(prefix="stanbkt_load_")
    try:
        unpack_model_archive(archive_path, temp_dir)

        metadata_path = os.path.join(temp_dir, METADATA_SAVE_FILE)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Fit metadata file '{metadata_path}' does not exist."
            )

        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            fit_metadata = fit_metadata_from_json(metadata_file.read())

        resolved_model_class, init_kwargs = _resolve_model_class(temp_dir)
        parsed_init_kwargs = _parse_model_init_kwargs(init_kwargs)
        saved_fit_method = parsed_init_kwargs.get("fit_method")
        if saved_fit_method != fit_metadata.fit_method:
            raise ValueError(
                "Saved model metadata and fit metadata disagree on fit_method. "
                f"Model metadata: '{saved_fit_method}', fit metadata: '{fit_metadata.fit_method}'."
            )

        model = resolved_model_class(**parsed_init_kwargs)
        model.fits = model.fit_class._load(temp_dir)
        model._is_fitted = model.fits.num_fitted_kcs > 0
        model._loaded_artifact_dir = temp_dir
        return model
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
