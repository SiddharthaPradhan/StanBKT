"""Filesystem persistence helpers for fit artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import warnings
from dataclasses import replace
import pandas as pd
from cmdstanpy import from_csv as cmdstan_from_csv

from stanbkt.fits.fit_types import (
    CmdStanFit,
    FitMetadata,
    FitMethod,
    FitSaveFolder,
    FitSaves,
)
from stanbkt.fits.persistence.metadata import (
    fit_metadata_from_json,
    fit_metadata_to_json,
)

"""Base folder for per-KC fit CSV output."""
FIT_SAVE_FOLDER = "model_fits"

"""Subfolder under :data:`FIT_SAVE_FOLDER` for summary cache CSV files."""
CACHE_SAVE_FOLDER = "cache"

"""JSON filename for serialized :class:`~stanbkt.fits.fit_types.FitMetadata`."""
METADATA_SAVE_FILE = "fit_metadata.json"


def sanitize_kc_name(kc: str) -> str:
    """Convert a KC identifier into a filesystem-safe token.

    Parameters
    ----------
    kc : str
        Raw knowledge component identifier.

    Returns
    -------
    str
        Sanitized token safe for folder names.
    """
    normalized = re.sub(r"\s+", "_", kc.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", normalized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or "kc"


def add_hash_suffix(unsanitized_kc: str, sanitized_kc: str) -> str:
    """Append a deterministic hash suffix to avoid folder collisions.

    Parameters
    ----------
    unsanitized_kc : str
        Original unsanitized KC identifier.
    sanitized_kc : str
        Sanitized KC token.

    Returns
    -------
    str
        Folder key with hash suffix.
    """
    hash_suffix = hashlib.sha256(unsanitized_kc.encode()).hexdigest()[:8]
    return f"{sanitized_kc}_{hash_suffix}"


def get_fit_save_folder(kc: str) -> str:
    """Build the deterministic fit save folder name for a KC.

    Parameters
    ----------
    kc : str
        Knowledge component identifier.

    Returns
    -------
    str
        Folder name under :data:`FIT_SAVE_FOLDER`.
    """
    return add_hash_suffix(kc, sanitize_kc_name(kc))


def get_summary_cache_file(kc: str) -> str:
    """Build the deterministic summary cache CSV filename for a KC.

    Parameters
    ----------
    kc : str
        Knowledge component identifier.

    Returns
    -------
    str
        Cache CSV filename.
    """
    return f"{get_fit_save_folder(kc)}_summary.csv"


def _copy_fit_csvfiles_if_available(fit: CmdStanFit, target_dir: str) -> bool:
    """Copy CmdStan CSV files into ``target_dir`` without mutating the fit object."""
    runset = getattr(fit, "runset", None)
    csv_files = None
    if runset is not None:
        csv_files = getattr(runset, "csv_files", None)

    if not isinstance(csv_files, (list, tuple)):
        return False

    normalized_csv_files = [os.fspath(csv_file) for csv_file in csv_files]
    if not normalized_csv_files or not all(
        os.path.exists(csv_file) for csv_file in normalized_csv_files
    ):
        return False

    os.makedirs(target_dir, exist_ok=True)
    for csv_file in normalized_csv_files:
        shutil.copy2(csv_file, os.path.join(target_dir, os.path.basename(csv_file)))
    return True


def load_fit_artifacts(
    base_save_location: str,
    expected_fit_method: FitMethod,
) -> tuple[FitMetadata, dict[str, CmdStanFit], dict[str, pd.DataFrame]]:
    """Load fit, metadata, and cache artifacts from disk.

    Parameters
    ----------
    base_save_location : str
        Root path containing fit artifacts.
    expected_fit_method : FitMethod
        Fit method expected by caller.

    Returns
    -------
    tuple[FitMetadata, dict[str, BaseCmdStanFit], dict[str, pandas.DataFrame]]
        Parsed metadata, loaded fits by KC, and loaded cache by KC.

    Raises
    ------
    FileNotFoundError
        If root folder or metadata file is missing.
    ValueError
        If metadata fit method mismatches ``expected_fit_method``.
    """
    if not os.path.exists(base_save_location):
        raise FileNotFoundError(
            f"Fit save location '{base_save_location}' does not exist."
        )

    fit_metadata_path = os.path.join(base_save_location, METADATA_SAVE_FILE)
    if not os.path.exists(fit_metadata_path):
        raise FileNotFoundError(
            f"Fit metadata file '{fit_metadata_path}' does not exist. Saved fits may be corrupted or failed to save."
            " Please refit the model."
        )

    with open(fit_metadata_path, "r", encoding="utf-8") as metadata_file:
        loaded_fit_metadata = fit_metadata_from_json(metadata_file.read())

    if loaded_fit_metadata.fit_method != expected_fit_method:
        raise ValueError(
            f"Cannot load fit metadata with method '{loaded_fit_metadata.fit_method.value}' "
            f"into loader expecting method '{expected_fit_method.value}'."
        )

    fits: dict[str, CmdStanFit] = {}
    summary_cache: dict[str, pd.DataFrame] = {}
    error_kcs: set[str] = set()
    error_cache: set[FitSaveFolder] = set()

    fits_base_location = os.path.join(base_save_location, FIT_SAVE_FOLDER)
    cache_base_location = os.path.join(fits_base_location, CACHE_SAVE_FOLDER)
    for fit_save in loaded_fit_metadata.fit_saves.values():
        kc_fit_save_folder = os.path.join(fits_base_location, str(fit_save.save_folder))

        if not os.path.exists(kc_fit_save_folder):
            warnings.warn(
                (
                    f"Missing saved fit folder for KC '{fit_save.kc}' at "
                    f"'{kc_fit_save_folder}'. Skipping this KC during load."
                ),
                stacklevel=2,
            )
            error_kcs.add(fit_save.kc)
            continue

        try:
            loaded_fit = cmdstan_from_csv(kc_fit_save_folder)
        except Exception as exc:
            warnings.warn(
                (
                    f"Failed to load saved fit for KC '{fit_save.kc}' from "
                    f"'{kc_fit_save_folder}': {exc}. Skipping this KC during load."
                ),
                stacklevel=2,
            )
            error_kcs.add(fit_save.kc)
            continue

        if not isinstance(loaded_fit, CmdStanFit):
            warnings.warn(
                (
                    f"Encountered unsupported Fit type '{type(loaded_fit).__name__}' "
                    f"for KC '{fit_save.kc}'. Skipping this KC during load."
                ),
                stacklevel=2,
            )
            error_kcs.add(fit_save.kc)
            continue

        fits[fit_save.kc] = loaded_fit

        if fit_save.summary_cache_available:
            cache_file_path = os.path.join(
                cache_base_location,
                get_summary_cache_file(fit_save.kc),
            )
            if not os.path.exists(cache_file_path):
                warnings.warn(
                    (
                        f"Summary cache for KC '{fit_save.kc}' was marked available in metadata, "
                        f"but file was missing at '{cache_file_path}'."
                    ),
                    stacklevel=2,
                )
                error_cache.add(fit_save)
                continue

            try:
                summary_cache[fit_save.kc] = pd.read_csv(cache_file_path)
            except Exception as exc:
                warnings.warn(
                    (
                        f"Failed to load summary cache for KC '{fit_save.kc}' from "
                        f"'{cache_file_path}': {exc}."
                    ),
                    stacklevel=2,
                )
                error_cache.add(fit_save)
                continue

    if error_kcs:
        for error_kc in error_kcs:
            summary_cache.pop(error_kc, None)
            fits.pop(error_kc, None)
            loaded_fit_metadata.fit_saves.pop(error_kc, None)
    if error_cache:
        for error_fit_save in error_cache:
            # set summary_cache_available to False, using replace as FitSaveFolder is frozen
            fixed = replace(error_fit_save, summary_cache_available=False)
            loaded_fit_metadata.fit_saves[error_fit_save.kc] = fixed

    return loaded_fit_metadata, fits, summary_cache


def save_fit_artifacts(
    base_save_location: str,
    fits: dict[str, CmdStanFit],
    fit_metadata: FitMetadata,
    summary_cache: dict[str, pd.DataFrame],
) -> FitMetadata:
    """Persist fit CSVs, cache CSVs, and metadata to disk.

    Parameters
    ----------
    base_save_location : str
        Root path where artifacts are persisted.
    fits : dict[str, BaseCmdStanFit]
        Fitted CmdStan objects keyed by KC.
    fit_metadata : FitMetadata
        Metadata containing save-folder mapping.
    summary_cache : dict[str, pandas.DataFrame]
        Summary cache DataFrames keyed by KC.

    Returns
    -------
    FitMetadata
        Updated metadata with ``summary_cache_available`` flags.
    """
    fits_base_location = os.path.join(base_save_location, FIT_SAVE_FOLDER)
    cache_base_location = os.path.join(fits_base_location, CACHE_SAVE_FOLDER)
    os.makedirs(fits_base_location, exist_ok=True)
    os.makedirs(cache_base_location, exist_ok=True)

    updated_fit_saves: FitSaves = {}
    for fit_save in fit_metadata.fit_saves.values():
        kc_name = fit_save.kc
        if kc_name not in fits:
            continue

        fit = fits[kc_name]
        kc_fit_save_folder = os.path.join(fits_base_location, str(fit_save.save_folder))
        if not _copy_fit_csvfiles_if_available(fit, kc_fit_save_folder):
            fit.save_csvfiles(kc_fit_save_folder)

        cache_file_name = get_summary_cache_file(kc_name)
        cache_file_path = os.path.join(cache_base_location, cache_file_name)
        if kc_name in summary_cache:
            summary_cache[kc_name].to_csv(cache_file_path, index=False)
            cache_available = True
        else:
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)
            cache_available = False

        updated_fit_saves[kc_name] = FitSaveFolder(
            kc=kc_name,
            save_folder=fit_save.save_folder,
            summary_cache_available=cache_available,
        )

    fit_metadata.fit_saves = updated_fit_saves

    fit_metadata_path = os.path.join(base_save_location, METADATA_SAVE_FILE)
    with open(fit_metadata_path, "w", encoding="utf-8") as metadata_file:
        metadata_file.write(fit_metadata_to_json(fit_metadata))

    return fit_metadata
