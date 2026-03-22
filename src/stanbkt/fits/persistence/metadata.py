"""Metadata serialization helpers for fit persistence."""

from __future__ import annotations
import natsort
import json
from stanbkt.fits.fit_types import FitMetadata, FitMethod, FitSaveFolder, FitSaves


def fit_metadata_to_json(fit_metadata: FitMetadata, *, indent: int = 2) -> str:
    """Serialize fit metadata to JSON.

    Parameters
    ----------
    fit_metadata : FitMetadata
        Metadata object to serialize.
    indent : int, default=2
        JSON indentation level.

    Returns
    -------
    str
        Serialized metadata JSON string.

    Raises
    ------
    ValueError
        If metadata or fit method is empty.
    """
    if fit_metadata.fit_saves is None or fit_metadata.fit_method is None:
        raise ValueError("Fit metadata is empty. Has the model been fitted yet?")

    payload = {
        "fit_method": fit_metadata.fit_method.value,
        "fit_saves": [
            {
                "kc": fit_save.kc,
                "save_folder": str(fit_save.save_folder),
                "summary_cache_available": fit_save.summary_cache_available,
            }
            for fit_save in natsort.natsorted(
                fit_metadata.fit_saves,
                key=lambda f: (f.kc, str(f.save_folder)),
            )
        ],
    }
    return json.dumps(payload, indent=indent, sort_keys=True)


def fit_metadata_from_json(raw_text: str) -> FitMetadata:
    """Deserialize fit metadata from JSON.

    Parameters
    ----------
    raw_text : str
        Metadata JSON string.

    Returns
    -------
    FitMetadata
        Parsed metadata object.

    Raises
    ------
    ValueError
        If required schema fields are missing or invalid.
    """
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError(
            "Error parsing fit metadata: top-level JSON must be an object."
        )

    fit_method_raw = data.get("fit_method")
    fit_saves_data = data.get("fit_saves")

    try:
        fit_method = FitMethod(fit_method_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Error parsing fit metadata: 'fit_method' must be one of "
            "'mcmc', 'mle', 'vb', or 'pathfinder'."
        ) from exc

    if not isinstance(fit_saves_data, list):
        raise ValueError(
            "Error parsing fit metadata: top-level key 'fit_saves' must be an array."
        )

    parsed_fit_saves: FitSaves = set()
    for entry in fit_saves_data:
        if not isinstance(entry, dict):
            raise ValueError(
                "Error parsing fit metadata: each 'fit_saves' entry must be an object."
            )

        kc = entry.get("kc")
        if not isinstance(kc, str):
            raise ValueError(
                "Error parsing fit metadata: each 'fit_saves' entry must include string field 'kc'."
            )

        save_folder = entry.get("save_folder")
        if not isinstance(save_folder, str):
            raise ValueError(
                f"Error parsing fit metadata: metadata for KC '{kc}' must include string field 'save_folder'."
            )

        summary_cache_available = entry.get("summary_cache_available", False)
        if not isinstance(summary_cache_available, bool):
            raise ValueError(
                f"Error parsing fit metadata: metadata for KC '{kc}' must include boolean field 'summary_cache_available' when provided."
            )

        parsed_fit_saves.add(
            FitSaveFolder(
                kc=kc,
                save_folder=save_folder,
                summary_cache_available=summary_cache_available,
            )
        )

    return FitMetadata(fit_method=fit_method, fit_saves=parsed_fit_saves)
