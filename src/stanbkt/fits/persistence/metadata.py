"""Metadata serialization helpers for fit persistence."""

from __future__ import annotations
import natsort
import json
from stanbkt.fits.fit_types import FitMetadata, FitMethod, FitSaveEntry, FitSaves


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
                **(
                    {"group2index": fit_save.group2index}
                    if fit_save.group2index not in (None, {})
                    else {}
                ),
                **(
                    {"groups": natsort.natsorted(list(fit_save.groups))}
                    if fit_save.groups not in (None, set())
                    else {}
                ),
            }
            for fit_save in natsort.natsorted(
                fit_metadata.fit_saves.values(),
                key=lambda f: (f.kc, str(f.save_folder)),
            )
        ],
        "summary_percentiles": list(fit_metadata.summary_percentiles),
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
    summary_percentiles_raw = data.get("summary_percentiles", [2.5, 97.5])

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

    if (
        not isinstance(summary_percentiles_raw, list)
        or len(summary_percentiles_raw) != 2
        or not all(isinstance(v, (int, float)) for v in summary_percentiles_raw)
    ):
        raise ValueError(
            "Error parsing fit metadata: 'summary_percentiles' must be a two-element array of numbers."
        )
    summary_percentiles = (
        float(summary_percentiles_raw[0]),
        float(summary_percentiles_raw[1]),
    )

    parsed_fit_saves: FitSaves = {}
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

        group2index_raw = entry.get("group2index", None)
        if group2index_raw is None:
            parsed_group2index: dict[str, int] | None = None
        else:
            if not isinstance(group2index_raw, dict):
                raise ValueError(
                    f"Error parsing fit metadata: metadata for KC '{kc}' field 'group2index' must be an object when provided."
                )
            parsed_group2index = {}
            for group_name, index in group2index_raw.items():
                if not isinstance(group_name, str) or not isinstance(index, int):
                    raise ValueError(
                        f"Error parsing fit metadata: metadata for KC '{kc}' field 'group2index' must map string group IDs to integer indices."
                    )
                parsed_group2index[group_name] = index

        groups_raw = entry.get("groups", None)
        if groups_raw is None:
            parsed_groups: set[str] | None = None
        else:
            if not isinstance(groups_raw, list) or not all(
                isinstance(group_name, str) for group_name in groups_raw
            ):
                raise ValueError(
                    f"Error parsing fit metadata: metadata for KC '{kc}' field 'groups' must be an array of strings when provided."
                )
            parsed_groups = {str(group_name) for group_name in groups_raw}

        parsed_fit_saves[kc] = FitSaveEntry(
            kc=kc,
            save_folder=save_folder,
            summary_cache_available=summary_cache_available,
            group2index=parsed_group2index,
            groups=parsed_groups,
        )

    return FitMetadata(
        fit_method=fit_method,
        fit_saves=parsed_fit_saves,
        summary_percentiles=summary_percentiles,
    )
