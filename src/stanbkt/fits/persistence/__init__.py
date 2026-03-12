"""Persistence primitives for fit artifacts.

This subpackage contains serialization and filesystem I/O for fit metadata,
fit CSV artifacts, and summary cache files.
"""

from stanbkt.fits.persistence.fit_io import (
    CACHE_SAVE_FOLDER,
    FIT_SAVE_FOLDER,
    METADATA_SAVE_FILE,
    add_hash_suffix,
    get_fit_save_folder,
    get_summary_cache_file,
    load_fit_artifacts,
    sanitize_kc_name,
    save_fit_artifacts,
)
from stanbkt.fits.persistence.metadata import (
    fit_metadata_from_json,
    fit_metadata_to_json,
)

__all__ = [
    "CACHE_SAVE_FOLDER",
    "FIT_SAVE_FOLDER",
    "METADATA_SAVE_FILE",
    "add_hash_suffix",
    "fit_metadata_from_json",
    "fit_metadata_to_json",
    "get_fit_save_folder",
    "get_summary_cache_file",
    "load_fit_artifacts",
    "sanitize_kc_name",
    "save_fit_artifacts",
]
