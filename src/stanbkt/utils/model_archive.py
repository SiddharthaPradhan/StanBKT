"""Helpers for packaging StanBKT model artifacts into a single archive."""

from __future__ import annotations

import os
from pathlib import Path
import zipfile


MODEL_ARCHIVE_SUFFIX = ".stanbktmod"


def normalize_model_archive_path(path: str | os.PathLike[str]) -> str:
    """Return a normalized archive path with the StanBKT model suffix."""
    archive_path = os.fspath(path)
    if archive_path.endswith(MODEL_ARCHIVE_SUFFIX):
        return archive_path
    return f"{archive_path}{MODEL_ARCHIVE_SUFFIX}"


def pack_model_directory(source_dir: str, archive_path: str | os.PathLike[str]) -> str:
    """Package a directory of model artifacts into a compressed archive."""
    normalized_archive_path = normalize_model_archive_path(archive_path)
    os.makedirs(os.path.dirname(normalized_archive_path) or ".", exist_ok=True)

    source_root = Path(source_dir)
    with zipfile.ZipFile(
        normalized_archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive_file:
        for file_path in sorted(source_root.rglob("*")):
            if file_path.is_file():
                archive_file.write(file_path, file_path.relative_to(source_root))

    return normalized_archive_path


def unpack_model_archive(archive_path: str | os.PathLike[str], target_dir: str) -> str:
    """Extract a StanBKT model archive into a target directory."""
    normalized_archive_path = normalize_model_archive_path(archive_path)
    if not os.path.exists(normalized_archive_path):
        raise FileNotFoundError(
            f"Model archive '{normalized_archive_path}' does not exist."
        )
    if not zipfile.is_zipfile(normalized_archive_path):
        raise ValueError(
            f"Model archive '{normalized_archive_path}' is not a valid zip archive."
        )

    with zipfile.ZipFile(normalized_archive_path, mode="r") as archive_file:
        archive_file.extractall(target_dir)

    return normalized_archive_path
