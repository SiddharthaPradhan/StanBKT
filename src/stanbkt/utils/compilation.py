from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import cmdstanpy as csp
from platformdirs import PlatformDirs

_INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"(?P<path>[^"]+)"', re.MULTILINE)
_CACHE_NAMESPACE = "compiled_stan"


def _as_path(stan_file: str | os.PathLike[str]) -> Path:
    """Validate and resolve a Stan source path.

    Parameters
    ----------
    stan_file : str or os.PathLike[str]
        Path to a Stan source file.

    Returns
    -------
    Path
        Absolute resolved path to the Stan source file.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the provided path does not point to a ``.stan`` file.
    """
    path = Path(stan_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Stan file does not exist: {path}")
    if path.suffix != ".stan":
        raise ValueError(f"Expected a .stan file, got: {path}")
    return path


def _normalize_for_hash(value: Any) -> Any:
    """Normalize a value into a deterministic JSON-serializable form.

    Parameters
    ----------
    value : Any
        Value to normalize before hashing.

    Returns
    -------
    Any
        A recursively normalized value with deterministic ordering for
        dictionaries and sets.
    """
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, set):
        normalized_items = [_normalize_for_hash(item) for item in value]
        return sorted(
            normalized_items, key=lambda item: json.dumps(item, sort_keys=True)
        )
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _read_text(path: Path) -> str:
    """Read a UTF-8 text file.

    Parameters
    ----------
    path : Path
        File path to read.

    Returns
    -------
    str
        File contents as text.
    """
    return path.read_text(encoding="utf-8")


def _resolve_include_path(include_path: str, parent_file: Path) -> Path:
    """Resolve an included Stan file relative to its parent file.

    Parameters
    ----------
    include_path : str
        Relative path extracted from a Stan ``#include`` directive.
    parent_file : Path
        Stan file containing the include directive.

    Returns
    -------
    Path
        Absolute resolved path to the included Stan file.

    Raises
    ------
    FileNotFoundError
        If the included file cannot be found.
    """
    resolved = (parent_file.parent / include_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Included Stan file '{include_path}' referenced by '{parent_file}' does not exist."
        )
    return resolved


def _collect_stan_sources(
    stan_file: Path,
    seen: set[Path] | None = None,
) -> list[Path]:
    """Collect a Stan file and all recursively included Stan sources.

    Parameters
    ----------
    stan_file : Path
        Entry-point Stan file.
    seen : set[Path] or None, default=None
        Visited-file set used to avoid processing the same file multiple times
        during recursive include traversal.

    Returns
    -------
    list[Path]
        List of absolute source paths consisting of the entry file and all
        transitively included local Stan files.
    """
    if seen is None:
        seen = set()

    normalized_file = stan_file.resolve()
    if normalized_file in seen:
        return []

    seen.add(normalized_file)
    source_paths = [normalized_file]
    source_text = _read_text(normalized_file)
    for match in _INCLUDE_PATTERN.finditer(source_text):
        include_file = _resolve_include_path(match.group("path"), normalized_file)
        source_paths.extend(_collect_stan_sources(include_file, seen=seen))
    return source_paths


def _get_source_root(source_paths: list[Path]) -> Path:
    """Return the common root directory for a collection of Stan sources.

    Parameters
    ----------
    source_paths : list[Path]
        Collection of Stan source paths.

    Returns
    -------
    Path
        Common parent directory for the provided source files.
    """
    common_root = os.path.commonpath([str(path.parent) for path in source_paths])
    return Path(common_root)


def _build_dict_for_hash(
    stan_file: Path,
    cpp_options: dict[str, Any] | None,
    stanc_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the normalized payload used to compute the cache key.

    Parameters
    ----------
    stan_file : Path
        Entry-point Stan file.
    cpp_options : dict[str, Any] or None
        C++ compiler options forwarded to CmdStanPy.
    stanc_options : dict[str, Any] or None
        Stan compiler options forwarded to CmdStanPy.

    Returns
    -------
    dict[str, Any]
        Normalized payload containing all relevant source file contents and normalized compile options.
    """
    # get all source files and contents, including original stan file
    source_paths = sorted(
        _collect_stan_sources(stan_file), key=lambda path: path.as_posix()
    )
    source_root = _get_source_root(source_paths)
    source_payload = []
    for source_path in source_paths:
        source_payload.append(
            {
                "path": source_path.relative_to(source_root).as_posix(),
                "contents": _read_text(source_path),
            }
        )

    return {
        "sources": source_payload,
        "cpp_options": _normalize_for_hash(cpp_options or {}),
        "stanc_options": _normalize_for_hash(stanc_options or {}),
    }


def stan_model_cache_key(
    stan_file: str | os.PathLike[str],
    cpp_options: dict[str, Any] | None = None,
    stanc_options: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic cache key for a compiled Stan model.

    Parameters
    ----------
    stan_file : str or os.PathLike[str]
        Path to the entry-point Stan source file.
    cpp_options : dict[str, Any] or None, default=None
        C++ compiler options forwarded to CmdStanPy.
    stanc_options : dict[str, Any] or None, default=None
        Stan compiler options forwarded to CmdStanPy.

    Returns
    -------
    str
        SHA-256 hex digest derived from the Stan source contents, transitive
        local includes, and normalized compile options.
    """
    resolved_stan_file = _as_path(stan_file)
    payload = _build_dict_for_hash(
        resolved_stan_file,
        cpp_options=cpp_options,
        stanc_options=stanc_options,
    )
    serialized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def get_stan_model_cache_dir(
    stan_file: str | os.PathLike[str],
    cpp_options: dict[str, Any] | None = None,
    stanc_options: dict[str, Any] | None = None,
) -> Path:
    """Return the cache directory for a compiled Stan executable.

    Parameters
    ----------
    stan_file : str or os.PathLike[str]
        Path to the entry-point Stan source file.
    cpp_options : dict[str, Any] or None, default=None
        C++ compiler options forwarded to CmdStanPy.
    stanc_options : dict[str, Any] or None, default=None
        Stan compiler options forwarded to CmdStanPy.

    Returns
    -------
    Path
        Platform-specific cache directory for the executable associated with
        the given source and compile configuration.
    """
    resolved_stan_file = _as_path(stan_file)
    cache_key = stan_model_cache_key(
        resolved_stan_file,
        cpp_options=cpp_options,
        stanc_options=stanc_options,
    )
    cache_root = PlatformDirs(appname="stanbkt", appauthor=False).user_cache_path
    return cache_root / _CACHE_NAMESPACE / f"{resolved_stan_file.stem}-{cache_key}"


def _cached_executable_path(stan_file: Path, cache_dir: Path) -> Path:
    """Return the cached executable path for a Stan file within a cache dir.

    Parameters
    ----------
    stan_file : Path
        Entry-point Stan file.
    cache_dir : Path
        Cache directory for the compiled artifact.

    Returns
    -------
    Path
        Executable path under ``cache_dir`` using the Stan model stem and the
        platform-appropriate executable suffix.
    """
    suffix = ".exe" if os.name == "nt" else ""
    return cache_dir / f"{stan_file.stem}{suffix}"


def compile_stan_model(
    stan_file: str | os.PathLike[str],
    cpp_options: dict[str, Any] | None = None,
    stanc_options: dict[str, Any] | None = None,
) -> csp.CmdStanModel:
    """Compile or reuse a cached Stan executable and return a CmdStan model.

    Parameters
    ----------
    stan_file : str or os.PathLike[str]
        Path to the entry-point Stan source file.
    cpp_options : dict[str, Any] or None, default=None
        C++ compiler options forwarded to CmdStanPy.
    stanc_options : dict[str, Any] or None, default=None
        Stan compiler options forwarded to CmdStanPy.

    Returns
    -------
    cmdstanpy.CmdStanModel
        Model instance backed by a cached compiled executable.

    Raises
    ------
    RuntimeError
        If CmdStanPy finishes compilation without reporting an executable path.

    Notes
    -----
    The package Stan source files remain in place and are not copied into the
    cache. Only the compiled executable is stored under the platform cache
    directory.
    """
    # stan path resolution
    resolved_stan_file = _as_path(stan_file)
    cache_dir = get_stan_model_cache_dir(
        resolved_stan_file,
        cpp_options=cpp_options,
        stanc_options=stanc_options,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    # check for cached executable
    cached_exe_file = _cached_executable_path(resolved_stan_file, cache_dir)

    if cached_exe_file.exists():
        return csp.CmdStanModel(
            stan_file=str(resolved_stan_file),
            exe_file=str(cached_exe_file),
            stanc_options=stanc_options,
            cpp_options=cpp_options,
        )

    compiled_model = csp.CmdStanModel(
        stan_file=str(resolved_stan_file),
        stanc_options=stanc_options,
        cpp_options=cpp_options,
        force_compile=True,
    )

    if compiled_model.exe_file is None:
        raise RuntimeError(
            "CmdStanPy failed to produce a compiled executable. Perhaps reconfigure and/or reinstall CmdStanPy?"
        )

    compiled_exe_file = Path(compiled_model.exe_file)
    shutil.copy2(compiled_exe_file, cached_exe_file)
    if compiled_exe_file != cached_exe_file and compiled_exe_file.exists():
        compiled_exe_file.unlink()

    return csp.CmdStanModel(
        stan_file=str(resolved_stan_file),
        exe_file=str(cached_exe_file),
        stanc_options=stanc_options,
        cpp_options=cpp_options,
    )
