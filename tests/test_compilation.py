from __future__ import annotations

import os
from pathlib import Path

import cmdstanpy
import pytest

from stanbkt.utils import compilation


class _FakePlatformDirs:
    def __init__(self, cache_root: Path):
        self.user_cache_path = cache_root


def test_stan_model_cache_key_is_stable_across_option_order(tmp_path: Path) -> None:
    stan_file = tmp_path / "model.stan"
    stan_file.write_text(
        "data { int<lower=0> n; } parameters { real y; } model { y ~ normal(0, 1); }",
        encoding="utf-8",
    )

    cpp_options_a = {"STAN_THREADS": True, "O1": True}
    cpp_options_b = {"O1": True, "STAN_THREADS": True}
    stanc_options_a = {"include-paths": ["a", "b"], "warn-pedantic": True}
    stanc_options_b = {"warn-pedantic": True, "include-paths": ["a", "b"]}

    cache_key_a = compilation.stan_model_cache_key(
        stan_file,
        cpp_options=cpp_options_a,
        stanc_options=stanc_options_a,
    )
    cache_key_b = compilation.stan_model_cache_key(
        stan_file,
        cpp_options=cpp_options_b,
        stanc_options=stanc_options_b,
    )

    assert cache_key_a == cache_key_b


def test_stan_model_cache_key_changes_when_included_file_changes(
    tmp_path: Path,
) -> None:
    includes_dir = tmp_path / "includes"
    includes_dir.mkdir()
    included = includes_dir / "shared.stan"
    included.write_text("functions { real foo() { return 1; } }", encoding="utf-8")

    stan_file = tmp_path / "model.stan"
    stan_file.write_text(
        '#include "includes/shared.stan"\nmodel { foo() ~ normal(0, 1); }',
        encoding="utf-8",
    )

    first_key = compilation.stan_model_cache_key(stan_file)
    included.write_text("functions { real foo() { return 2; } }", encoding="utf-8")
    second_key = compilation.stan_model_cache_key(stan_file)

    assert first_key != second_key


def test_compile_stan_model_caches_only_executable_and_reuses_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    include_dir = tmp_path / "stan"
    include_dir.mkdir()
    included = include_dir / "shared.stan"
    included.write_text("functions { real foo() { return 1; } }", encoding="utf-8")

    stan_file = include_dir / "model.stan"
    stan_file.write_text(
        '#include "shared.stan"\nmodel { target += foo(); }',
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    class FakeCmdStanModel:
        def __init__(
            self,
            stan_file: str | None = None,
            exe_file: str | None = None,
            stanc_options: dict | None = None,
            cpp_options: dict | None = None,
            force_compile: bool = False,
            compile: bool | None = None,
        ) -> None:
            calls.append(
                {
                    "stan_file": stan_file,
                    "exe_file": exe_file,
                    "stanc_options": stanc_options,
                    "cpp_options": cpp_options,
                    "force_compile": force_compile,
                    "compile": compile,
                }
            )
            self.exe_file = exe_file
            if stan_file is not None and exe_file is None:
                source_exe = Path(stan_file).with_suffix(
                    ".exe" if os.name == "nt" else ""
                )
                source_exe.write_text("compiled", encoding="utf-8")
                self.exe_file = str(source_exe)

    monkeypatch.setattr(compilation.csp, "CmdStanModel", FakeCmdStanModel)

    cpp_options = {"STAN_THREADS": True}
    stanc_options = {"test": True}
    cache_dir = compilation.get_stan_model_cache_dir(
        stan_file,
        cpp_options=cpp_options,
        stanc_options=stanc_options,
    )

    compilation.compile_stan_model(
        stan_file,
        cpp_options=cpp_options,
        stanc_options=stanc_options,
    )
    compilation.compile_stan_model(
        stan_file,
        cpp_options=cpp_options,
        stanc_options=stanc_options,
    )

    cached_exe = compilation._cached_executable_path(stan_file, cache_dir)

    assert len(calls) == 3
    assert calls[0]["exe_file"] is None
    assert calls[0]["force_compile"] is True
    assert calls[1]["exe_file"] == str(cached_exe)
    assert calls[2]["exe_file"] == str(cached_exe)
    assert cached_exe.exists()
    assert not (cache_dir / "model.stan").exists()
    assert not (cache_dir / "shared.stan").exists()
    assert not Path(stan_file).with_suffix(".exe" if os.name == "nt" else "").exists()


@pytest.mark.slow
def test_compile_stan_model_end_to_end_recompiles_on_source_change(
    tmp_path: Path,
) -> None:
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan installation is not configured.")

    stan_dir = tmp_path / "stan"
    stan_dir.mkdir()
    run_token = tmp_path.name

    include_file = stan_dir / "shared.stan"
    include_file.write_text(
        f"functions {{ real foo() {{ return 1; }} }} // {run_token}",
        encoding="utf-8",
    )

    stan_file = stan_dir / "model.stan"
    stan_file.write_text(
        '#include "shared.stan"\nmodel { target += foo(); }\n',
        encoding="utf-8",
    )

    first_cache_dir = compilation.get_stan_model_cache_dir(stan_file)
    first_cached_exe = compilation._cached_executable_path(stan_file, first_cache_dir)
    if first_cached_exe.exists():
        first_cached_exe.unlink()

    first_model = compilation.compile_stan_model(stan_file)
    assert first_model.exe_file == str(first_cached_exe)
    assert first_cached_exe.exists()

    first_cached_mtime = first_cached_exe.stat().st_mtime_ns
    second_model = compilation.compile_stan_model(stan_file)
    assert second_model.exe_file == str(first_cached_exe)
    assert first_cached_exe.stat().st_mtime_ns == first_cached_mtime

    include_file.write_text(
        f"functions {{ real foo() {{ return 2; }} }} // {run_token}",
        encoding="utf-8",
    )

    second_cache_dir = compilation.get_stan_model_cache_dir(stan_file)
    second_cached_exe = compilation._cached_executable_path(stan_file, second_cache_dir)
    assert second_cache_dir != first_cache_dir
    if second_cached_exe.exists():
        second_cached_exe.unlink()

    third_model = compilation.compile_stan_model(stan_file)
    assert third_model.exe_file == str(second_cached_exe)
    assert second_cached_exe.exists()
    assert not Path(stan_file).with_suffix(".exe" if os.name == "nt" else "").exists()


def test_get_cache_root(tmp_path: Path, monkeypatch) -> None:
    """Test that get_cache_root returns the correct directory."""
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    result = compilation.get_cache_root()
    assert result == cache_root / "compiled_stan"


def test_list_cached_models_empty(tmp_path: Path, monkeypatch) -> None:
    """Test listing cached models when cache is empty."""
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    cached_models = compilation.list_cached_models()
    assert cached_models == []


def test_list_cached_models_with_models(tmp_path: Path, monkeypatch) -> None:
    """Test listing cached models when cache has models."""
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    # Create fake cache directories
    cache_dir = compilation.get_cache_root()
    cache_dir.mkdir(parents=True)
    (cache_dir / "model1-abc123").mkdir()
    (cache_dir / "model2-def456").mkdir()

    cached_models = compilation.list_cached_models()
    assert len(cached_models) == 2
    assert any("model1-abc123" in str(m) for m in cached_models)
    assert any("model2-def456" in str(m) for m in cached_models)


def test_clear_stan_cache_entire_cache(tmp_path: Path, monkeypatch) -> None:
    """Test clearing the entire Stan cache."""
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    # Create fake cache directories
    cache_dir = compilation.get_cache_root()
    cache_dir.mkdir(parents=True)
    model1_dir = cache_dir / "model1-abc123"
    model2_dir = cache_dir / "model2-def456"
    model1_dir.mkdir()
    model2_dir.mkdir()
    (model1_dir / "executable").write_text("fake exe", encoding="utf-8")
    (model2_dir / "executable").write_text("fake exe", encoding="utf-8")

    # Clear entire cache
    count = compilation.clear_stan_cache()
    assert count == 2
    assert not cache_dir.exists()


def test_clear_stan_cache_specific_model(tmp_path: Path, monkeypatch) -> None:
    """Test clearing cache for a specific model."""
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    # Create a simple Stan file
    stan_file = tmp_path / "model.stan"
    stan_file.write_text(
        "parameters { real y; } model { y ~ normal(0, 1); }",
        encoding="utf-8",
    )

    # Create fake cache for this model
    cache_dir = compilation.get_stan_model_cache_dir(stan_file)
    cache_dir.mkdir(parents=True)
    (cache_dir / "executable").write_text("fake exe", encoding="utf-8")

    # Clear specific model cache
    count = compilation.clear_stan_cache(stan_file)
    assert count == 1
    assert not cache_dir.exists()


def test_clear_stan_cache_nonexistent(tmp_path: Path, monkeypatch) -> None:
    """Test clearing cache when it doesn't exist."""
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        compilation,
        "PlatformDirs",
        lambda appname, appauthor=False: _FakePlatformDirs(cache_root),
    )

    # Clear cache that doesn't exist
    count = compilation.clear_stan_cache()
    assert count == 0
