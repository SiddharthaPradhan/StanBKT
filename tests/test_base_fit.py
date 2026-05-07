import hashlib
import json
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

import stanbkt.fits.persistence.fit_io as persistence_io
from stanbkt.fits.persistence.fit_io import (
    FitMetadata,
    FitSaveEntry,
    FitMethod,
    CACHE_SAVE_FOLDER,
    FIT_SAVE_FOLDER,
    METADATA_SAVE_FILE,
    sanitize_kc_name,
    add_hash_suffix,
    get_fit_save_folder,
    get_summary_cache_file,
)
from stanbkt.fits.persistence.metadata import (
    fit_metadata_to_json,
    fit_metadata_from_json,
)
from stanbkt.fits.core.base import FitBase
from stanbkt.utils.verbose import VerbosityLevel

# ---------------------------------------------------------------------------
# Minimal concrete subclass — only satisfies abstract interface
# ---------------------------------------------------------------------------


class _ConcreteFit(FitBase):
    @property
    def _fit_method(self) -> FitMethod:
        return FitMethod.MCMC

    def _create_inits(self, kc=None):
        return {}

    def _summary(self, kcs=None, kc_col_name="kc_id", percentiles=(2.5, 97.5)):
        return pd.DataFrame()


class _VBConcreteFit(FitBase):
    @property
    def _fit_method(self) -> FitMethod:
        return FitMethod.VB

    def _create_inits(self, kc=None):
        return {}

    def _summary(self, kcs=None, kc_col_name="kc_id", percentiles=(2.5, 97.5)):
        return pd.DataFrame()


@pytest.fixture(autouse=True)
def _mock_fit_method_detection_for_test_doubles(monkeypatch):
    # Most tests in this module use lightweight doubles instead of real CmdStan fit types.
    monkeypatch.setattr(
        FitMethod,
        "infer_fit_method_from_stan_fit",
        staticmethod(lambda _: FitMethod.MCMC),
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestBaseFitInit:

    def test_kc_fits_starts_empty(self):
        fit = _ConcreteFit()
        assert fit.stan_fits == {}

    def test_fit_metadata_starts_empty(self):
        fit = _ConcreteFit()
        assert fit._fit_metadata.fit_method == FitMethod.MCMC
        assert fit._fit_metadata.fit_saves == {}

    def test_summary_cache_starts_empty(self):
        fit = _ConcreteFit()
        assert fit._summary_cache == {}

    def test_default_verbose_is_info(self):
        fit = _ConcreteFit()
        assert fit.verbose == VerbosityLevel.INFO

    def test_explicit_verbose(self):
        fit = _ConcreteFit(verbose=VerbosityLevel.WARN)
        assert fit.verbose == VerbosityLevel.WARN


# ---------------------------------------------------------------------------
# add_fit
# ---------------------------------------------------------------------------


class TestAddFit:
    def test_adds_new_kc_to_fits(self):
        fit = _ConcreteFit()
        mock_stan_fit = MagicMock()
        fit.add_fit("kc_a", mock_stan_fit)
        assert "kc_a" in fit.stan_fits
        assert fit.stan_fits["kc_a"] is mock_stan_fit

    def test_adds_metadata_entry_for_new_kc(self):
        fit = _ConcreteFit()
        fit.add_fit("kc_a", MagicMock())
        assert "kc_a" in fit._fit_metadata.fit_saves

    def test_metadata_entry_contains_save_folder(self):
        fit = _ConcreteFit()
        fit.add_fit("kc_a", MagicMock())
        matching_entries = [fit._fit_metadata.fit_saves["kc_a"]]
        assert len(matching_entries) == 1
        assert isinstance(matching_entries[0].save_folder, str)
        assert len(matching_entries[0].save_folder) > 0

    def test_adding_multiple_kcs(self):
        fit = _ConcreteFit()
        fit.add_fit("kc_a", MagicMock())
        fit.add_fit("kc_b", MagicMock())
        assert set(fit.stan_fits.keys()) == {"kc_a", "kc_b"}
        assert set(fit._fit_metadata.fit_saves.keys()) == {"kc_a", "kc_b"}

    def test_overwrite_replaces_fit(self):
        fit = _ConcreteFit()
        first = MagicMock()
        second = MagicMock()
        fit.add_fit("kc_a", first)
        fit.add_fit("kc_a", second, overwrite_kcs=True)
        assert fit.stan_fits["kc_a"] is second

    def test_overwrite_clears_summary_cache(self):
        fit = _ConcreteFit()
        fit.add_fit("kc_a", MagicMock())
        fit._summary_cache["kc_a"] = pd.DataFrame({"x": [1]})
        fit.add_fit("kc_a", MagicMock(), overwrite_kcs=True)
        assert "kc_a" not in fit._summary_cache

    def test_overwrite_does_not_clear_other_kc_cache(self):
        fit = _ConcreteFit()
        fit.add_fit("kc_a", MagicMock())
        fit.add_fit("kc_b", MagicMock())
        fit._summary_cache["kc_b"] = pd.DataFrame({"x": [1]})
        fit.add_fit("kc_a", MagicMock(), overwrite_kcs=True)
        assert "kc_b" in fit._summary_cache

    def test_overwrite_emits_warning(self, capsys):
        # Must use WARN verbose level so the warning is not suppressed
        fit = _ConcreteFit(verbose=VerbosityLevel.WARN)
        fit.add_fit("kc_a", MagicMock())
        fit.add_fit("kc_a", MagicMock(), overwrite_kcs=True)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "kc_a" in out


# ---------------------------------------------------------------------------
# _update_summary_cache
# ---------------------------------------------------------------------------


class TestUpdateSummaryCache:
    def test_stores_dataframe(self):
        fit = _ConcreteFit()
        df = pd.DataFrame({"mean": [0.5], "std": [0.1]})
        fit._update_summary_cache("kc_a", df)
        pd.testing.assert_frame_equal(fit._summary_cache["kc_a"], df)

    def test_overwrites_existing_entry(self):
        fit = _ConcreteFit()
        fit._update_summary_cache("kc_a", pd.DataFrame({"mean": [0.1]}))
        new_df = pd.DataFrame({"mean": [0.9]})
        fit._update_summary_cache("kc_a", new_df)
        pd.testing.assert_frame_equal(fit._summary_cache["kc_a"], new_df)


# ---------------------------------------------------------------------------
# _save / _load
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_save_with_empty_fits_is_noop_and_writes_no_files(self, tmp_path, capsys):
        save_dir = tmp_path / "fit_saves"
        fit = _ConcreteFit(verbose=VerbosityLevel.WARN)

        fit._save(save_dir)

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "not been fitted" in out
        assert not save_dir.exists()

    def test_save_then_load_round_trip_with_tmp_path(self, tmp_path, monkeypatch):
        save_dir = tmp_path / "fit_saves"
        fit = _ConcreteFit()

        class _DummySavedFit:
            def save_csvfiles(self, folder: str) -> None:
                os.makedirs(folder, exist_ok=True)
                with open(
                    os.path.join(folder, "mock_chain.csv"), "w", encoding="utf-8"
                ) as f:
                    f.write("lp__\n0\n")

        class _DummyLoadedFit:
            pass

        loaded_fit_obj = _DummyLoadedFit()
        fit.add_fit("kc_a", _DummySavedFit())  # ty:ignore[invalid-argument-type]
        original_summary_df = pd.DataFrame({"mean": [0.5], "std": [0.1]})
        fit._update_summary_cache("kc_a", original_summary_df)

        monkeypatch.setattr(persistence_io, "CmdStanFit", (_DummyLoadedFit,))
        monkeypatch.setattr(
            persistence_io, "cmdstan_from_csv", lambda _: loaded_fit_obj
        )

        fit._save(str(save_dir))

        loaded = _ConcreteFit._load(str(save_dir))

        assert loaded._fit_metadata == fit._fit_metadata
        assert set(loaded.stan_fits.keys()) == {"kc_a"}
        assert loaded.stan_fits["kc_a"] is loaded_fit_obj
        pd.testing.assert_frame_equal(
            loaded._summary_cache["kc_a"], original_summary_df
        )

        assert (save_dir / "fit_metadata.json").exists()
        saved_metadata = next(iter(fit._fit_metadata.fit_saves.values()))
        saved_folder = saved_metadata.save_folder
        cache_file = get_summary_cache_file("kc_a")

        assert (save_dir / FIT_SAVE_FOLDER / str(saved_folder)).exists()
        assert (save_dir / FIT_SAVE_FOLDER / CACHE_SAVE_FOLDER / cache_file).exists()
        assert saved_metadata.summary_cache_available is True

    def test_summary_percentiles_persisted_through_save_load(
        self, tmp_path, monkeypatch
    ):
        save_dir = tmp_path / "fit_saves"
        fit = _ConcreteFit(summary_percentiles=(5.0, 95.0))

        class _DummySavedFit:
            def save_csvfiles(self, folder: str) -> None:
                os.makedirs(folder, exist_ok=True)
                with open(
                    os.path.join(folder, "mock_chain.csv"), "w", encoding="utf-8"
                ) as f:
                    f.write("lp__\n0\n")

        class _DummyLoadedFit:
            pass

        fit.add_fit("kc_a", _DummySavedFit())  # ty:ignore[invalid-argument-type]
        monkeypatch.setattr(persistence_io, "CmdStanFit", (_DummyLoadedFit,))
        monkeypatch.setattr(
            persistence_io, "cmdstan_from_csv", lambda _: _DummyLoadedFit()
        )
        fit._save(str(save_dir))

        loaded = _ConcreteFit._load(str(save_dir))

        assert loaded._summary_percentiles == (5.0, 95.0)
        assert loaded._fit_metadata.summary_percentiles == (5.0, 95.0)

    def test_load_warns_and_skips_kc_when_loaded_fit_type_is_unsupported(
        self, tmp_path, monkeypatch
    ):
        save_dir = tmp_path / "fit_saves"
        fit = _ConcreteFit()

        class _DummySavedFit:
            def save_csvfiles(self, folder: str) -> None:
                os.makedirs(folder, exist_ok=True)
                with open(
                    os.path.join(folder, "mock_chain.csv"), "w", encoding="utf-8"
                ) as f:
                    f.write("lp__\n0\n")

        fit.add_fit("kc_a", _DummySavedFit())  # ty:ignore[invalid-argument-type]
        fit._update_summary_cache("kc_a", pd.DataFrame({"mean": [0.5]}))
        fit._save(str(save_dir))

        monkeypatch.setattr(persistence_io, "cmdstan_from_csv", lambda _: object())

        with pytest.warns(UserWarning, match="unsupported Fit type"):
            loaded = _ConcreteFit._load(str(save_dir))

        assert loaded.stan_fits == {}
        assert "kc_a" not in loaded._summary_cache
        assert loaded._fit_metadata.fit_saves == {}

    def test_load_raises_on_fit_method_mismatch(self, tmp_path):
        save_dir = tmp_path / "fit_saves"
        os.makedirs(save_dir, exist_ok=True)
        fit_metadata_path = save_dir / METADATA_SAVE_FILE
        fit_metadata_path.write_text(
            fit_metadata_to_json(FitMetadata(fit_method=FitMethod.MCMC, fit_saves={})),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="expecting method 'vb'"):
            _VBConcreteFit._load(str(save_dir))


# ---------------------------------------------------------------------------
# _sanitize_kc_name
# ---------------------------------------------------------------------------


class TestSanitizeKCName:
    @pytest.mark.parametrize(
        "kc, expected",
        [
            ("algebra", "algebra"),
            ("  spaces  ", "spaces"),
            ("kc name", "kc_name"),
            ("knowledge_component_1", "knowledge_component_1"),
            ("a.b.c", "a.b.c"),
            (
                "kc!@#",
                "kc",
            ),  # special chars stripped to underscores, trailing _ stripped
            ("", "kc"),  # empty → default
            ("...", "kc"),  # dots allowed but stripped at edges → empty → default
            ("_leading", "leading"),  # leading underscore stripped
            ("trailing_", "trailing"),  # trailing underscore stripped
        ],
    )
    def test_sanitize(self, kc, expected):
        assert sanitize_kc_name(kc) == expected

    def test_collapses_multiple_underscores(self):
        result = sanitize_kc_name("kc  name")  # 2 spaces → 2 underscores → 1
        assert "__" not in result
        assert result == "kc_name"

    def test_returns_string(self):
        assert isinstance(sanitize_kc_name("test"), str)


# ---------------------------------------------------------------------------
# _add_hash_suffix
# ---------------------------------------------------------------------------


class TestAddHashSuffix:
    def test_result_format(self):
        result = add_hash_suffix("kc_a", "kc_a")
        assert result == f"kc_a_{hashlib.sha256('kc_a'.encode()).hexdigest()[:8]}"

    def test_suffix_is_8_hex_chars(self):
        result = add_hash_suffix("some kc", "some_kc")
        suffix = result.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_deterministic(self):
        r1 = add_hash_suffix("kc_a", "kc_a")
        r2 = add_hash_suffix("kc_a", "kc_a")
        assert r1 == r2

    def test_different_originals_produce_different_suffixes(self):
        r1 = add_hash_suffix("kc_a", "kc")
        r2 = add_hash_suffix("kc_b", "kc")
        assert r1 != r2

    def test_sanitized_prefix_used(self):
        result = add_hash_suffix("kc_a", "my_sanitized_name")
        assert result.startswith("my_sanitized_name_")


# ---------------------------------------------------------------------------
# _get_fit_save_folder
# ---------------------------------------------------------------------------


class TestGetFitSaveFolder:
    def test_result_is_string(self):
        assert isinstance(get_fit_save_folder("kc_a"), str)

    def test_folder_is_non_empty(self):
        assert get_fit_save_folder("kc_a") != ""

    def test_deterministic(self):
        r1 = get_fit_save_folder("kc_a")
        r2 = get_fit_save_folder("kc_a")
        assert r1 == r2

    def test_different_kcs_produce_different_folders(self):
        r1 = get_fit_save_folder("kc_a")
        r2 = get_fit_save_folder("kc_b")
        assert r1 != r2

    def test_combines_sanitized_name_and_hash(self):
        result = get_fit_save_folder("algebra")
        sanitized = sanitize_kc_name("algebra")
        expected = add_hash_suffix("algebra", sanitized)
        assert result == expected


# ---------------------------------------------------------------------------
# fit_metadata_to_json / fit_metadata_from_json
# ---------------------------------------------------------------------------


class TestFitMetadataToJson:
    def test_produces_valid_json(self):
        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                "kc_a": FitSaveEntry(
                    kc="kc_a",
                    save_folder="kc_a_abc12345",
                    summary_cache_available=True,
                )
            },
        )
        raw = fit_metadata_to_json(metadata)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_contains_fit_method(self):
        raw = fit_metadata_to_json(FitMetadata(fit_method=FitMethod.MCMC))
        parsed = json.loads(raw)
        assert parsed["fit_method"] == "mcmc"

    def test_contains_fits_key(self):
        metadata = FitMetadata(
            fit_method=FitMethod.MLE,
            fit_saves={
                "kc_a": FitSaveEntry(
                    kc="kc_a",
                    save_folder="folder_abc",
                    summary_cache_available=False,
                )
            },
        )
        raw = fit_metadata_to_json(metadata)
        parsed = json.loads(raw)
        assert "fit_saves" in parsed
        assert any(
            entry["kc"] == "kc_a"
            and entry["save_folder"] == "folder_abc"
            and entry["summary_cache_available"] is False
            for entry in parsed["fit_saves"]
        )

    def test_returns_string(self):
        raw = fit_metadata_to_json(FitMetadata(fit_method=FitMethod.VB))
        assert isinstance(raw, str)


class TestFitMetadataFromJson:
    def _make_json(self, fit_method="mcmc", fit_saves=None) -> str:
        if fit_saves is None:
            fit_saves = [
                {
                    "kc": "kc_a",
                    "save_folder": "kc_a_abc12345",
                    "summary_cache_available": False,
                }
            ]
        return json.dumps({"fit_method": fit_method, "fit_saves": fit_saves})

    def test_round_trip(self):
        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                "kc_a": FitSaveEntry(kc="kc_a", save_folder="kc_a_abc12345"),
                "kc_b": FitSaveEntry(kc="kc_b", save_folder="kc_b_def67890"),
            },
        )
        raw = fit_metadata_to_json(metadata)
        loaded_metadata = fit_metadata_from_json(raw)
        assert loaded_metadata.fit_method == FitMethod.MCMC
        assert loaded_metadata.fit_saves == metadata.fit_saves

    @pytest.mark.parametrize(
        "method", [FitMethod.MCMC, FitMethod.MLE, FitMethod.VB, FitMethod.PATHFINDER]
    )
    def test_accepts_all_valid_fit_methods(self, method):
        raw = self._make_json(fit_method=method)
        loaded_metadata = fit_metadata_from_json(raw)
        assert loaded_metadata.fit_method == method

    def test_raises_for_invalid_fit_method(self):
        raw = self._make_json(fit_method="sampling")
        with pytest.raises(ValueError, match="fit_method"):
            fit_metadata_from_json(raw)

    def test_raises_for_missing_fit_method(self):
        raw = json.dumps({"fit_saves": []})
        with pytest.raises(ValueError, match="fit_method"):
            fit_metadata_from_json(raw)

    def test_raises_for_non_dict_top_level(self):
        with pytest.raises(ValueError, match="top-level JSON must be an object"):
            fit_metadata_from_json(json.dumps([1, 2, 3]))

    def test_raises_when_fit_saves_is_not_list(self):
        raw = json.dumps({"fit_method": "mcmc", "fit_saves": "not_a_list"})
        with pytest.raises(ValueError, match="'fit_saves' must be an array"):
            fit_metadata_from_json(raw)

    def test_raises_when_fit_saves_is_missing(self):
        raw = json.dumps({"fit_method": "mcmc"})
        with pytest.raises(ValueError, match="'fit_saves' must be an array"):
            fit_metadata_from_json(raw)

    def test_raises_when_entry_is_not_dict(self):
        raw = json.dumps({"fit_method": "mcmc", "fit_saves": ["bad_entry"]})
        with pytest.raises(ValueError, match="must be an object"):
            fit_metadata_from_json(raw)

    def test_raises_when_save_folder_missing(self):
        raw = json.dumps({"fit_method": "mcmc", "fit_saves": [{"kc": "kc_a"}]})
        with pytest.raises(ValueError, match="save_folder"):
            fit_metadata_from_json(raw)

    def test_raises_when_save_folder_is_not_string(self):
        raw = json.dumps(
            {
                "fit_method": "mcmc",
                "fit_saves": [
                    {
                        "kc": "kc_a",
                        "save_folder": 123,
                        "summary_cache_available": False,
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="save_folder"):
            fit_metadata_from_json(raw)

    def test_raises_when_kc_is_missing(self):
        raw = json.dumps(
            {
                "fit_method": "mcmc",
                "fit_saves": [{"save_folder": "x", "summary_cache_available": False}],
            }
        )
        with pytest.raises(ValueError, match="field 'kc'"):
            fit_metadata_from_json(raw)

    def test_raises_when_summary_cache_available_is_not_bool(self):
        raw = json.dumps(
            {
                "fit_method": "mcmc",
                "fit_saves": [
                    {
                        "kc": "kc_a",
                        "save_folder": "folder",
                        "summary_cache_available": "yes",
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="summary_cache_available"):
            fit_metadata_from_json(raw)

    def test_empty_fit_saves(self):
        raw = self._make_json(fit_saves=[])
        loaded_metadata = fit_metadata_from_json(raw)
        assert loaded_metadata.fit_method == FitMethod.MCMC
        assert loaded_metadata.fit_saves == {}

    def test_summary_percentiles_round_trips(self):
        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            summary_percentiles=(5.0, 95.0),
        )
        raw = fit_metadata_to_json(metadata)
        loaded = fit_metadata_from_json(raw)
        assert loaded.summary_percentiles == (5.0, 95.0)

    def test_summary_percentiles_defaults_when_missing_from_json(self):
        raw = json.dumps({"fit_method": "mcmc", "fit_saves": []})
        loaded = fit_metadata_from_json(raw)
        assert loaded.summary_percentiles == (2.5, 97.5)

    def test_group_mapping_round_trips(self):
        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                "kc_a": FitSaveEntry(
                    kc="kc_a",
                    save_folder="kc_a_abc12345",
                    group2index={"g1": 1, "g2": 2},
                    groups={"g1", "g2"},
                )
            },
        )
        raw = fit_metadata_to_json(metadata)
        loaded = fit_metadata_from_json(raw)
        assert loaded.fit_saves["kc_a"].group2index == {"g1": 1, "g2": 2}
        assert loaded.fit_saves["kc_a"].groups == {"g1", "g2"}

    def test_raises_when_summary_percentiles_is_wrong_length(self):
        raw = json.dumps(
            {"fit_method": "mcmc", "fit_saves": [], "summary_percentiles": [5]}
        )
        with pytest.raises(ValueError, match="summary_percentiles"):
            fit_metadata_from_json(raw)

    def test_raises_when_summary_percentiles_is_not_a_list(self):
        raw = json.dumps(
            {"fit_method": "mcmc", "fit_saves": [], "summary_percentiles": "5,95"}
        )
        with pytest.raises(ValueError, match="summary_percentiles"):
            fit_metadata_from_json(raw)
