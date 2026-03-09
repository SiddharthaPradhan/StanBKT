import hashlib
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stanbkt.fits.base import BaseFit, FIT_METHOD, FitMetadata
from stanbkt.utils.verbose import VerbosityLevel


# ---------------------------------------------------------------------------
# Minimal concrete subclass — only satisfies abstract interface
# ---------------------------------------------------------------------------

class _ConcreteFit(BaseFit):
    @property
    def _fit_method(self) -> FIT_METHOD:
        return "mcmc"

    def _create_inits(self, kc=None):
        return {}

    def summary(self, kc):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestBaseFitInit:
    def test_save_base_location_stored(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        assert fit.save_base_location == "/tmp/test"

    def test_fits_starts_empty(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        assert fit.fits == {}

    def test_fit_metadata_starts_empty(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        assert fit.fit_metadata == {}

    def test_summary_cache_starts_empty(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        assert fit.summary_cache == {}

    def test_default_verbose_is_info(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        assert fit.verbose == VerbosityLevel.INFO

    def test_explicit_verbose(self):
        fit = _ConcreteFit(save_base_location="/tmp/test", verbose=VerbosityLevel.WARN)
        assert fit.verbose == VerbosityLevel.WARN


# ---------------------------------------------------------------------------
# add_fit
# ---------------------------------------------------------------------------

class TestAddFit:
    def test_adds_new_kc_to_fits(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        mock_stan_fit = MagicMock()
        fit.add_fit("kc_a", mock_stan_fit)
        assert "kc_a" in fit.fits
        assert fit.fits["kc_a"] is mock_stan_fit

    def test_adds_metadata_entry_for_new_kc(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        fit.add_fit("kc_a", MagicMock())
        assert "kc_a" in fit.fit_metadata

    def test_metadata_entry_contains_save_folder(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        fit.add_fit("kc_a", MagicMock())
        entry = fit.fit_metadata["kc_a"]
        assert "save_folder" in entry
        assert isinstance(entry["save_folder"], str)
        assert len(entry["save_folder"]) > 0

    def test_adding_multiple_kcs(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        fit.add_fit("kc_a", MagicMock())
        fit.add_fit("kc_b", MagicMock())
        assert set(fit.fits.keys()) == {"kc_a", "kc_b"}
        assert set(fit.fit_metadata.keys()) == {"kc_a", "kc_b"}

    def test_overwrite_replaces_fit(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        first = MagicMock()
        second = MagicMock()
        fit.add_fit("kc_a", first)
        fit.add_fit("kc_a", second)
        assert fit.fits["kc_a"] is second

    def test_overwrite_clears_summary_cache(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        fit.add_fit("kc_a", MagicMock())
        fit.summary_cache["kc_a"] = pd.DataFrame({"x": [1]})
        fit.add_fit("kc_a", MagicMock())
        assert "kc_a" not in fit.summary_cache

    def test_overwrite_does_not_clear_other_kc_cache(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        fit.add_fit("kc_a", MagicMock())
        fit.add_fit("kc_b", MagicMock())
        fit.summary_cache["kc_b"] = pd.DataFrame({"x": [1]})
        fit.add_fit("kc_a", MagicMock())
        assert "kc_b" in fit.summary_cache

    def test_overwrite_emits_warning(self, capsys):
        # Must use WARN verbose level so the warning is not suppressed
        fit = _ConcreteFit(save_base_location="/tmp/test", verbose=VerbosityLevel.WARN)
        fit.add_fit("kc_a", MagicMock())
        fit.add_fit("kc_a", MagicMock())
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "kc_a" in out


# ---------------------------------------------------------------------------
# update_summary_cache
# ---------------------------------------------------------------------------

class TestUpdateSummaryCache:
    def test_stores_dataframe(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        df = pd.DataFrame({"mean": [0.5], "std": [0.1]})
        fit.update_summary_cache("kc_a", df)
        pd.testing.assert_frame_equal(fit.summary_cache["kc_a"], df)

    def test_overwrites_existing_entry(self):
        fit = _ConcreteFit(save_base_location="/tmp/test")
        fit.update_summary_cache("kc_a", pd.DataFrame({"mean": [0.1]}))
        new_df = pd.DataFrame({"mean": [0.9]})
        fit.update_summary_cache("kc_a", new_df)
        pd.testing.assert_frame_equal(fit.summary_cache["kc_a"], new_df)


# ---------------------------------------------------------------------------
# _sanitize_kc_name
# ---------------------------------------------------------------------------

class TestSanitizeKCName:
    @pytest.mark.parametrize("kc, expected", [
        ("algebra", "algebra"),
        ("  spaces  ", "spaces"),
        ("kc name", "kc_name"),
        ("knowledge_component_1", "knowledge_component_1"),
        ("a.b.c", "a.b.c"),
        ("kc!@#", "kc"),          # special chars stripped to underscores, trailing _ stripped
        ("", "kc"),               # empty → default
        ("...", "kc"),            # dots allowed but stripped at edges → empty → default
        ("_leading", "leading"),  # leading underscore stripped
        ("trailing_", "trailing"),  # trailing underscore stripped
    ])
    def test_sanitize(self, kc, expected):
        assert BaseFit._sanitize_kc_name(kc) == expected

    def test_collapses_multiple_underscores(self):
        result = BaseFit._sanitize_kc_name("kc  name")  # 2 spaces → 2 underscores → 1
        assert "__" not in result
        assert result == "kc_name"

    def test_returns_string(self):
        assert isinstance(BaseFit._sanitize_kc_name("test"), str)


# ---------------------------------------------------------------------------
# _add_hash_suffix
# ---------------------------------------------------------------------------

class TestAddHashSuffix:
    def test_result_format(self):
        result = BaseFit._add_hash_suffix("kc_a", "kc_a")
        assert result == f"kc_a_{hashlib.sha256('kc_a'.encode()).hexdigest()[:8]}"

    def test_suffix_is_8_hex_chars(self):
        result = BaseFit._add_hash_suffix("some kc", "some_kc")
        suffix = result.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_deterministic(self):
        r1 = BaseFit._add_hash_suffix("kc_a", "kc_a")
        r2 = BaseFit._add_hash_suffix("kc_a", "kc_a")
        assert r1 == r2

    def test_different_originals_produce_different_suffixes(self):
        r1 = BaseFit._add_hash_suffix("kc_a", "kc")
        r2 = BaseFit._add_hash_suffix("kc_b", "kc")
        assert r1 != r2

    def test_sanitized_prefix_used(self):
        result = BaseFit._add_hash_suffix("kc_a", "my_sanitized_name")
        assert result.startswith("my_sanitized_name_")


# ---------------------------------------------------------------------------
# _get_fit_save_folder
# ---------------------------------------------------------------------------

class TestGetFitSaveFolder:
    def test_result_is_string(self):
        assert isinstance(BaseFit._get_fit_save_folder("kc_a"), str)

    def test_folder_is_non_empty(self):
        assert BaseFit._get_fit_save_folder("kc_a") != ""

    def test_deterministic(self):
        r1 = BaseFit._get_fit_save_folder("kc_a")
        r2 = BaseFit._get_fit_save_folder("kc_a")
        assert r1 == r2

    def test_different_kcs_produce_different_folders(self):
        r1 = BaseFit._get_fit_save_folder("kc_a")
        r2 = BaseFit._get_fit_save_folder("kc_b")
        assert r1 != r2

    def test_combines_sanitized_name_and_hash(self):
        result = BaseFit._get_fit_save_folder("algebra")
        sanitized = BaseFit._sanitize_kc_name("algebra")
        expected = BaseFit._add_hash_suffix("algebra", sanitized)
        assert result == expected


# ---------------------------------------------------------------------------
# fit_metadata_to_json / fit_metadata_from_json
# ---------------------------------------------------------------------------

class TestFitMetadataToJson:
    def test_produces_valid_json(self):
        metadata: FitMetadata = {"kc_a": {"save_folder": "kc_a_abc12345"}}
        raw = BaseFit.fit_metadata_to_json("mcmc", metadata)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_contains_fit_method(self):
        raw = BaseFit.fit_metadata_to_json("mcmc", {})
        parsed = json.loads(raw)
        assert parsed["fit_method"] == "mcmc"

    def test_contains_fits_key(self):
        metadata: FitMetadata = {"kc_a": {"save_folder": "folder_abc"}}
        raw = BaseFit.fit_metadata_to_json("mle", metadata)
        parsed = json.loads(raw)
        assert "fits" in parsed
        assert parsed["fits"]["kc_a"]["save_folder"] == "folder_abc"

    def test_returns_string(self):
        raw = BaseFit.fit_metadata_to_json("vb", {})
        assert isinstance(raw, str)


class TestFitMetadataFromJson:
    def _make_json(self, fit_method="mcmc", fits=None) -> str:
        if fits is None:
            fits = {"kc_a": {"save_folder": "kc_a_abc12345"}}
        return json.dumps({"fit_method": fit_method, "fits": fits})

    def test_round_trip(self):
        metadata: FitMetadata = {
            "kc_a": {"save_folder": "kc_a_abc12345"},
            "kc_b": {"save_folder": "kc_b_def67890"},
        }
        raw = BaseFit.fit_metadata_to_json("mcmc", metadata)
        loaded_method, loaded_metadata = BaseFit.fit_metadata_from_json(raw)
        assert loaded_method == "mcmc"
        assert loaded_metadata == metadata

    @pytest.mark.parametrize("method", ["mcmc", "mle", "vb", "pathfinder"])
    def test_accepts_all_valid_fit_methods(self, method):
        raw = self._make_json(fit_method=method)
        loaded_method, _ = BaseFit.fit_metadata_from_json(raw)
        assert loaded_method == method

    def test_raises_for_invalid_fit_method(self):
        raw = self._make_json(fit_method="sampling")
        with pytest.raises(ValueError, match="fit_method"):
            BaseFit.fit_metadata_from_json(raw)

    def test_raises_for_missing_fit_method(self):
        raw = json.dumps({"fits": {}})
        with pytest.raises(ValueError, match="fit_method"):
            BaseFit.fit_metadata_from_json(raw)

    def test_raises_for_non_dict_top_level(self):
        with pytest.raises(ValueError, match="top-level JSON must be an object"):
            BaseFit.fit_metadata_from_json(json.dumps([1, 2, 3]))

    def test_raises_when_fits_is_not_dict(self):
        raw = json.dumps({"fit_method": "mcmc", "fits": "not_a_dict"})
        with pytest.raises(ValueError, match="'fits' must be an object"):
            BaseFit.fit_metadata_from_json(raw)

    def test_raises_when_fits_is_missing(self):
        raw = json.dumps({"fit_method": "mcmc"})
        with pytest.raises(ValueError, match="'fits' must be an object"):
            BaseFit.fit_metadata_from_json(raw)

    def test_raises_when_entry_is_not_dict(self):
        raw = json.dumps({"fit_method": "mcmc", "fits": {"kc_a": "bad_entry"}})
        with pytest.raises(ValueError, match="must be an object"):
            BaseFit.fit_metadata_from_json(raw)

    def test_raises_when_save_folder_missing(self):
        raw = json.dumps({"fit_method": "mcmc", "fits": {"kc_a": {}}})
        with pytest.raises(ValueError, match="save_folder"):
            BaseFit.fit_metadata_from_json(raw)

    def test_raises_when_save_folder_is_not_string(self):
        raw = json.dumps({"fit_method": "mcmc", "fits": {"kc_a": {"save_folder": 123}}})
        with pytest.raises(ValueError, match="save_folder"):
            BaseFit.fit_metadata_from_json(raw)

    def test_empty_fits_dict(self):
        raw = self._make_json(fits={})
        loaded_method, loaded_metadata = BaseFit.fit_metadata_from_json(raw)
        assert loaded_method == "mcmc"
        assert loaded_metadata == {}
