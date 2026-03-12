import pytest

import stanbkt.fits.fit_types as fit_types
from stanbkt.fits.fit_types import FitMetadata, FitMethod, FitSaveFolder


class TestFitMethodResolution:
    def test_get_method_from_fit_resolves_all_supported_methods(self, monkeypatch):
        class _DummyMCMC:
            pass

        class _DummyMLE:
            pass

        class _DummyVB:
            pass

        class _DummyPathfinder:
            pass

        monkeypatch.setattr(fit_types, "CmdStanMCMC", _DummyMCMC)
        monkeypatch.setattr(fit_types, "CmdStanMLE", _DummyMLE)
        monkeypatch.setattr(fit_types, "CmdStanVB", _DummyVB)
        monkeypatch.setattr(fit_types, "CmdStanPathfinder", _DummyPathfinder)

        assert FitMethod.get_method_from_fit(_DummyMCMC()) == FitMethod.MCMC
        assert FitMethod.get_method_from_fit(_DummyMLE()) == FitMethod.MLE
        assert FitMethod.get_method_from_fit(_DummyVB()) == FitMethod.VB
        assert FitMethod.get_method_from_fit(_DummyPathfinder()) == FitMethod.PATHFINDER

    def test_get_method_from_fit_raises_for_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported fit type"):
            FitMethod.get_method_from_fit(object())


class TestFitMetadataTypes:
    def test_fit_save_folder_default_cache_flag_is_false(self):
        entry = FitSaveFolder(kc="kc_a", save_folder="kc_a_12345678")
        assert entry.summary_cache_available is False

    def test_fit_save_folder_is_hashable_for_set_usage(self):
        first = FitSaveFolder(
            kc="kc_a", save_folder="folder", summary_cache_available=True
        )
        second = FitSaveFolder(
            kc="kc_a", save_folder="folder", summary_cache_available=True
        )
        assert {first, second} == {first}

    def test_fit_metadata_can_hold_multiple_entries(self):
        entry_a = FitSaveFolder(kc="kc_a", save_folder="a")
        entry_b = FitSaveFolder(kc="kc_b", save_folder="b")
        metadata = FitMetadata(fit_method=FitMethod.MCMC, fit_saves={entry_a, entry_b})
        assert metadata.fit_method == FitMethod.MCMC
        assert {entry.kc for entry in metadata.fit_saves} == {"kc_a", "kc_b"}
