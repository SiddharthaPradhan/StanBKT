import os

import pandas as pd
import pytest

import stanbkt.fits.persistence.fit_io as fit_io
from stanbkt.fits.fit_types import FitMetadata, FitMethod, FitSaveFolder


class _DummySavedFit:
    def save_csvfiles(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "mock_chain.csv"), "w", encoding="utf-8") as f:
            f.write("lp__\n0\n")


class _DummyLoadedFit:
    pass


class TestSaveFitArtifacts:
    def test_save_fit_artifacts_skips_metadata_entries_without_fit(self, tmp_path):
        save_dir = tmp_path / "fit_saves"

        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                FitSaveFolder(kc="kc_with_fit", save_folder="fit_folder"),
                FitSaveFolder(kc="kc_without_fit", save_folder="missing_folder"),
            },
        )

        updated = fit_io.save_fit_artifacts(
            base_save_location=str(save_dir),
            fits={"kc_with_fit": _DummySavedFit()},  # ty:ignore[invalid-argument-type]
            fit_metadata=metadata,
            summary_cache={"kc_with_fit": pd.DataFrame({"mean": [0.2]})},
        )

        assert {entry.kc for entry in updated.fit_saves} == {"kc_with_fit"}

    def test_save_fit_artifacts_removes_stale_cache_csv(self, tmp_path):
        save_dir = tmp_path / "fit_saves"
        fits_root = save_dir / fit_io.FIT_SAVE_FOLDER
        cache_root = fits_root / fit_io.CACHE_SAVE_FOLDER
        cache_root.mkdir(parents=True, exist_ok=True)

        kc = "kc_a"
        cache_file = cache_root / fit_io.get_summary_cache_file(kc)
        cache_file.write_text("mean\n0.1\n", encoding="utf-8")

        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                FitSaveFolder(kc=kc, save_folder=fit_io.get_fit_save_folder(kc))
            },
        )

        fit_io.save_fit_artifacts(
            base_save_location=str(save_dir),
            fits={kc: _DummySavedFit()},  # ty:ignore[invalid-argument-type]
            fit_metadata=metadata,
            summary_cache={},
        )

        assert not cache_file.exists()


class TestLoadFitArtifacts:
    def test_load_fit_artifacts_does_not_require_cache_file_when_flag_false(
        self, tmp_path, monkeypatch
    ):
        save_dir = tmp_path / "fit_saves"
        kc = "kc_a"
        folder = fit_io.get_fit_save_folder(kc)

        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                FitSaveFolder(
                    kc=kc,
                    save_folder=folder,
                    summary_cache_available=False,
                )
            },
        )

        fit_io.save_fit_artifacts(
            base_save_location=str(save_dir),
            fits={kc: _DummySavedFit()},  # ty:ignore[invalid-argument-type]
            fit_metadata=metadata,
            summary_cache={},
        )

        monkeypatch.setattr(fit_io, "CmdStanFit", (_DummyLoadedFit,))
        monkeypatch.setattr(fit_io, "cmdstan_from_csv", lambda _: _DummyLoadedFit())

        loaded_metadata, fits, summary_cache = fit_io.load_fit_artifacts(
            base_save_location=str(save_dir),
            expected_fit_method=FitMethod.MCMC,
        )

        assert {entry.kc for entry in loaded_metadata.fit_saves} == {kc}
        assert set(fits.keys()) == {kc}
        assert summary_cache == {}

    def test_load_fit_artifacts_retains_kc_when_summary_cache_csv_is_malformed(
        self, tmp_path, monkeypatch
    ):
        save_dir = tmp_path / "fit_saves"
        kc = "kc_a"
        folder = fit_io.get_fit_save_folder(kc)

        metadata = FitMetadata(
            fit_method=FitMethod.MCMC,
            fit_saves={
                FitSaveFolder(
                    kc=kc,
                    save_folder=folder,
                    summary_cache_available=True,
                )
            },
        )
        dummy_fit = _DummySavedFit()
        fit_io.save_fit_artifacts(
            base_save_location=str(save_dir),
            fits={kc: dummy_fit},  # ty:ignore[invalid-argument-type]
            fit_metadata=metadata,
            summary_cache={kc: pd.DataFrame({"mean": [0.5]})},
        )

        cache_path = (
            save_dir
            / fit_io.FIT_SAVE_FOLDER
            / fit_io.CACHE_SAVE_FOLDER
            / fit_io.get_summary_cache_file(kc)
        )
        # Write invalid bytes to force parser failure.
        cache_path.write_bytes(b"\x00\x81\x82")

        monkeypatch.setattr(fit_io, "CmdStanFit", (_DummyLoadedFit,))
        monkeypatch.setattr(fit_io, "cmdstan_from_csv", lambda _: _DummyLoadedFit())

        with pytest.warns(UserWarning, match="Failed to load summary cache"):
            loaded_metadata, fits, summary_cache = fit_io.load_fit_artifacts(
                base_save_location=str(save_dir),
                expected_fit_method=FitMethod.MCMC,
            )

        # Loaded fit instances are reconstructed from disk, so identity differs.
        assert set(fits.keys()) == {kc}
        assert isinstance(fits[kc], _DummyLoadedFit)
        assert summary_cache == {}
        assert loaded_metadata.fit_saves.pop().summary_cache_available == False
