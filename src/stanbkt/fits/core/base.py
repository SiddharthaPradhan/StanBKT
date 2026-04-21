from __future__ import annotations
from dataclasses import replace
import os

from abc import ABC, abstractmethod
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from typing import Union

import pandas as pd
from cmdstanpy import from_csv as cmdstan_from_csv

from stanbkt.fits.fit_types import CmdStanFit, FitMetadata, FitMethod, FitSaveFolder
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


# TODO: create_inits
# diagnose support for MCMC
# check API in __init__.py.
class FitBase(VerboseMixin, ABC):
    """Base class for StanBKT fits.

    This class provides shared fit state management and delegates all persistence
    to :mod:`stanbkt.fits.persistence`.

    Attributes
    ----------
    fits : dict[str, CmdStanFit]
        Mapping of knowledge component IDs to CmdStan fit objects.
    num_fitted_kcs : int
        Number of knowledge components that have been fitted.
    _fit_metadata : FitMetadata
        Metadata used to resolve persisted fit folders.
    _summary_cache : dict[str, pd.DataFrame]
        Cached summary DataFrames for each knowledge component.
    _summary_percentiles : tuple[float, float], default (2.5, 97.5)
        Percentiles used for generating summary statistics. Values should be in range [1, 99].

    """

    def __init__(
        self,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        fits: dict[str, CmdStanFit] | None = None,
        fit_metadata: FitMetadata | None = None,
        fit_artifact_base_location: str | None = None,
        cache_summary: bool = True,
        summary_percentiles: tuple[float, float] = (2.5, 97.5),
        _summary_cache: dict[str, pd.DataFrame] | None = None,
    ):
        """Initialize fit container state.

        Parameters
        ----------
        verbose : VerbosityLevel, default VerbosityLevel.INFO
            Verbosity level used for logging.
        fits : dict[str, CmdStanFit] | None, optional
            Existing per-KC fit objects to initialize from.
        fit_metadata : FitMetadata | None, optional
            Persisted metadata for this fit collection.
        fit_artifact_base_location : str | None, optional
            Optional folder where per-KC CmdStan CSV artifacts are persisted. When set,
            evicted fits can be lazily reloaded from disk.
        cache_summary : bool, default True
            Whether generated summaries should be cached in memory.
        summary_percentiles : tuple[float, float], default (2.5, 97.5)
            Default percentile bounds used by summary computations.
        _summary_cache : dict[str, pd.DataFrame] | None, optional
            Existing summary cache keyed by KC.
        """
        super().__init__(verbose=verbose)
        self.stan_fits: dict[str, CmdStanFit] = fits.copy() if fits is not None else {}
        self.num_fitted_kcs = len(self.stan_fits)
        self._fit_metadata: FitMetadata = (
            fit_metadata
            if fit_metadata is not None
            else FitMetadata(
                fit_method=self._fit_method,
                summary_percentiles=summary_percentiles,
            )
        )
        self._summary_cache: dict[str, pd.DataFrame] = (
            _summary_cache.copy() if _summary_cache is not None else {}
        )
        self._summary_percentiles: tuple[float, float] = (
            self._fit_metadata.summary_percentiles
        )
        self._should_cache_summary: bool = cache_summary
        self._fit_artifact_base_location: str | None = fit_artifact_base_location
        self.num_fitted_kcs = len(self.get_fitted_kcs())

    def __str__(self) -> str:
        """Return a user-friendly string representation of the fit."""
        class_name = self.__class__.__name__
        lines = [
            f"{class_name}(",
            f"  fit_method={self._fit_method.value}",
            f"  num_kcs={self.num_fitted_kcs}",
        ]

        if self.num_fitted_kcs > 0:
            kc_list = sorted(self.get_fitted_kcs())
            if len(kc_list) <= 5:
                lines.append(f"  kcs={kc_list}")
            else:
                lines.append(f"  kcs={kc_list[:5]} ... ({len(kc_list) - 5} more)")

        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation of the fit."""
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"num_fitted_kcs={self.num_fitted_kcs}, "
            f"fit_method={self._fit_method!r}, "
            f"verbose={self.verbose!r})"
        )

    def add_fit(self, kc: str, fit: CmdStanFit, overwrite_kcs: bool = False) -> None:
        """Add a fit for a knowledge component to the model's fit state.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.
        fit : CmdStanFit
            CmdStan fit object to add for the KC.
        overwrite_kcs : bool, default=False
            Whether to overwrite existing fits for KCs that are being added again.

        Raises
        ------
        ValueError
            If the fit's method is incompatible with the model's fit method, or if a fit
            for the KC already exists and ``overwrite_kcs=False``.

        """
        # check if the fit method matches the class's fit method
        kc_fit_method = FitMethod.infer_fit_method_from_stan_fit(fit)
        if kc_fit_method != self._fit_method:
            raise ValueError(
                (
                    f"Cannot add fit with method '{kc_fit_method.value}' when already fitted with method "
                    f"'{self._fit_metadata.fit_method.value}'."
                )
            )

        # check if there is already a fit for this KC
        if self.has_kc(kc):
            if not overwrite_kcs:
                raise ValueError(
                    f"Fit for KC '{kc}' already exists. Set 'overwrite_kcs=True' to overwrite."
                )
            self.log(
                f"Overwriting existing fit for KC '{kc}'.", level=VerbosityLevel.WARN
            )
            if self._should_cache_summary:
                self._summary_cache.pop(
                    kc, None
                )  # remove summary cache for this updated KC fit
            self.stan_fits.pop(kc, None)
        self.stan_fits[kc] = fit
        # Presume we can create the save folder based on the KC.
        # Sid thinks this is a reasonable assumption since the base folder will be forced to be
        # unique, and the KC will be sanitized to be filesystem safe.
        # Collisions will be handled by appending a hash suffix to the folder name.
        self._fit_metadata.fit_saves.pop(kc, None)
        self._fit_metadata.fit_saves[kc] = FitSaveFolder(
            kc=kc,
            save_folder=get_fit_save_folder(kc),
            summary_cache_available=False,
        )
        self.num_fitted_kcs = len(self.get_fitted_kcs())

    def get_fitted_kcs(self) -> set[str]:
        """Return all known fitted KCs, including disk-only entries."""
        return set(self._fit_metadata.fit_saves.keys()).union(self.stan_fits.keys())

    def set_fit_artifact_base_location(
        self, base_location: str | os.PathLike[str]
    ) -> None:
        """Configure where fit artifacts are stored for lazy reloading."""
        self._fit_artifact_base_location = os.fspath(base_location)

    def release_fit_from_memory(self, kc: str) -> None:
        """Persist a KC fit to disk and evict the in-memory CmdStan fit object.

        Raises
        ------
        RuntimeError
            If no artifact base location has been configured.
        KeyError
            If the KC does not exist or is not currently loaded in memory.
        """
        if self._fit_artifact_base_location is None:
            raise RuntimeError(
                "Fit artifact base location is not configured. "
                "Set it before releasing fits from memory."
            )
        if kc not in self.get_fitted_kcs():
            raise KeyError(f"No fit found for KC '{kc}'.")
        if kc not in self.stan_fits:
            raise KeyError(
                f"Fit for KC '{kc}' is not currently loaded in memory and cannot be released."
            )

        existing_metadata = self._fit_metadata
        saved_single_metadata = save_fit_artifacts(
            base_save_location=self._fit_artifact_base_location,
            fits={kc: self.stan_fits[kc]},
            fit_metadata=replace(
                existing_metadata,
                fit_saves={kc: existing_metadata.fit_saves[kc]},
            ),
            summary_cache={
                kc: self._summary_cache[kc] for kc in [kc] if kc in self._summary_cache
            },
        )
        existing_metadata.fit_saves[kc] = saved_single_metadata.fit_saves[kc]
        self._fit_metadata = existing_metadata
        self.stan_fits.pop(kc, None)
        self.num_fitted_kcs = len(self.get_fitted_kcs())

    def get_fit(self, kc: str) -> CmdStanFit:
        """Get the fit for a knowledge component.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.

        Returns
        -------
        CmdStanFit
            CmdStan fit object for the specified KC.

        Raises
        ------
        KeyError
            If no fit exists for the specified KC.
        """
        if kc not in self.stan_fits:
            if kc not in self._fit_metadata.fit_saves:
                raise KeyError(f"No fit found for KC '{kc}'.")
            if self._fit_artifact_base_location is None:
                raise KeyError(
                    f"Fit for KC '{kc}' is not loaded in memory and no artifact location is configured."
                )

            fit_save = self._fit_metadata.fit_saves[kc]
            kc_fit_save_folder = os.path.join(
                self._fit_artifact_base_location,
                FIT_SAVE_FOLDER,
                str(fit_save.save_folder),
            )
            loaded_fit = cmdstan_from_csv(kc_fit_save_folder)
            self.stan_fits[kc] = loaded_fit
        return self.stan_fits[kc]

    def has_kc(self, kc: str) -> bool:
        """Check if a fit exists for a knowledge component.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.

        Returns
        -------
        bool
            True if a fit exists for the specified KC, False otherwise.
        """
        return kc in self.get_fitted_kcs()

    def _clear_summary_cache_if_stale(
        self, percentiles: tuple[float, float], force=False
    ) -> None:
        """Clear the summary cache if ``percentiles`` differ from the cached percentiles.

        Should be called at the start of ``summary()`` before any cache reads.
        """
        if force or percentiles != self._summary_percentiles:
            self.log(
                f"Percentiles {percentiles} differ from cached summary percentiles "
                f"{self._summary_percentiles}. Clearing summary cache.",
                level=VerbosityLevel.DEBUG,
            )
            self._summary_percentiles = percentiles
            self._fit_metadata.summary_percentiles = percentiles
            self._summary_cache.clear()

    def _update_summary_cache(self, kc: str, kc_summary_df: pd.DataFrame) -> None:
        """Insert or replace summary cache entry and update metadata.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.
        kc_summary_df : pd.DataFrame
            Summary DataFrame for the KC.
        """
        if kc in self._summary_cache:
            self.log(
                f"Overwriting existing summary cache for KC '{kc}'.",
                level=VerbosityLevel.DEBUG,
            )
        if kc not in self._fit_metadata.fit_saves:
            fit_save_entry = FitSaveFolder(
                kc=kc,
                save_folder=get_fit_save_folder(kc),
                summary_cache_available=True,
            )
        else:
            fit_save_entry = replace(
                self._fit_metadata.fit_saves[kc], summary_cache_available=True
            )
        self._fit_metadata.fit_saves[kc] = fit_save_entry
        self._summary_cache[kc] = kc_summary_df

    @classmethod
    def _load(cls, base_save_location: str, lazy: bool = False) -> FitBase:
        """Load fit artifacts from disk into a ``BaseFit`` subclass instance.

        Parameters
        ----------
        base_save_location : str
            Root folder containing persisted fit artifacts.
        lazy : bool, default False
            Whether to defer loading CmdStan fit objects until first access.

        Returns
        -------
        BaseFit
            Instantiated subclass populated with loaded fits and cache.

        Raises
        ------
        FileNotFoundError
            If required artifact files are missing.
        ValueError
            If metadata fit method mismatches subclass method.
        """
        expected_fit_method = cls()._fit_method
        loaded_fit_metadata, fits, summary_cache = load_fit_artifacts(
            base_save_location=base_save_location,
            expected_fit_method=expected_fit_method,
            lazy=lazy,
        )

        return cls(
            fits=fits,
            fit_metadata=loaded_fit_metadata,
            fit_artifact_base_location=base_save_location,
            _summary_cache=summary_cache,
        )

    def _save(self, save_base_location: str) -> None:
        """Save fits, summary cache, and metadata to disk.

        Parameters
        ----------
        save_base_location : str
            Root folder to save fit artifacts.

        Returns
        -------
        None
        """
        # save fits if not empty
        if not self.get_fitted_kcs():
            # users cannot remove fits once added, so empty means this model has never been fitted.
            self.log(
                "Model has not been fitted. Skipping fit save.",
                level=VerbosityLevel.WARN,
            )
            return

        # Ensure persisted-only fits are loaded before writing into a new save location.
        for kc in self.get_fitted_kcs():
            if kc not in self.stan_fits:
                self.get_fit(kc)

        stale_kcs = {
            fit_save.kc
            for fit_save in self._fit_metadata.fit_saves.values()
            if fit_save.kc not in self.stan_fits
        }
        for stale_kc in stale_kcs:
            self.log(
                f"Fit metadata contains KC '{stale_kc}' that is not present in fits. Skipping save for this KC.",
                level=VerbosityLevel.DEBUG,
            )

        self._fit_metadata.fit_saves = {
            fit_save.kc: fit_save
            for fit_save in self._fit_metadata.fit_saves.values()
            if fit_save.kc in self.stan_fits
        }

        self._fit_metadata = save_fit_artifacts(
            base_save_location=save_base_location,
            fits=self.stan_fits,
            fit_metadata=self._fit_metadata,
            summary_cache=self._summary_cache,
        )
        self.set_fit_artifact_base_location(save_base_location)

    def summary(
        self,
        kcs: Union[list[str], str, None] = None,
        kc_col_name: str = "kc_id",
        percentiles: tuple[float, float] = (2.5, 97.5),
    ) -> pd.DataFrame:
        """Public wrapper that delegates to the subclass :meth:`_summary` implementation."""
        return self._summary(kcs=kcs, kc_col_name=kc_col_name, percentiles=percentiles)

    @property
    def _fit_method(self) -> FitMethod:
        """Return the fitting method implemented by the subclass."""
        raise NotImplementedError("Subclasses must implement the _fit_method property.")

    @abstractmethod
    def _summary(
        self,
        kcs: Union[list[str], str, None] = None,
        kc_col_name: str = "kc_id",
        percentiles: tuple[float, float] = (2.5, 97.5),
    ) -> pd.DataFrame:
        """Return per-KC fit summary statistics.

        Parameters
        ----------
        kcs : Union[list[str], str, None], optional
            KCs to summarize; if None, summarize all fitted KCs.
        kc_col_name : str, default "kc_id"
            Name for the KC column in returned output.
        percentiles : tuple[float, float], default (2.5, 97.5)
            Percentile bounds to include in summary output.

        Returns
        -------
        pd.DataFrame
            Summary table indexed or labeled by KC and parameter.
        """
        raise NotImplementedError("Subclasses must implement the _summary method.")

    # TODO: check if all fit types support create_inits
    # if so I can move create_inits to the base class and implement it here.
    @abstractmethod
    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        """Create initialization payload for CmdStanPy fitting routines.

        Parameters
        ----------
        kc : Union[list[str], str, None], optional
            KC identifier(s) used to shape initialization payloads.

        Returns
        -------
        object
            CmdStanPy-compatible initialization object.
        """
        raise NotImplementedError("Subclasses must implement the _create_inits method.")
