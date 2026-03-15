from __future__ import annotations

from abc import ABC, abstractmethod
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from typing import Union

import pandas as pd
from cmdstanpy import from_csv as cmdstan_from_csv

from stanbkt.fits.fit_types import BaseCmdStanFit, FitMetadata, FitMethod, FitSaveFolder
from stanbkt.fits.persistence import (
    CACHE_SAVE_FOLDER,
    FIT_SAVE_FOLDER,
    METADATA_SAVE_FILE,
    add_hash_suffix,
    fit_metadata_from_json,
    fit_metadata_to_json,
    get_fit_save_folder,
    get_summary_cache_file,
    load_fit_artifacts,
    sanitize_kc_name,
    save_fit_artifacts,
)


# TODO: create_inits
# diagnose support for MCMC
# check API in __init__.py.
class BaseFit(VerboseMixin, ABC):
    """Base class for StanBKT fits.

    This class provides shared fit state management and delegates all persistence
    to :mod:`stanbkt.fits.persistence`.

    Attributes
    ----------
    fits : dict[str, BaseCmdStanFit]
        Mapping of knowledge component IDs to CmdStan fit objects.
    num_fitted_kcs : int
        Number of knowledge components that have been fitted.
    fit_metadata : FitMetadata
        Metadata used to resolve persisted fit folders.
    summary_cache : dict[str, pandas.DataFrame]
        Cached summary DataFrames for each knowledge component.

    """

    def __init__(
        self,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        fits: dict[str, BaseCmdStanFit] | None = None,
        fit_metadata: FitMetadata | None = None,
        summary_cache: dict[str, pd.DataFrame] | None = None,
    ):
        super().__init__(verbose=verbose)
        self.kc_fits: dict[str, BaseCmdStanFit] = (
            fits.copy() if fits is not None else {}
        )
        self.num_fitted_kcs = len(self.kc_fits)
        self.fit_metadata: FitMetadata = (
            fit_metadata
            if fit_metadata is not None
            else FitMetadata(fit_method=self._fit_method)
        )
        self.summary_cache: dict[str, pd.DataFrame] = (
            summary_cache.copy() if summary_cache is not None else {}
        )

    def add_fit(
        self, kc: str, fit: BaseCmdStanFit, overwrite_kcs: bool = False
    ) -> None:
        """Add a fit for a knowledge component to the model's fit state.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.
        fit : BaseCmdStanFit
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
        kc_fit_method = FitMethod.get_method_from_fit(fit)
        if kc_fit_method != self._fit_method:
            raise ValueError(
                (
                    f"Cannot add fit with method '{kc_fit_method.value}' when already fitted with method "
                    f"'{self.fit_metadata.fit_method.value}'."
                )
            )

        # check if there is already a fit for this KC
        if kc in self.kc_fits:
            if not overwrite_kcs:
                raise ValueError(
                    f"Fit for KC '{kc}' already exists. Set 'overwrite_kcs=True' to overwrite."
                )
            self._print(
                f"Overwriting existing fit for KC '{kc}'.", level=VerbosityLevel.WARN
            )
            self.summary_cache.pop(
                kc, None
            )  # remove summary cache for this updated KC fit
        self.kc_fits[kc] = fit
        self.num_fitted_kcs = len(self.kc_fits)
        # Presume we can create the save folder based on the KC.
        # Sid thinks this is a reasonable assumption since the base folder will be forced to be
        # unique, and the KCwill be sanitized to be filesystem safe.
        # Collisions will be handled by appending a hash suffix to the folder name.
        self.fit_metadata.fit_saves = {
            entry for entry in self.fit_metadata.fit_saves if entry.kc != kc
        }
        metadata_entry = FitSaveFolder(
            kc=kc,
            save_folder=BaseFit._get_fit_save_folder(kc),
            summary_cache_available=False,
        )
        self.fit_metadata.fit_saves.add(metadata_entry)

    def _update_summary_cache(self, kc: str, kc_summary_df: pd.DataFrame):
        if kc in self.summary_cache:
            self._print(
                f"Overwriting existing summary cache for KC '{kc}'.",
                level=VerbosityLevel.DEBUG,
            )
        self.summary_cache[kc] = kc_summary_df

    @classmethod
    def _load(cls, base_save_location: str) -> BaseFit:
        """Load fit artifacts from disk into a ``BaseFit`` subclass instance.

        Parameters
        ----------
        base_save_location : str
            Root folder containing persisted fit artifacts.

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
        )

        return cls(
            fits=fits,
            fit_metadata=loaded_fit_metadata,
            summary_cache=summary_cache,
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
        if not self.kc_fits:
            # users cannot remove fits once added, so empty means this model has never been fitted.
            self._print(
                "Model has not been fitted. Skipping fit save.",
                level=VerbosityLevel.WARN,
            )
            return

        stale_kcs = {
            fit_save.kc
            for fit_save in self.fit_metadata.fit_saves
            if fit_save.kc not in self.kc_fits
        }
        for stale_kc in stale_kcs:
            self._print(
                f"Fit metadata contains KC '{stale_kc}' that is not present in fits. Skipping save for this KC.",
                level=VerbosityLevel.DEBUG,
            )

        self.fit_metadata.fit_saves = {
            fit_save
            for fit_save in self.fit_metadata.fit_saves
            if fit_save.kc in self.kc_fits
        }

        self.fit_metadata = save_fit_artifacts(
            base_save_location=save_base_location,
            fits=self.kc_fits,
            fit_metadata=self.fit_metadata,
            summary_cache=self.summary_cache,
        )

    @property
    def _fit_method(self) -> FitMethod:
        raise NotImplementedError("Subclasses must implement the _fit_method property.")

    @staticmethod
    def _get_fit_save_folder(kc: str) -> str:
        """Compatibility wrapper for fit save folder naming helper.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.

        Returns
        -------
        str
            Deterministic fit save folder name.
        """
        return get_fit_save_folder(kc)

    @staticmethod
    def _sanitize_kc_name(kc: str) -> str:
        """Compatibility wrapper for KC sanitization helper.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.

        Returns
        -------
        str
            Filesystem-safe KC token.
        """
        return sanitize_kc_name(kc)

    @staticmethod
    def _add_hash_suffix(unsanitized_kc: str, sanitized_kc: str) -> str:
        """Compatibility wrapper for KC hash-suffix helper.

        Parameters
        ----------
        unsanitized_kc : str
            Original KC identifier.
        sanitized_kc : str
            Sanitized KC identifier.

        Returns
        -------
        str
            Hash-suffixed KC folder token.
        """
        return add_hash_suffix(unsanitized_kc, sanitized_kc)

    @staticmethod
    def _get_summary_cache_file(kc: str) -> str:
        """Compatibility wrapper for summary cache filename helper.

        Parameters
        ----------
        kc : str
            Knowledge component identifier.

        Returns
        -------
        str
            Summary cache CSV filename for the KC.
        """
        return get_summary_cache_file(kc)

    @staticmethod
    def fit_metadata_to_json(fit_metadata: FitMetadata, *, indent: int = 2) -> str:
        """Compatibility wrapper for metadata serialization helper.

        Parameters
        ----------
        fit_metadata : FitMetadata
            Fit metadata object.
        indent : int, default=2
            JSON indentation level.

        Returns
        -------
        str
            Serialized metadata JSON string.
        """
        return fit_metadata_to_json(fit_metadata, indent=indent)

    @staticmethod
    def fit_metadata_from_json(raw_text: str) -> FitMetadata:
        """Compatibility wrapper for metadata deserialization helper.

        Parameters
        ----------
        raw_text : str
            Serialized metadata JSON string.

        Returns
        -------
        FitMetadata
            Parsed metadata object.
        """
        return fit_metadata_from_json(raw_text)

    # TODO: check if all fit types support create_inits
    # if so I can move create_inits to the base class and implement it here.
    @abstractmethod
    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        raise NotImplementedError("Subclasses must implement the _create_inits method.")

    @abstractmethod
    def summary(self, kc: Union[list[str], str]) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the summary method.")
