from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from abc import ABC, abstractmethod
import hashlib
import json
import os
import re
from typing import TypedDict, Union, cast, Literal
from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanPathfinder
import pandas as pd
from cmdstanpy import from_csv as cmdstan_from_csv
import pickle

BaseCmdStanFit = Union[CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanPathfinder]
FIT_METHOD = Literal["mcmc", "mle", "vb", "pathfinder"]


class FitMetadataEntry(TypedDict):
    save_folder: str


FitMetadata = dict[str, FitMetadataEntry]


class FitMetadataRoot(TypedDict):
    fit_method: FIT_METHOD
    fits: FitMetadata


FIT_SAVE_FOLDER = "model_fits"  # base folder for fit saves.
SUMMARY_CACHE_SAVE_FILE = (
    "summary_cache.pickle"  # location where summary cache is saved
)
METADATA_SAVE_FILE = "fit_metadata.json"  # location where fit metadata is saved

# TODO: create_inits
# diagnose support for MCMC


class BaseFit(VerboseMixin, ABC):
    """Base class for StanBKT fits.

    This class serves as a base for different types of fits (i.e., MCMC, MLE, VB, Pathfinder).

    Attributes:
        save_base_location (str): The base location where fit files are saved.
        fits (dict[str, BaseCmdStanFit]): A dictionary mapping KCs to CmdStan fit objects.
        summary_cache (dict[str, pd.DataFrame]): A cache for storing summary DataFrames for each fit, keyed by KCs.

    """

    def __init__(
        self, save_base_location: str, verbose: VerbosityLevel = VerbosityLevel.INFO
    ):
        super().__init__(verbose=verbose)
        self.save_base_location: str = save_base_location
        self.fits: dict[str, BaseCmdStanFit] = dict()
        self.fit_metadata: FitMetadata = dict()
        self.summary_cache: dict[str, pd.DataFrame] = dict()

    def add_fit(self, kc: str, fit: BaseCmdStanFit):
        # check if there is already a fit for this KC
        if kc in self.fits:
            self._print(
                f"Overwriting existing fit for KC '{kc}'.", level=VerbosityLevel.WARN
            )
            self.summary_cache.pop(
                kc, None
            )  # remove summary cache for this updated KC fit
        self.fits[kc] = fit
        # Presume we can create the save folder based on the KC.
        # Sid thinks this is a reasonable assumption since the base folder will be forced to be
        # unique, and the KC will be sanitized to be filesystem safe.
        # Collisions will be handled by appending a hash suffix to the folder name.
        metadata_entry: FitMetadataEntry = {
            "save_folder": BaseFit._get_fit_save_folder(kc),
        }
        self.fit_metadata[kc] = metadata_entry

    def update_summary_cache(self, kc: str, kc_summary_df: pd.DataFrame):
        self.summary_cache[kc] = kc_summary_df

    def _load(self, base_save_location: str) -> tuple[
        FitMetadata,  # fit metadata
        dict[str, BaseCmdStanFit],  # fits
        dict[str, pd.DataFrame] | None,  # summary cache
    ]:
        if not os.path.exists(base_save_location):
            raise FileNotFoundError(
                f"Fit save location '{base_save_location}' does not exist."
            )
        fits: dict[str, BaseCmdStanFit] = dict()
        summary_cache: dict[str, pd.DataFrame] | None = dict()
        # try to load summary cache if it exists
        summary_cache_path = os.path.join(base_save_location, SUMMARY_CACHE_SAVE_FILE)
        if os.path.exists(summary_cache_path):
            with open(summary_cache_path, "rb") as f:
                summary_cache = pickle.load(f)
        else:
            summary_cache = None
        fit_metadata_path = os.path.join(base_save_location, METADATA_SAVE_FILE)
        fit_metadata: FitMetadata = dict()
        if os.path.exists(fit_metadata_path):
            with open(fit_metadata_path, "r", encoding="utf-8") as f:
                loaded_fit_method, fit_metadata = self.fit_metadata_from_json(f.read())
            if loaded_fit_method != self._fit_method:
                self._print(
                    (
                        f"Loaded fit metadata method '{loaded_fit_method}' does not match "
                        f"current fit method '{self._fit_method}'."
                    ),
                    level=VerbosityLevel.WARN,
                )

        for kc, metadata in fit_metadata.items():
            kc_fit_save_folder = os.path.join(
                base_save_location, metadata["save_folder"]
            )
            if os.path.exists(kc_fit_save_folder):
                loaded_fit = cmdstan_from_csv(kc_fit_save_folder)
                fits[kc] = cast(BaseCmdStanFit, loaded_fit)

        return fit_metadata, fits, summary_cache

    def _save(self) -> None:
        """Saves the fits, summary cache, and fit metadata to disk."""
        # create the save directory if it doesn't exist
        os.makedirs(self.save_base_location, exist_ok=True)

        # save summary cache if not empty
        if self.summary_cache:
            summary_cache_path = os.path.join(
                self.save_base_location, SUMMARY_CACHE_SAVE_FILE
            )
            with open(summary_cache_path, "wb") as f:
                pickle.dump(self.summary_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        # save fits if not empty
        if self.fits:
            for kc, fit in self.fits.items():
                metadata_entry: FitMetadataEntry | None = self.fit_metadata.get(kc)
                if (
                    metadata_entry is None
                ):  # should not be possible since add_fit creates metadata entries and populates self.fits_metadata.
                    created_entry: FitMetadataEntry = {
                        "save_folder": BaseFit._get_fit_save_folder(kc),
                    }
                    self.fit_metadata[kc] = created_entry
                    metadata_entry = created_entry

                kc_fit_save_folder = os.path.join(
                    self.save_base_location, metadata_entry["save_folder"]
                )
                fit.save_csvfiles(kc_fit_save_folder)

        if self.fit_metadata:
            fit_metadata_path = os.path.join(
                self.save_base_location, METADATA_SAVE_FILE
            )
            with open(fit_metadata_path, "w", encoding="utf-8") as f:
                f.write(self.fit_metadata_to_json(self._fit_method, self.fit_metadata))

    @property
    def _fit_method(self) -> FIT_METHOD:
        raise NotImplementedError("Subclasses must implement the _fit_method property.")

    @staticmethod
    def _get_fit_save_folder(kc: str) -> str:
        sanitized_kc = BaseFit._sanitize_kc_name(kc)
        return BaseFit._add_hash_suffix(kc, sanitized_kc)

    @staticmethod
    def _sanitize_kc_name(kc: str) -> str:
        """Converts a KC string into a filesystem-safe folder token."""
        normalized = re.sub(r"\s+", "_", kc.strip())
        sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", normalized)
        sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
        # if the sanitized name is empty, use a default name with a hash suffix to ensure uniqueness
        if not sanitized:
            sanitized = "kc"
        return sanitized

    @staticmethod
    def _add_hash_suffix(unsanitized_kc: str, sanitized_kc: str) -> str:
        """Appends a hash suffix to the sanitized KC name to ensure uniqueness."""
        hash_suffix = hashlib.sha256(unsanitized_kc.encode()).hexdigest()[:8]
        return f"{sanitized_kc}_{hash_suffix}"

    @staticmethod
    def fit_metadata_to_json(
        fit_method: FIT_METHOD, fit_metadata: FitMetadata, *, indent: int = 2
    ) -> str:
        data: FitMetadataRoot = {
            "fit_method": fit_method,
            "fits": fit_metadata,
        }
        return json.dumps(data, indent=indent, sort_keys=True, default=str)

    @staticmethod
    def fit_metadata_from_json(raw_text: str) -> tuple[FIT_METHOD, FitMetadata]:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError(
                "Error parsing fit metadata: top-level JSON must be an object."
            )

        fit_method = data.get("fit_method")
        fits = data.get("fits")

        valid_fit_methods: set[FIT_METHOD] = {"mcmc", "mle", "vb", "pathfinder"}
        if fit_method not in valid_fit_methods:
            raise ValueError(
                "Error parsing fit metadata: 'fit_method' must be one of "
                "'mcmc', 'mle', 'vb', or 'pathfinder'."
            )
        if not isinstance(fits, dict):
            raise ValueError(
                "Error parsing fit metadata: top-level key 'fits' must be an object."
            )

        parsed: FitMetadata = {}
        for kc, entry in fits.items():
            if not isinstance(kc, str):
                raise ValueError(
                    "Error parsing fit metadata: each fit metadata key (KC) must be a string."
                )
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Error parsing fit metadata: metadata for KC '{kc}' must be an object."
                )

            save_folder = entry.get("save_folder")
            if not isinstance(save_folder, str):
                raise ValueError(
                    f"Error parsing fit metadata: metadata for KC '{kc}' must include string field 'save_folder'."
                )

            parsed[kc] = {
                "save_folder": save_folder,
            }

        return cast(FIT_METHOD, fit_method), parsed

    # TODO: check if all fit types support create_inits
    # if so I can move create_inits to the base class and implement it here.
    @abstractmethod
    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        raise NotImplementedError("Subclasses must implement the _create_inits method.")

    @abstractmethod
    def summary(self, kc: Union[list[str], str]) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement the summary method.")
