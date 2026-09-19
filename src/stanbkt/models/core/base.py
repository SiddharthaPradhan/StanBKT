"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.

"""

from __future__ import annotations
import warnings
from collections import deque
from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from natsort import natsort_keygen, natsorted
import re
from stanbkt.fits.fit_factory import FitFactory
import json

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, Literal, Optional, Tuple, Union, Final, Callable
import numpy as np
import numpy.typing as npt
import cmdstanpy as csp
import pandas as pd
import os
import tempfile
from numba import njit, prange
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from stanbkt.fits.fit_options import MCMCFitOptions, PFFitOptions, StanFitOptions
from stanbkt.fits.fit_types import CmdStanFit, FitSaveEntry
from stanbkt.fits.fit_types import FitMethod
from stanbkt.fits.core.base import FitBase as BaseFit
from stanbkt.models.model_types import (
    ModelType,
    InitKnowledgeStrategy,
)
from stanbkt.models.error import FitMethodMismatchError
from stanbkt.models.priors import PriorsBase
from stanbkt.models.predictions import predict_posterior
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import (
    iter_kc_data,
    prepare_student_covariates,
    ColumnNames,
    KCData,
    _DEFAULT_KC_ID,
    _NA_FILL_VALUE,
    _PKNOW,
    _PCORRECT,
)
from stanbkt.utils.model_archive import pack_model_directory
from stanbkt.utils.summary_utils import label_summary_index
from stanbkt.utils.posterior_utils import (
    gq_to_draws,
    _process_single_kc_gq,
    _summarize_single_kc_gq,
)

# cache njit dispatchers for the point-estimate prediction kernels, keyed by
# (predictor function, fastmath, parallel) so repeated predict() calls with the same
# flags reuse the same compiled Dispatcher instead of re-wrapping it every call.
_POINT_ESTIMATE_NUMBA_KERNEL_CACHE: dict[tuple[Callable, bool, bool], Callable] = {}


def _get_point_estimate_numba_kernel(
    state_predictor: Callable, fast_math: bool, parallel: bool
) -> Callable:
    cache_key = (state_predictor, fast_math, parallel)
    kernel = _POINT_ESTIMATE_NUMBA_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        kernel = njit(fastmath=fast_math, parallel=parallel, cache=True)(state_predictor)
        _POINT_ESTIMATE_NUMBA_KERNEL_CACHE[cache_key] = kernel
    return kernel


class BKTModelBase(VerboseMixin, ABC):
    """Abstract base class for Stan Bayesian Knowledge Tracing (BKT) models.

    This class defines the interface that all Stan BKT model implementations must follow.

    Attributes
    ----------
    fit_method : FitMethod
        The method used for fitting the model (e.g., MCMC, VB, MAP).
    individual_initial_knowledge : bool
        Whether to initial know states are individualized to the student. If False,
        a single initial knowledge parameter is estimated for all students.
    initital_knowledge_strategy : InitKnowledgeStrategy
        Strategy for estimating initial knowledge. This is only applicable if `individual_initial_knowledge` is True.
        When  this is set to `CORRECTNESS_ONLY`, only the correctness data is used to estimate initial knowledge.
        When `JOINT`, model requires student level covariate (e.g. pretest) to jointly estimate initial knowledge (uses correctness and covariate).
        For example, `CORRECTNESS_ONLY` uses only the correctness of the first interaction for each student to inform
        initial knowledge estimates, while `FIRST_INTERACTION` uses the correctness of the first interaction with each KC for each student.
    verbose : VerbosityLevel
        Verbosity level for logging.
    stan_compile_kwargs : dict
        Additional keyword arguments for Stan model compilation.
    cpp_compile_kwargs : dict
        Additional keyword arguments for C++ compilation of the Stan model.
    low_memory : bool
        Whether to evict each KC's fit to disk after fitting, reloading it lazily on
        next access. Reduces peak memory usage when fitting many KCs at the cost of
        some performance.
    """

    def __init__(
        self,
        fit_method: FitMethod | str = FitMethod.MCMC,
        individual_initial_knowledge: bool = False,
        init_knowledge_strategy: InitKnowledgeStrategy = InitKnowledgeStrategy.CORRECTNESS_ONLY,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: Optional[Dict[str, Any]] = None,
        cpp_compile_kwargs: Optional[Dict[str, Any]] = None,
        low_memory: bool = False,
    ):

        # verify if initial_knowledge_strategy is valid given the individual_initial_knowledge setting
        if (
            not individual_initial_knowledge
            and init_knowledge_strategy != InitKnowledgeStrategy.CORRECTNESS_ONLY
        ):
            raise ValueError(
                f"Invalid combination of 'individual_initial_knowledge' and 'init_knowledge_strategy'. "
                f"When 'individual_initial_knowledge' is False, 'init_knowledge_strategy' must be 'CORRECTNESS_ONLY'. "
                f"Got individual_initial_knowledge={individual_initial_knowledge} and init_knowledge_strategy={init_knowledge_strategy}."
            )
        super().__init__(verbose=verbose)
        resolved_fit_method = FitMethod(fit_method)
        self._fit_method: Final[FitMethod] = resolved_fit_method
        self.fit_class: Final[type[BaseFit]] = FitFactory.get_fit_class_from_method(
            resolved_fit_method
        )
        # TODO: NEED defaults for compile kwargs?
        # Does this need to be a dataclass?
        # TODO: catch the Error thrown and re-raise with custom error for invalid
        self.stan_compile_kwargs: dict[str, Any] = stan_compile_kwargs or {}
        self.cpp_compile_kwargs: dict[str, Any] = cpp_compile_kwargs or {}
        self.individual_initial_knowledge: bool = individual_initial_knowledge
        self.init_knowledge_strategy: InitKnowledgeStrategy = init_knowledge_strategy
        # Flag to control whether the model uses group-specific parameters
        self._use_groups: bool = False
        # Model is instantiated lazily during first fit and cached
        self._stan_model: Optional[csp.CmdStanModel] = None
        self._hidden_states_model: Optional[csp.CmdStanModel] = None
        self._smoothed_hidden_states_model: Optional[csp.CmdStanModel] = None
        self.fits: BaseFit = self.fit_class()
        self._is_fitted: bool = False
        self.low_memory: bool = low_memory
        self._fit_artifact_tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def __str__(self) -> str:
        """Return a user-friendly string representation of the model."""
        class_name = self.__class__.__name__
        fit_status = "fitted" if self._is_fitted else "not fitted"
        num_kcs = self.fits.num_fitted_kcs if self._is_fitted else 0

        lines = [
            f"{class_name}(",
            f"  fit_method={self._fit_method.value}",
            f"  status={fit_status}",
        ]

        if self._is_fitted and num_kcs > 0:
            lines.append(f"  num_kcs={num_kcs}")

        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation of the model."""
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"fit_method={self._fit_method!r}, "
            f"verbose={self.verbose!r}, "
            f"is_fitted={self._is_fitted})"
        )

    def fit(
        self,
        data: pd.DataFrame,
        priors: Optional[dict[str, PriorsBase] | PriorsBase] = None,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        stan_fit_options: Optional[Union[StanFitOptions, dict[str, Any]]] = None,
        overwrite_kcs: bool = False,
        student_covariates: Optional[pd.DataFrame] = None,
        n_kcs_workers: int = -1,
    ) -> BKTModelBase:
        """
        Fit the BKT model to data. Each KC is fitted independently with its own model.
        Additional KCs can be fitted by calling fit again with new data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the training data. Must include columns for:
            Student ID, Problem ID, and Correctness (0/1).
            If the KC column is absent, all interactions are assumed to belong to a single knowledge component.
        priors : dict[str, BayesianPriors] or BayesianPriors, optional
            Prior specifications for the model parameters. Can be provided as:
            - A single BayesianPriors object applied to all KCs.
            - A dictionary mapping KC IDs to their specific BayesianPriors.
            If None, default priors will be used for all KCs.
        column_mapping : dict, optional
            Mapping of expected column names. Keys should be 'student_id', 'problem_id', 'correct', and 'kc_id'.
            If None, default column names are used.
        stan_fit_options : StanFitOptions or dict, optional
                Additional keyword arguments to pass to the Stan fitting method. If a dict is passed, it will be forwarded as-is to the CmdStanPy fit method.
                It is recommended to use the typed :class:`StanFitOptions` for better type checking and validation. The accepted options depend on the chosen fit method. For example:
                - MCMC parameters (e.g., iter_sampling, chains, seed)
                - VB parameters (e.g., iter, tol_rel_obj)
                If None, default fitting options for the chosen fit method will be used.
        overwrite_kcs : bool, default=False
            Whether to overwrite existing fits for KCs that are already fitted.
            If False, an error will be raised if attempting to fit a KC that already has a fit.
            If True, existing fits for the same KCs will be overwritten with the new fits.
        student_covariates : pd.DataFrame, optional
            One row per student with covariate columns, required when
            ``init_knowledge_strategy=InitKnowledgeStrategy.JOINT``. All columns
            other than the student ID column are treated as covariates, in the
            DataFrame's own column order.
        n_kcs_workers : int, default=-1
            Number of KCs fitted concurrently. Each KC is an independent Stan run, so results
            match a sequential fit. The Stan settings take priority: one fit already uses
            ``parallel_chains * threads_per_chain`` cores for MCMC (``num_threads`` for
            Pathfinder, 1 otherwise). With ``-1`` (auto) the number of workers is the available
            CPUs divided by that, capped by the number of KCs, and the Stan settings are never
            changed. A positive value is used as given, but ``n_kcs_workers`` times the cores
            per fit must not exceed the available CPUs, otherwise a ``ValueError`` is raised.
            ``1`` always fits the KCs sequentially.

        Returns
        -------
        BKTModelBase
            The fitted BKT model instance.

        Raises
        ------
        ValueError
            If data validation fails or incompatible cpp_compile_kwargs and stan_fit_options.
        """
        if self.low_memory:
            if self._fit_artifact_tmpdir is None:
                self._fit_artifact_tmpdir = tempfile.TemporaryDirectory(
                    prefix="stanbkt_fit_cache_"
                )
            self.fits.set_fit_artifact_base_location(self._fit_artifact_tmpdir.name)
            self.fits.release_after_summary = True

        if self._stan_model is None:
            self._compile_model(self._stan_model_filename)

        # ensure _compile_model succeeded before proceeding
        # this helps with: (1) type checking and  (2) catches any missed compilation failures
        if self._stan_model is None:
            raise RuntimeError(
                "Stan model compilation failed. The model object is None after compilation. "
                "Ensure the Stan source file is valid and that CmdStanPy is correctly installed."
            )

        # validate priors
        if priors is None:
            # use default priors for all KCs
            self.log(
                "No priors provided, using default priors for all KCs.",
                level=VerbosityLevel.DEBUG,
            )
            priors = self._default_priors()
        else:
            self._default_priors_class()._validate(
                priors, type(self), self.init_knowledge_strategy
            )

        # check stan_fit_options
        if stan_fit_options is None:
            stan_fit_options = FitFactory.create_default_fit_options(self._fit_method)
        else:
            # convert to StanFitOptions if it is a dict, mainly for better type checking and validation,
            # but also to ensure compatibility with the FitFactory verification method
            if isinstance(stan_fit_options, dict):
                stan_fit_options = FitFactory.create_fit_options_from_dict(
                    stan_fit_options, self._fit_method
                )
            # verify compatibility of provided options with the fit method
            FitFactory.verify_fit_options_compatibility(
                stan_fit_options,
                self._fit_method,
                cpp_compile_kwargs=self.cpp_compile_kwargs,
            )

        resolved_mapping = ColumnNames.apply_default_mapping(column_mapping)
        prepared_covariates, covariate_columns = self._prepare_joint_covariates(
            data, student_covariates, resolved_mapping
        )

        data_kcs = self._data_kc_ids(data, resolved_mapping)
        n_kcs_workers = self._resolve_n_kcs_workers(
            n_kcs_workers, self._stan_cores_per_fit(stan_fit_options), len(data_kcs)
        )

        def _fit_kc(kc_id, kc_data):
            # the `priors.get` is valid but `ty` is not currently smart enough, hence the ignore
            kc_priors = (
                priors.get(
                    str(kc_id), self._default_priors()
                )  # ty:ignore[no-matching-overload]
                if isinstance(priors, dict)
                else priors
            )
            data_dict = self._build_stan_data_dict(kc_data, kc_priors)
            return self._fit_stan_model_using_method(
                data_dict=data_dict, fit_options=stan_fit_options
            )

        def _record_kc(kc_id, kc_data, fit_result):
            self.fits.add_fit(
                str(kc_id),
                fit_result,
                overwrite_kcs=overwrite_kcs,
                group2index=kc_data.group_2_index,
                groups=(
                    set(kc_data.group_2_index.keys())
                    if kc_data.group_2_index is not None
                    else None
                ),
                student2index=(
                    {str(sid): i + 1 for i, sid in enumerate(kc_data.student_ids)}
                    if self.individual_initial_knowledge
                    else None
                ),
                covariate_columns=kc_data.covariate_columns,
            )
            if self.low_memory:
                self.fits.release_fit_from_memory(str(kc_id))
            self.log(f"Finished fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)
            self._is_fitted = True

        # fail before any KC is fitted so no work is wasted on an existing KC later in the data
        if not overwrite_kcs:
            for kc in data_kcs:
                if self.fits.has_kc(kc):
                    raise ValueError(
                        f"Fit for KC '{kc}' already exists. Set 'overwrite=True' to overwrite."
                    )

        # fits are recorded in KC order so the fit state matches a sequential run
        pending: deque = deque()
        with ThreadPoolExecutor(max_workers=n_kcs_workers) as executor:
            try:
                for kc_id, kc_data in iter_kc_data(
                    data=data,
                    col_mapping=resolved_mapping,
                    return_groups=self._use_groups,
                    print_fn=self.log,
                    student_covariates=prepared_covariates,
                    covariate_columns=covariate_columns,
                ):
                    self.log(f"Fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)
                    if n_kcs_workers == 1:
                        _record_kc(kc_id, kc_data, _fit_kc(kc_id, kc_data))
                        continue

                    pending.append(
                        (kc_id, kc_data, executor.submit(_fit_kc, kc_id, kc_data))
                    )
                    if len(pending) >= n_kcs_workers:
                        done_kc_id, done_kc_data, future = pending.popleft()
                        _record_kc(done_kc_id, done_kc_data, future.result())

                while pending:
                    done_kc_id, done_kc_data, future = pending.popleft()
                    _record_kc(done_kc_id, done_kc_data, future.result())
            except BaseException:
                for _, _, future in pending:
                    future.cancel()
                raise
        return self

    def _get_fit_save_entry(self, kc_id: str) -> FitSaveEntry | None:
        get_entry_fn = getattr(self.fits, "get_fit_save_entry", None)
        if not callable(get_entry_fn):
            return None
        fit_save_entry = get_entry_fn(kc_id)
        if isinstance(fit_save_entry, FitSaveEntry):
            return fit_save_entry
        return None

    def _align_kc_group_indices_with_fit_metadata(
        self, kc_id: str, kc_data: KCData
    ) -> KCData:
        if not self._use_groups:
            return kc_data

        fit_save_entry = self._get_fit_save_entry(str(kc_id))
        if fit_save_entry is None or fit_save_entry.group2index in (None, {}):
            return kc_data

        if kc_data.groups is None or kc_data.group_2_index is None:
            raise ValueError(
                f"KC '{kc_id}' requires group data for prediction because fit metadata contains group indices."
            )

        fit_group2index = {
            str(group_name): int(index)
            for group_name, index in fit_save_entry.group2index.items()
        }
        trained_groups = (
            {str(group_name) for group_name in fit_save_entry.groups}
            if fit_save_entry.groups not in (None, set())
            else set(fit_group2index.keys())
        )
        incoming_groups = {
            str(group_name) for group_name in kc_data.group_2_index.keys()
        }

        unknown_groups = incoming_groups - trained_groups
        if unknown_groups:
            raise ValueError(
                f"Prediction data for KC '{kc_id}' contains unseen groups: {sorted(unknown_groups)}. "
                f"Expected groups from fit metadata: {sorted(trained_groups)}."
            )

        index2group = {
            int(index): str(group_name)
            for group_name, index in kc_data.group_2_index.items()
        }
        aligned_groups = np.empty(kc_data.groups.shape, dtype=np.int32)
        for i, group_index in enumerate(kc_data.groups):
            group_name = index2group.get(int(group_index))
            if group_name is None:
                raise ValueError(
                    f"Prediction data for KC '{kc_id}' has group index '{int(group_index)}' with no group mapping."
                )
            if group_name not in fit_group2index:
                raise ValueError(
                    f"Prediction data for KC '{kc_id}' contains unseen group '{group_name}'."
                )
            aligned_groups[i] = fit_group2index[group_name]

        return replace(
            kc_data,
            groups=aligned_groups,
            group_2_index=dict(fit_group2index),
        )

    def _prepare_joint_covariates(
        self,
        data: pd.DataFrame,
        student_covariates: Optional[pd.DataFrame],
        resolved_mapping: dict[str, str],
    ) -> tuple[Optional[pd.DataFrame], Optional[list[str]]]:
        """Validate and index student covariates for the JOINT strategy.

        Returns ``(None, None)`` when the model isn't using JOINT. Raises if
        JOINT is active and no covariates were supplied.
        """
        if self.init_knowledge_strategy != InitKnowledgeStrategy.JOINT:
            return None, None
        if student_covariates is None:
            raise ValueError(
                "'student_covariates' must be provided when init_knowledge_strategy="
                "InitKnowledgeStrategy.JOINT."
            )
        student_col = resolved_mapping[ColumnNames.STUDENT_ID]
        all_students = data[student_col].astype(str).unique()
        return prepare_student_covariates(
            student_covariates, all_students, student_id_col=student_col
        )

    def _check_covariate_columns_match(
        self, kc_id: str, covariate_columns: Optional[list[str]]
    ) -> None:
        """Raise if predict-time covariate columns/order differ from the fitted KC's."""
        fit_save_entry = self._get_fit_save_entry(kc_id)
        expected = (
            list(fit_save_entry.covariate_columns)
            if fit_save_entry is not None and fit_save_entry.covariate_columns is not None
            else None
        )
        actual = list(covariate_columns) if covariate_columns is not None else None
        if expected is not None and expected != actual:
            raise ValueError(
                f"'student_covariates' columns for KC '{kc_id}' do not match the columns "
                f"used at fit time. Expected {expected}, got {actual}."
            )

    @abstractmethod
    def _default_priors(self) -> PriorsBase:
        """Return default priors for the model parameters."""
        raise NotImplementedError(
            "Subclasses must implement the _default_priors method to provide default priors."
        )

    @abstractmethod
    def _default_priors_class(self) -> type[PriorsBase]:
        """Return default priors class for the model parameters."""
        raise NotImplementedError(
            "Subclasses must implement the _default_priors_class method to provide default priors class."
        )

    def _fit_stan_model_using_method(
        self, data_dict: dict[str, Any], fit_options: StanFitOptions
    ) -> CmdStanFit:
        if self._stan_model is None:
            raise RuntimeError("Stan model is not compiled. Cannot fit the model.")

        if self._fit_method == FitMethod.MCMC:
            return self._stan_model.sample(data=data_dict, **fit_options.to_dict())
        elif self._fit_method == FitMethod.VB:
            return self._stan_model.variational(data=data_dict, **fit_options.to_dict())
        elif self._fit_method == FitMethod.MLE:
            return self._stan_model.optimize(data=data_dict, **fit_options.to_dict())
        elif self._fit_method == FitMethod.PATHFINDER:
            return self._stan_model.pathfinder(data=data_dict, **fit_options.to_dict())
        else:
            raise ValueError(
                f"Invalid fitting method '{self._fit_method}'. Supported methods are '{FitMethod.MCMC}', '{FitMethod.VB}', '{FitMethod.MLE}', and '{FitMethod.PATHFINDER}'."
            )

    def summary(
        self,
        kcs: Union[list[str], str, None] = None,
        percentiles: Tuple[float, float] = (2.5, 97.5),
        column_mapping: dict[str, str] = {},
        clear_cache: bool = False,
        label_indexes: bool = True,
    ) -> Any:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        kcs : Union[list[str], str, None], optional
            KCs to summarize. Can be a single KC string, a list of KCs, or None.
            If None, summarizes all fitted KCs.
        percentiles : tuple of float, default=(2.5, 97.5)
            Percentiles to include in summary. Values should be in range [1, 99].
            Ignored when the fit method is MLE, MLE produces a point estimate only.
        clear_cache : bool, default=False
            Whether to refresh the cached summaries.
        label_indexes : bool, default=True
            Whether to replace Stan indexes (``learn[2]``) with the group, student or
            covariate they represent. The index becomes ``[kc, parameter, axis, label]``.
            Set to False for the raw CmdStan parameter names.

        Returns
        -------
        pandas.DataFrame
            Summary statistics.

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        self._fit_check("summary")
        if self._fit_method == FitMethod.MLE and percentiles != (2.5, 97.5):
            warnings.warn(
                "percentiles is ignored for MLE fits. MLE produces a point estimate only.",
                UserWarning,
                stacklevel=2,
            )
        if clear_cache:
            self.fits._clear_summary_cache_if_stale(percentiles, force=True)

        # validate percentiles
        if not (
            len(percentiles) == 2
            and (1 <= percentiles[0] <= 99)
            and (1 <= percentiles[1] <= 99)
            and (percentiles[0] < percentiles[1])
        ):
            raise ValueError(
                f"'percentiles' must be a tuple of two numeric values between 1 and 99 (inclusive), where the first value is less than the second. Got {percentiles}."
            )
        kc_col_name = column_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID)

        summary_df = self.fits._summary(
            kcs=kcs,
            percentiles=percentiles,
            kc_col_name=kc_col_name,
        )
        if not label_indexes:
            return summary_df

        kc_entries = {
            str(kc): self._get_fit_save_entry(str(kc))
            for kc in summary_df.index.get_level_values(0).unique()
        }
        return label_summary_index(
            summary_df,
            group2index={kc: e.group2index if e else None for kc, e in kc_entries.items()},
            student2index={
                kc: e.student2index if e else None for kc, e in kc_entries.items()
            },
            covariate_columns={
                kc: e.covariate_columns if e else None for kc, e in kc_entries.items()
            },
            group_col_name=column_mapping.get(ColumnNames.GROUP, ColumnNames.GROUP),
            student_col_name=column_mapping.get(
                ColumnNames.STUDENT_ID, ColumnNames.STUDENT_ID
            ),
        )

    def _fit_check(self, referrer: Optional[str] = None) -> None:
        """Check if model has been fitted."""
        if not self._is_fitted or self.fits.num_fitted_kcs == 0:
            raise RuntimeError(
                f"Model must be fitted before calling {referrer + '()' if referrer else 'this method'}"
            )

    def _get_model_init_kwargs(self) -> dict[str, Any]:
        """Return constructor kwargs required to reconstruct this model instance."""
        return {
            "fit_method": self._fit_method.value,
            "verbose": int(self.verbose),
            "stan_compile_kwargs": self.stan_compile_kwargs,
            "cpp_compile_kwargs": self.cpp_compile_kwargs,
            "low_memory": self.low_memory,
            "individual_initial_knowledge": self.individual_initial_knowledge,
            "init_knowledge_strategy": self.init_knowledge_strategy.value,
        }

    def save(self, save_base_location: str | os.PathLike[str]) -> None:
        """Save fitted model artifacts to a compressed archive.

        Parameters
        ----------
        save_base_location : str | os.PathLike[str]
            Archive path where fitted model artifacts should be saved.

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        self._fit_check()
        archive_path = os.fspath(save_base_location)

        with tempfile.TemporaryDirectory(prefix="stanbkt_save_") as temp_dir:
            self.fits._save(temp_dir)

            model_metadata = {
                "model_module": self.__class__.__module__,
                "model_qualname": self.__class__.__qualname__,
                "model_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                "model_init_kwargs": self._get_model_init_kwargs(),
            }
            model_metadata_path = os.path.join(temp_dir, "model_metadata.json")
            with open(
                model_metadata_path, "w", encoding="utf-8"
            ) as model_metadata_file:
                json.dump(model_metadata, model_metadata_file, indent=2, sort_keys=True)

            pack_model_directory(temp_dir, archive_path)

    def check_data_contains_fitted_kcs(self, kcs: set[str]) -> None:
        """Check if data contains any KC that was fitted.
        Raises an error if data contains KCs that were not fitted.
        """
        self._fit_check()
        fitted_kcs: set[str] = self.fits.get_fitted_kcs()
        if not len(self.get_kcs_in_fitted_kcs(kcs)) > 0:
            raise ValueError(
                f"Data contains no KCs that were previously fitted. Given KCs: {kcs}, fitted KCs: {fitted_kcs}"
            )
        kcs_not_fitted = kcs - fitted_kcs
        if kcs_not_fitted:
            self.log(
                f"Data contains {len(kcs_not_fitted)} KCs that were not fitted.",
                level=VerbosityLevel.WARN,
            )
            self.log(
                f"{list(kcs_not_fitted)} do not have fits.",
                level=VerbosityLevel.INFO,
            )

    def get_kcs_in_fitted_kcs(self, kcs: set[str]) -> set[str]:
        """Return the set of KCs in the data that were fitted previously."""
        self._fit_check()
        fitted_kcs: set[str] = self.fits.get_fitted_kcs()
        return kcs.intersection(fitted_kcs)

    @staticmethod
    def _empty_point_estimate_prediction_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                ColumnNames.KC_ID,
                ColumnNames.STUDENT_ID,
                ColumnNames.PROBLEM_ID,
                "pKnow",
                "pCorrectness",
                ColumnNames.CORRECTNESS,
            ]
        )

    def _drop_unseen_students(
        self, data: pd.DataFrame, resolved_mapping: dict[str, str]
    ) -> pd.DataFrame:
        """Drop students missing from each KC's fit when initial knowledge is a free per-student parameter.

        Only applies to ``individual_initial_knowledge`` with CORRECTNESS_ONLY, where an
        unseen student has no fitted value to use.
        """
        if (
            not self.individual_initial_knowledge
            or self.init_knowledge_strategy == InitKnowledgeStrategy.JOINT
        ):
            return data
        kc_str = data[resolved_mapping[ColumnNames.KC_ID]].astype(str)
        student_str = data[resolved_mapping[ColumnNames.STUDENT_ID]].astype(str)
        keep = pd.Series(True, index=data.index)
        dropped: set[str] = set()
        for kc in kc_str.unique():
            fit_save_entry = self._get_fit_save_entry(kc)
            if fit_save_entry is None or not fit_save_entry.student2index:
                continue
            unseen = (kc_str == kc) & ~student_str.isin(fit_save_entry.student2index)
            keep &= ~unseen
            dropped.update(student_str[unseen].unique())
        if not keep.any():
            raise ValueError(
                "None of the students in 'data' were part of the fit. Individualized "
                "initial knowledge with CORRECTNESS_ONLY can only predict fitted students."
            )
        if dropped:
            shown = natsorted(dropped)[:10]
            more = f" and {len(dropped) - 10} more" if len(dropped) > 10 else ""
            warnings.warn(
                f"Dropping {len(dropped)} student(s) that were not part of the fit: "
                f"{shown}{more}.",
                UserWarning,
                stacklevel=3,
            )
        return data.loc[keep]

    def _prepare_point_estimate_prediction_inputs(
        self,
        data: Optional[pd.DataFrame],
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ],
        point_estimate: Literal["mean", "median", "mode"],
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError("'point_estimate' must be 'mean', 'median', or 'mode'.")

        resolved_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = resolved_mapping[ColumnNames.KC_ID]

        working_data = data
        if kc_column_name not in working_data.columns:
            working_data = working_data.copy()
            working_data[kc_column_name] = _DEFAULT_KC_ID

        observed_kcs = set(working_data[kc_column_name].astype(str).unique())
        self.check_data_contains_fitted_kcs(observed_kcs)
        overlapping_kcs = self.get_kcs_in_fitted_kcs(observed_kcs)
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()
        filtered_data = self._drop_unseen_students(filtered_data, resolved_mapping)

        return filtered_data, resolved_mapping

    def _predict_point_estimate_common(
        self,
        data: Optional[pd.DataFrame],
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ],
        point_estimate: Literal["mean", "median", "mode"],
        parallel: bool,
        fast_math: bool,
        state_predictor: Callable,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        filtered_data, resolved_mapping = (
            self._prepare_point_estimate_prediction_inputs(
                data=data,
                column_mapping=column_mapping,
                point_estimate=point_estimate,
            )
        )

        if filtered_data.empty:
            return self._empty_point_estimate_prediction_frame()

        prepared_covariates, covariate_columns = self._prepare_joint_covariates(
            filtered_data, student_covariates, resolved_mapping
        )

        njit_predict_numba: Callable = _get_point_estimate_numba_kernel(
            state_predictor, fast_math, parallel
        )

        predictions: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=filtered_data,
            col_mapping=resolved_mapping,
            return_groups=self._use_groups,
            print_fn=self.log,
            student_covariates=prepared_covariates,
            covariate_columns=covariate_columns,
        ):
            kc_id_str = str(kc_id)
            kc_data = self._align_kc_group_indices_with_fit_metadata(
                kc_id_str, kc_data
            )
            if kc_data.covariates is not None:
                self._check_covariate_columns_match(
                    kc_id_str, kc_data.covariate_columns
                )
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
                groups=kc_data.groups if self._use_groups else None,
                kc_data=kc_data,
                kc_id=kc_id_str,
            )
            p_know, p_correctness = njit_predict_numba(
                correctness=kc_data.correctness,
                prior=prior,
                learn=learn,
                forget=forget,
                guess=guess,
                slip=slip,
                lengths=kc_data.lengths,
            )
            kc_predictions = self._state_arrays_to_long_df(
                p_know=p_know,
                p_correctness=p_correctness,
                kc_data=kc_data,
                correctness_col_name=resolved_mapping.get(
                    ColumnNames.CORRECTNESS, ColumnNames.CORRECTNESS
                ),
            )
            kc_predictions.insert(0, ColumnNames.KC_ID, str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
            return self._empty_point_estimate_prediction_frame()

        return pd.concat(predictions, ignore_index=True)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        parallel: bool = True,
        fast_math: bool = True,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict hidden states using point-estimate parameters from fitted posteriors."""
        self._fit_check(referrer="predict")
        return self._predict_point_estimate_common(
            data=data,
            column_mapping=column_mapping,
            point_estimate=point_estimate,
            parallel=parallel,
            fast_math=fast_math,
            state_predictor=type(self)._predict_hidden_states_numba,
            student_covariates=student_covariates,
        )

    def predict_smoothed(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        parallel: bool = True,
        fast_math: bool = True,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict smoothed hidden states using point-estimate parameters."""
        self._fit_check(referrer="predict_smoothed")
        return self._predict_point_estimate_common(
            data=data,
            column_mapping=column_mapping,
            point_estimate=point_estimate,
            parallel=parallel,
            fast_math=fast_math,
            state_predictor=type(self)._predict_hidden_states_smoothed_numba,
            student_covariates=student_covariates,
        )

    @staticmethod
    def _predict_hidden_states_numba(
        correctness: npt.NDArray[np.float64],
        prior: npt.NDArray[np.float64],
        learn: npt.NDArray[np.float64],
        forget: npt.NDArray[np.float64],
        guess: npt.NDArray[np.float64],
        slip: npt.NDArray[np.float64],
        lengths: npt.NDArray[np.int64],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated deterministic forward recursion for point-estimate prediction."""
        n_students, n_problems = correctness.shape
        p_know: npt.NDArray[np.float64] = np.full(
            (n_students, n_problems), _NA_FILL_VALUE, dtype=np.float64
        )
        p_correctness: npt.NDArray[np.float64] = np.full(
            (n_students, n_problems), _NA_FILL_VALUE, dtype=np.float64
        )

        for student_idx in prange(n_students):  # ty:ignore[not-iterable]
            prior_s = prior[student_idx]
            learn_s = learn[student_idx]
            forget_s = forget[student_idx]
            guess_s = guess[student_idx]
            slip_s = slip[student_idx]

            one_minus_slip = 1.0 - slip_s
            one_minus_guess = 1.0 - guess_s
            one_minus_forget = 1.0 - forget_s

            p_know[student_idx, 0] = prior_s
            p_correctness[student_idx, 0] = (
                prior_s * one_minus_slip + (1.0 - prior_s) * guess_s
            )

            for problem_idx in prange(
                lengths[student_idx] - 1
            ):  # ty:ignore[not-iterable]
                current_p_know = p_know[student_idx, problem_idx]
                if correctness[student_idx, problem_idx]:
                    numerator = current_p_know * one_minus_slip
                    denominator = numerator + (1.0 - current_p_know) * guess_s
                else:
                    numerator = current_p_know * slip_s
                    denominator = numerator + (1.0 - current_p_know) * one_minus_guess

                p_know_given_obs = numerator / denominator
                next_p_know = (
                    p_know_given_obs * one_minus_forget
                    + (1.0 - p_know_given_obs) * learn_s
                )
                p_know[student_idx, problem_idx + 1] = next_p_know
                p_correctness[student_idx, problem_idx + 1] = (
                    next_p_know * one_minus_slip + (1.0 - next_p_know) * guess_s
                )

        return p_know, p_correctness

    @staticmethod
    def _predict_hidden_states_smoothed_numba(
        correctness: np.ndarray,
        prior: np.ndarray,
        learn: np.ndarray,
        forget: np.ndarray,
        guess: np.ndarray,
        slip: np.ndarray,
        lengths: npt.NDArray[np.int64],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated forward-backward recursion for smoothed state probabilities."""
        n_students, n_problems = correctness.shape
        p_smooth = np.full((n_students, n_problems), _NA_FILL_VALUE, dtype=np.float64)
        p_correctness = np.full(
            (n_students, n_problems), _NA_FILL_VALUE, dtype=np.float64
        )

        for s in prange(n_students):  # type: ignore
            student_interaction_len = lengths[s]
            prior_s = prior[s]
            p_t = learn[s]
            p_f = forget[s]
            p_g = guess[s]
            p_s = slip[s]

            one_minus_p_s = 1.0 - p_s
            one_minus_p_g = 1.0 - p_g

            e0 = np.full(student_interaction_len, _NA_FILL_VALUE, dtype=np.float64)
            e1 = np.full(student_interaction_len, _NA_FILL_VALUE, dtype=np.float64)
            for t in prange(student_interaction_len):  # type: ignore
                if correctness[s, t] != 0:
                    e1[t] = one_minus_p_s
                    e0[t] = p_g
                else:
                    e1[t] = p_s
                    e0[t] = one_minus_p_g

            alpha0 = np.empty(student_interaction_len, dtype=np.float64)
            alpha1 = np.empty(student_interaction_len, dtype=np.float64)
            scale = np.empty(student_interaction_len, dtype=np.float64)

            a0 = (1.0 - prior_s) * e0[0]
            a1 = prior_s * e1[0]
            c0 = a0 + a1
            if c0 == 0.0:
                c0 = 1e-15
            a0 /= c0
            a1 /= c0
            alpha0[0] = a0
            alpha1[0] = a1
            scale[0] = c0

            for t in range(1, student_interaction_len):
                prev0 = alpha0[t - 1]
                prev1 = alpha1[t - 1]

                p_l0 = (1.0 - p_t) * prev0 + p_f * prev1
                p_l1 = p_t * prev0 + (1.0 - p_f) * prev1

                a0 = p_l0 * e0[t]
                a1 = p_l1 * e1[t]
                ct = a0 + a1
                if ct == 0.0:
                    ct = 1e-15
                a0 /= ct
                a1 /= ct

                alpha0[t] = a0
                alpha1[t] = a1
                scale[t] = ct

            beta0 = np.empty(student_interaction_len, dtype=np.float64)
            beta1 = np.empty(student_interaction_len, dtype=np.float64)
            beta0[student_interaction_len - 1] = 1.0
            beta1[student_interaction_len - 1] = 1.0

            for t in range(student_interaction_len - 2, -1, -1):
                b0_next = beta0[t + 1]
                b1_next = beta1[t + 1]

                b0 = (1.0 - p_t) * e0[t + 1] * b0_next + p_t * e1[t + 1] * b1_next
                b1 = p_f * e0[t + 1] * b0_next + (1.0 - p_f) * e1[t + 1] * b1_next

                ct_inv = 1.0 / scale[t + 1]
                b0 *= ct_inv
                b1 *= ct_inv

                beta0[t] = b0
                beta1[t] = b1

            for t in range(student_interaction_len):
                g0 = alpha0[t] * beta0[t]
                g1 = alpha1[t] * beta1[t]
                norm = g0 + g1
                if norm == 0.0:
                    norm = 1e-15
                p_smooth[s, t] = g1 / norm
                p_correctness[s, t] = (
                    p_smooth[s, t] * one_minus_p_s + (1.0 - p_smooth[s, t]) * p_g
                )

        return p_smooth, p_correctness

    @staticmethod
    def _state_arrays_to_long_df(
        p_know: np.ndarray,
        p_correctness: np.ndarray,
        kc_data: KCData,
        correctness_col_name: str = ColumnNames.CORRECTNESS,
    ) -> pd.DataFrame:
        """Convert dense state arrays to long-form prediction output using only valid entries."""
        # Ragged "arrays" for valid non-na entries
        student_id_segs: list[npt.NDArray] = []
        problem_id_segs: list[npt.NDArray] = []
        p_know_segs: list[npt.NDArray[np.float64]] = []
        p_correctness_segs: list[npt.NDArray[np.float64]] = []
        correctness_segs: list[npt.NDArray[np.int8]] = []

        for student_idx, (student_id, interaction) in enumerate(
            kc_data.student_inter_dict.items()
        ):
            length = interaction.length
            student_id_segs.append(np.full(length, str(student_id), dtype=object))
            problem_id_segs.append(np.asarray(interaction.problem_ids, dtype=object))
            p_know_segs.append(p_know[student_idx, :length].astype(np.float64))
            p_correctness_segs.append(
                p_correctness[student_idx, :length].astype(np.float64)
            )
            correctness_segs.append(
                kc_data.correctness[student_idx, :length].astype(np.int8)
            )

        student_ids = (
            np.concatenate(student_id_segs)
            if student_id_segs
            else np.empty(0, dtype=object)
        )
        problem_ids = (
            np.concatenate(problem_id_segs)
            if problem_id_segs
            else np.empty(0, dtype=object)
        )
        p_know_vals: npt.NDArray[np.float64] = (
            np.concatenate(p_know_segs)
            if p_know_segs
            else np.empty(0, dtype=np.float64)
        )
        correctness_vals: npt.NDArray[np.int8] = (
            np.concatenate(correctness_segs)
            if correctness_segs
            else np.empty(0, dtype=np.int8)
        )

        long_data_df: dict[str, Any] = {
            "student_id": pd.Categorical(student_ids),
            "problem_id": pd.Categorical(problem_ids),
            "pKnow": p_know_vals,
            "pCorrectness": (
                np.concatenate(p_correctness_segs)
                if p_correctness_segs
                else np.empty(0, dtype=np.float64)
            ),
        }
        long_data_df[correctness_col_name] = correctness_vals

        return pd.DataFrame(long_data_df)

    @abstractmethod
    def _extract_bkt_params_from_fit(
        self,
        fit: CmdStanFit,
        n_students: int,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        groups: Optional[npt.NDArray[np.int32]] = None,
        kc_data: Optional[KCData] = None,
        kc_id: Optional[str] = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Extract student-indexed BKT parameter arrays from fit artifacts."""
        raise NotImplementedError

    @staticmethod
    def _extract_named_param_matrix(
        fit: CmdStanFit, param_name: str
    ) -> npt.NDArray[np.float64]:
        """Extract a Stan variable's draws as a 2D ``(n_draws, n_values)`` array.

        Normalizes MLE/pathfinder point estimates (which have no draw dimension)
        to a single-row matrix.
        """
        stan_variable_fn = getattr(fit, "stan_variable", None)
        if callable(stan_variable_fn):
            values = np.asarray(stan_variable_fn(param_name), dtype=np.float64)
        else:
            values = np.asarray(
                BKTModelBase._extract_param_draws(fit, param_name), dtype=np.float64
            )
        if values.ndim == 0:
            return values.reshape(1, 1)
        if values.ndim == 1:
            return values.reshape(1, -1)
        return values

    @staticmethod
    def _extract_scalar_param_draws(
        fit: CmdStanFit, param_name: str
    ) -> npt.NDArray[np.float64]:
        """Extract a scalar Stan parameter's draws as a 1D ``(n_draws,)`` array."""
        return BKTModelBase._extract_named_param_matrix(fit, param_name).reshape(-1)

    def _train_student_positions(
        self, kc_id: str, student_ids: list[str]
    ) -> npt.NDArray[np.int64]:
        """0-based fit-time position of each student by ID, -1 if not in the fit."""
        fit_save_entry = self._get_fit_save_entry(kc_id)
        student2index = (
            fit_save_entry.student2index if fit_save_entry is not None else None
        )
        if not student2index:
            raise ValueError(
                f"Fit for KC '{kc_id}' has no stored student index. Refit the model "
                "to predict with individualized initial knowledge."
            )
        return np.array(
            [student2index.get(str(sid), 0) - 1 for sid in student_ids],
            dtype=np.int64,
        )

    def _individual_pi_know_draw_matrix(
        self, fit: CmdStanFit, kc_data: KCData, kc_id: str
    ) -> npt.NDArray[np.float64]:
        """Per-draw pi_know for every predict-time student, shape ``(n_draws, n_students)``.

        Students in the fit reuse their fitted latent value, matched by ID. Under JOINT
        the rest get the regression mean. Under CORRECTNESS_ONLY unseen students are
        dropped earlier by ``_drop_unseen_students``.
        """
        positions = self._train_student_positions(kc_id, kc_data.student_ids)
        seen = positions >= 0
        n_students = len(kc_data.student_ids)

        if self.init_knowledge_strategy == InitKnowledgeStrategy.JOINT:
            b0_draws = self._extract_scalar_param_draws(fit, "pi_b0_know_param")
            sigma_draws = self._extract_scalar_param_draws(fit, "pi_sigma_param")
            z_draws = self._extract_named_param_matrix(fit, "logit_pi_know_z")
            n_draws = b0_draws.shape[0]
            logits = np.repeat(b0_draws[:, None], n_students, axis=1)
            if kc_data.covariates is not None and kc_data.covariates.shape[1] > 0:
                b1_draws = self._extract_named_param_matrix(fit, "pi_b1_know_param")
                logits = logits + b1_draws @ kc_data.covariates.T
            logits[:, seen] += sigma_draws[:, None] * z_draws[:, positions[seen]]
        else:
            logits = self._extract_named_param_matrix(fit, "logit_pi_know_group")[
                :, positions
            ]
        return 1.0 / (1.0 + np.exp(-logits))

    def _extract_individual_pi_know_point_estimate(
        self,
        fit: CmdStanFit,
        kc_data: KCData,
        kc_id: str,
        point_estimate: Literal["mean", "median", "mode"],
    ) -> npt.NDArray[np.float64]:
        """Per-student pi_know point estimates under individualized initial knowledge."""
        draws = self._individual_pi_know_draw_matrix(fit, kc_data, kc_id)
        if point_estimate == "mean":
            return draws.mean(axis=0)
        if point_estimate == "median":
            return np.median(draws, axis=0)
        return np.array([self._modal_estimate(col) for col in draws.T], dtype=np.float64)

    @staticmethod
    def _modal_estimate(draws: npt.NDArray[np.float64]) -> float:
        """Compute a simple modal estimate from the draws using histogram binning.
        This is simple and fast. May need to look into using a KDE approach for smoother estimates.
        """
        counts, edges = np.histogram(draws, bins="auto")
        idx = int(np.argmax(counts))
        return float(0.5 * (edges[idx] + edges[idx + 1]))

    @staticmethod
    def _extract_param_point_estimate(
        fit: CmdStanFit,
        param_name: str,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
    ) -> float:
        draws = BKTModelBase._extract_param_draws(fit, param_name)
        if point_estimate == "mean":
            return float(np.mean(draws))
        if point_estimate == "median":
            return float(np.median(draws))
        return BKTModelBase._modal_estimate(draws)

    # TODO fix this monstrous function
    @staticmethod
    def _extract_param_draws(
        fit: CmdStanFit, param_name: str
    ) -> npt.NDArray[np.float64]:
        def _to_1d(values: Any) -> npt.NDArray[np.float64]:
            array: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64)
            if array.size == 0:
                raise ValueError(f"No values found for parameter '{param_name}'.")
            return array.ravel()

        stan_variable_fn = getattr(fit, "stan_variable", None)
        if callable(stan_variable_fn):
            try:
                return _to_1d(stan_variable_fn(param_name))
            except Exception:
                pass

        draws_pd_fn = getattr(fit, "draws_pd", None)
        if callable(draws_pd_fn):
            try:
                draws_pd = draws_pd_fn()
                series = BKTModelBase._find_param_series(draws_pd, param_name)
                if series is not None:
                    return _to_1d(series.to_numpy())
            except Exception:
                pass

        if hasattr(fit, "variational_sample"):
            sample = getattr(fit, "variational_sample")
            if isinstance(sample, pd.DataFrame):
                series = BKTModelBase._find_param_series(sample, param_name)
                if series is not None:
                    return _to_1d(series.to_numpy())

        optimized_df = getattr(fit, "optimized_params_pd", None)
        if isinstance(optimized_df, pd.DataFrame):
            try:
                series = BKTModelBase._find_param_series(optimized_df, param_name)
                if series is not None:
                    return _to_1d(series.to_numpy())
            except Exception:
                pass

        optimized_dict = getattr(fit, "optimized_params_dict", None)
        if isinstance(optimized_dict, dict):
            try:
                for key in (param_name, f"{param_name}[1]", f"{param_name}.1"):
                    if key in optimized_dict:
                        return _to_1d([optimized_dict[key]])
            except Exception:
                pass

        raise ValueError(
            f"Could not extract parameter '{param_name}' from fit type '{type(fit).__name__}'."
        )

    @staticmethod
    def _find_param_series(df: pd.DataFrame, param_name: str) -> Optional[pd.Series]:
        exact_candidates = [param_name, f"{param_name}[1]", f"{param_name}.1"]
        for col in exact_candidates:
            if col in df.columns:
                return df[col]

        col_strings = pd.Index(df.columns.astype(str))
        bracket_mask = col_strings.str.fullmatch(rf"{param_name}\[\s*1\s*\]")
        dot_mask = col_strings.str.fullmatch(rf"{param_name}\.1")
        bare_mask = col_strings.str.fullmatch(rf"{param_name}")
        matched = col_strings[bare_mask | bracket_mask | dot_mask]
        if len(matched) > 0:
            return df[matched[0]]

        prefix_matches = col_strings[col_strings.str.startswith(f"{param_name}[")]
        if len(prefix_matches) > 0:
            return df[prefix_matches[0]]

        return None

    def predict_posterior_stan(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> dict[str, csp.CmdStanGQ]:
        """Run Stan generated quantities for posterior state prediction.

        Parameters
        ----------
        data : pd.DataFrame
            Student interaction data. Must contain student ID, problem ID, and
            correctness columns for the KCs of interest.
        column_mapping : dict, optional
            Column name mapping.  Defaults to the standard ``ColumnNames`` defaults.

        Returns
        -------
        dict[str, CmdStanGQ]
            Mapping from KC ID to raw CmdStanGQ fit objects.  Pass the result to
            ``predict_posterior_draws`` to obtain draw-level DataFrames.
        """
        return predict_posterior(
            model=self,
            data=data,
            column_mapping=column_mapping,
            smoothed=False,
            output="stan",
            student_covariates=student_covariates,
        )

    def predict_posterior_draws(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
        backend: Literal["stan", "numba"] = "stan",
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """Return draw-level posterior prediction DataFrames.

        Parameters
        ----------
        data : pd.DataFrame
            Student interaction data used to remap Stan indices to original IDs.
        column_mapping : dict, optional
            Column name mapping.  Defaults to the standard ``ColumnNames`` defaults.
        stan_output : dict[str, CmdStanGQ], optional
            Pre-computed output from ``predict_posterior_stan``.  When provided
            the Stan generated-quantities step is skipped.
        backend : {"stan", "numba"}, default="stan"
            Backend used to produce posterior draws.
            - ``stan``: use Stan generated quantities output (current behavior).
            - ``numba``: run deterministic hidden-state recursion for each posterior parameter draw.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from KC ID to draw-level DataFrames.  Pass to
            ``stanbkt.utils.posterior_summary`` to obtain summary statistics.
        """
        return predict_posterior(
            model=self,
            data=data,
            column_mapping=column_mapping,
            smoothed=False,
            backend=backend,
            output="draws",
            stan_output=stan_output,
            student_covariates=student_covariates,
        )

    def predict_smoothed_posterior_stan(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> dict[str, csp.CmdStanGQ]:
        """Run Stan generated quantities for smoothed posterior state prediction.

        Parameters
        ----------
        data : pd.DataFrame
            Student interaction data.
        column_mapping : dict, optional
            Column name mapping.  Defaults to the standard ``ColumnNames`` defaults.

        Returns
        -------
        dict[str, CmdStanGQ]
            Mapping from KC ID to raw CmdStanGQ fit objects.  Pass the result to
            ``predict_smoothed_posterior_draws`` to obtain draw-level DataFrames.
        """
        return predict_posterior(
            model=self,
            data=data,
            column_mapping=column_mapping,
            smoothed=True,
            output="stan",
            student_covariates=student_covariates,
        )

    def predict_smoothed_posterior_draws(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
        backend: Literal["stan", "numba"] = "stan",
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> dict[str, pd.DataFrame]:
        """Return draw-level smoothed posterior prediction DataFrames.

        Parameters
        ----------
        data : pd.DataFrame
            Student interaction data used to remap Stan indices to original IDs.
        column_mapping : dict, optional
            Column name mapping.  Defaults to the standard ``ColumnNames`` defaults.
        stan_output : dict[str, CmdStanGQ], optional
            Pre-computed output from ``predict_smoothed_posterior_stan``.  When
            provided the Stan generated-quantities step is skipped.
        backend : {"stan", "numba"}, default="stan"
            Backend used to produce posterior draws.
            - ``stan``: use Stan generated quantities output (current behavior).
            - ``numba``: run deterministic hidden-state recursion for each posterior parameter draw.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from KC ID to draw-level DataFrames.  Pass to
            ``stanbkt.utils.posterior_summary`` to obtain summary statistics.
        """
        return predict_posterior(
            model=self,
            data=data,
            column_mapping=column_mapping,
            smoothed=True,
            backend=backend,
            output="draws",
            stan_output=stan_output,
            student_covariates=student_covariates,
        )

    def predict_posterior_summary(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        quantiles: list[float] = [0.025, 0.975],
        stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
        n_cores: int = 1,
        backend: Literal["stan", "numba"] = "stan",
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return per-observation posterior summaries without materializing all draws.

        Parameters
        ----------
        data : pd.DataFrame
            Student interaction data used to remap Stan indices to original IDs.
        column_mapping : dict, optional
            Column name mapping. Defaults to the standard ``ColumnNames`` defaults.
        quantiles : list[float], default=[0.025, 0.975]
            Quantiles to include in the returned posterior summary.
        stan_output : dict[str, CmdStanGQ], optional
            Pre-computed output from ``predict_posterior_stan``. When provided,
            the Stan generated-quantities step is skipped.
        n_cores : int, default=1
            Number of concurrent KC jobs to run. Use ``-1`` to use all available CPU
            cores.
        backend : {"stan", "numba"}, default="stan"
            Backend used to produce posterior summaries.
            - ``stan``: summarize Stan generated quantities output.
            - ``numba``: generate draw-level predictions via numba and summarize them.

        Returns
        -------
        pd.DataFrame
            Per-observation posterior summary statistics for the overlapping fitted KCs.

        Warning
        -------
        Setting ``n_cores`` greater than 1, or ``-1``, can substantially
        increase peak memory usage and will most likely cause out-of-memory failures
        when the dataset is large.
        """
        return predict_posterior(
            model=self,
            data=data,
            column_mapping=column_mapping,
            smoothed=False,
            backend=backend,
            output="summary",
            quantiles=quantiles,
            stan_output=stan_output,
            n_cores=n_cores,
            student_covariates=student_covariates,
        )

    def predict_smoothed_posterior_summary(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        quantiles: list[float] = [0.025, 0.975],
        stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
        n_cores: int = 1,
        backend: Literal["stan", "numba"] = "stan",
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return per-observation smoothed posterior summaries without materializing all draws.

        Parameters
        ----------
        data : pd.DataFrame
            Student interaction data used to remap Stan indices to original IDs.
        column_mapping : dict, optional
            Column name mapping. Defaults to the standard ``ColumnNames`` defaults.
        quantiles : list[float], default=[0.025, 0.975]
            Quantiles to include in the returned posterior summary.
        stan_output : dict[str, CmdStanGQ], optional
            Pre-computed output from ``predict_smoothed_posterior_stan``. When
            provided, the Stan generated-quantities step is skipped.
        n_cores : int, default=1
            Number of concurrent KC jobs to run. Use ``-1`` to use all available CPU
            cores.
        backend : {"stan", "numba"}, default="stan"
            Backend used to produce posterior summaries.
            - ``stan``: summarize Stan generated quantities output.
            - ``numba``: generate draw-level predictions via numba and summarize them.

        Returns
        -------
        pd.DataFrame
            Per-observation smoothed posterior summary statistics for the overlapping
            fitted KCs.

        Warning
        -------
        Setting ``n_cores`` greater than 1, or ``-1``, can substantially
        increase peak memory usage and will most likely cause out-of-memory failures
        when the dataset is large.
        """
        return predict_posterior(
            model=self,
            data=data,
            column_mapping=column_mapping,
            smoothed=True,
            backend=backend,
            output="summary",
            quantiles=quantiles,
            stan_output=stan_output,
            n_cores=n_cores,
            student_covariates=student_covariates,
        )

    def _augment_stan_data_for_predict(
        self,
        kc_id_str: str,
        kc_data: KCData,
        kc_fit_result: CmdStanFit,
        data_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind the fit-time population and map predict-time students to it by ID.

        The saved per-student parameters are sized by the fit-time population, which
        can differ from the predict-time ``nStudents``.
        """
        if not self.individual_initial_knowledge:
            return data_dict
        param_name = (
            "logit_pi_know_z"
            if self.init_knowledge_strategy == InitKnowledgeStrategy.JOINT
            else "logit_pi_know_group"
        )
        n_train = self._extract_named_param_matrix(kc_fit_result, param_name).shape[1]
        positions = self._train_student_positions(kc_id_str, kc_data.student_ids)
        return {
            **data_dict,
            "nTrainStudents": int(n_train),
            "train_student_idx": (positions + 1).astype(np.int32),
        }

    def _generate_single_kc_quantities(
        self,
        kc_id_str: str,
        kc_data: KCData,
        kc_fit_result: CmdStanFit,
        kc_priors: PriorsBase,
        gq_model: csp.CmdStanModel,
    ) -> tuple[str, csp.CmdStanGQ, KCData]:
        data_dict = self._build_stan_data_dict(kc_data, kc_priors)
        data_dict = self._augment_stan_data_for_predict(
            kc_id_str, kc_data, kc_fit_result, data_dict
        )
        gq_fit = gq_model.generate_quantities(
            data=data_dict,
            previous_fit=kc_fit_result,  # type: ignore[type-var]
        )
        return kc_id_str, gq_fit, kc_data

    def _predict_generated_quantities(
        self,
        data: pd.DataFrame,
        gq_model: csp.CmdStanModel,
        priors: Optional[dict[str, PriorsBase] | PriorsBase] = None,
        column_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
        _per_kc_callback: Optional[Callable[[str, Any, KCData], None]] = None,
        n_cores: int = 1,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> dict[str, csp.CmdStanGQ]:
        # validate priors
        if priors is None:
            # use default priors for all KCs
            self.log(
                "No priors provided, using default priors for all KCs.",
                level=VerbosityLevel.DEBUG,
            )
            priors = self._default_priors()
        else:
            self._default_priors_class()._validate(
                priors, type(self), self.init_knowledge_strategy
            )

        n_cores = self._resolve_n_cores(n_cores)
        resolved_mapping = ColumnNames.apply_default_mapping(column_mapping)
        prepared_covariates, covariate_columns = self._prepare_joint_covariates(
            data, student_covariates, resolved_mapping
        )

        gq_kc_fit: dict[str, csp.CmdStanGQ] = {}

        def _handle_single_kc_result(
            result: tuple[str, csp.CmdStanGQ, KCData],
        ) -> None:
            kc_id_str, gq_fit, kc_data = result
            if _per_kc_callback is not None:
                _per_kc_callback(kc_id_str, gq_fit, kc_data)
            else:
                gq_kc_fit[kc_id_str] = gq_fit

        if n_cores == 1:
            for kc_id, kc_data in iter_kc_data(
                data=data,
                col_mapping=resolved_mapping,
                return_groups=self._use_groups,
                print_fn=self.log,
                student_covariates=prepared_covariates,
                covariate_columns=covariate_columns,
            ):
                kc_id_str = str(kc_id)
                kc_data = self._align_kc_group_indices_with_fit_metadata(
                    kc_id_str, kc_data
                )
                if kc_data.covariates is not None:
                    self._check_covariate_columns_match(
                        kc_id_str, kc_data.covariate_columns
                    )

                kc_fit_result = self.fits.get_fit(kc_id_str)
                if kc_fit_result is None:
                    continue

                # the `priors.get` is valid but `ty` is not currently smart enough, hence the ignore
                kc_priors = (
                    priors.get(
                        str(kc_id), self._default_priors()
                    )  # ty:ignore[no-matching-overload]
                    if isinstance(priors, dict)
                    else priors
                )

                _handle_single_kc_result(
                    self._generate_single_kc_quantities(
                        kc_id_str=kc_id_str,
                        kc_data=kc_data,
                        kc_fit_result=kc_fit_result,
                        kc_priors=kc_priors,
                        gq_model=gq_model,
                    )
                )

            return gq_kc_fit

        pending: set[Future[tuple[str, csp.CmdStanGQ, KCData]]] = set()
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            for kc_id, kc_data in iter_kc_data(
                data=data,
                col_mapping=resolved_mapping,
                return_groups=self._use_groups,
                print_fn=self.log,
                student_covariates=prepared_covariates,
                covariate_columns=covariate_columns,
            ):
                kc_id_str = str(kc_id)
                kc_data = self._align_kc_group_indices_with_fit_metadata(
                    kc_id_str, kc_data
                )
                if kc_data.covariates is not None:
                    self._check_covariate_columns_match(
                        kc_id_str, kc_data.covariate_columns
                    )

                kc_fit_result = self.fits.get_fit(kc_id_str)
                if kc_fit_result is None:
                    continue

                kc_priors = (
                    priors.get(
                        str(kc_id), self._default_priors()
                    )  # ty:ignore[no-matching-overload]
                    if isinstance(priors, dict)
                    else priors
                )

                pending.add(
                    executor.submit(
                        self._generate_single_kc_quantities,
                        kc_id_str,
                        kc_data,
                        kc_fit_result,
                        kc_priors,
                        gq_model,
                    )
                )

                if len(pending) >= n_cores:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        _handle_single_kc_result(future.result())

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    _handle_single_kc_result(future.result())

        return gq_kc_fit

    def _process_predict_gq(
        self,
        posterior_draws_raw: dict[str, csp.CmdStanGQ],
        data: pd.DataFrame,
        col_mapping: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        """Post-process raw CmdStanGQ outputs into long-form DataFrames with remapped IDs."""
        return gq_to_draws(
            stan_output=posterior_draws_raw,
            data=data,
            col_mapping=col_mapping,
            print_fn=self.log,
        )

    @staticmethod
    def _empty_posterior_summary_frame(
        col_mapping: dict[str, str], quantiles: list[float]
    ) -> pd.DataFrame:
        stat_labels = ["mean", "std", "median"] + [f"{q * 100:.2f}%" for q in quantiles]
        summary_cols = [f"{_PKNOW}_{stat}" for stat in stat_labels]
        summary_cols += [f"{_PCORRECT}_{stat}" for stat in stat_labels]
        return pd.DataFrame(
            columns=[
                "kc_id",
                col_mapping[ColumnNames.STUDENT_ID],
                col_mapping[ColumnNames.PROBLEM_ID],
                col_mapping[ColumnNames.CORRECTNESS],
            ]
            + summary_cols
        )

    @staticmethod
    def _data_kc_ids(
        data: pd.DataFrame, resolved_mapping: Mapping[str, str]
    ) -> list[str]:
        """KC IDs iter_kc_data would yield: rows without an order value are dropped."""
        kc_column = resolved_mapping[ColumnNames.KC_ID]
        order_column = resolved_mapping[ColumnNames.ORDER]
        if kc_column not in data.columns:
            return [_DEFAULT_KC_ID]
        if order_column in data.columns:
            return list(
                data.loc[data[order_column].notna(), kc_column].astype(str).unique()
            )
        return []

    @staticmethod
    def _available_cpus() -> int:
        # affinity aware, os.cpu_count ignores cpusets and taskset limits
        count_fn = getattr(os, "process_cpu_count", None)
        if count_fn is not None:
            count = count_fn()
        elif hasattr(os, "sched_getaffinity"):
            count = len(os.sched_getaffinity(0))
        else:
            count = os.cpu_count()
        return count or 1

    @staticmethod
    def _stan_cores_per_fit(fit_options: StanFitOptions) -> int:
        """Cores one KC fit already uses, from the fit options."""
        if isinstance(fit_options, MCMCFitOptions):
            chains = fit_options.chains
            parallel_chains = fit_options.parallel_chains or chains
            return max(1, min(chains, parallel_chains) * fit_options.threads_per_chain)
        if isinstance(fit_options, PFFitOptions):
            return max(1, fit_options.num_threads or 1)
        return 1

    def _resolve_n_kcs_workers(
        self, n_kcs_workers: int, stan_cores_per_fit: int, n_kcs: int
    ) -> int:
        if isinstance(n_kcs_workers, bool) or not isinstance(
            n_kcs_workers, (int, np.integer)
        ):
            raise TypeError("'n_kcs_workers' must be -1 (auto) or a positive integer.")
        if n_kcs_workers != -1 and n_kcs_workers < 1:
            raise ValueError("'n_kcs_workers' must be -1 (auto) or a positive integer.")

        available = self._available_cpus()
        n_kcs = max(1, n_kcs)

        if n_kcs_workers == -1:
            workers = min(n_kcs, max(1, available // stan_cores_per_fit))
            if n_kcs > 1 and stan_cores_per_fit >= available:
                message = (
                    f"n_kcs_workers=-1 resolved to 1: the Stan settings already use "
                    f"{stan_cores_per_fit} cores per fit and {available} CPUs are available, "
                    "so KCs are fitted one at a time."
                )
                if stan_cores_per_fit > available:
                    message += (
                        " The Stan settings alone oversubscribe the available CPUs."
                    )
                self.log(message, level=VerbosityLevel.WARN)
            else:
                self.log(
                    f"Fitting {n_kcs} KC(s) with {workers} concurrent worker(s) "
                    f"({stan_cores_per_fit} Stan cores per fit, {available} CPUs available).",
                    level=VerbosityLevel.INFO if workers > 1 else VerbosityLevel.DEBUG,
                )
            return workers

        workers = min(int(n_kcs_workers), n_kcs)
        total = workers * stan_cores_per_fit
        if workers > 1 and total > available:
            raise ValueError(
                f"n_kcs_workers={workers} x {stan_cores_per_fit} Stan cores per fit = "
                f"{total} cores, but only {available} CPUs are available. Reduce "
                "n_kcs_workers or the Stan settings (chains, parallel_chains, "
                "threads_per_chain), or use n_kcs_workers=-1 to size it automatically."
            )
        return workers

    @staticmethod
    def _resolve_n_cores(n_cores: int) -> int:
        if n_cores == -1:
            return os.cpu_count() or 1
        if n_cores < 1:
            raise ValueError("'n_cores' must be -1 or at least 1.")
        return n_cores

    def _process_predict_summary_gq(
        self,
        posterior_summary_raw: dict[str, csp.CmdStanGQ],
        data: pd.DataFrame,
        col_mapping: dict[str, str],
        quantiles: list[float],
        n_cores: int = 1,
    ) -> pd.DataFrame:
        """Post-process raw CmdStanGQ outputs into per-observation summary statistics."""
        n_cores = self._resolve_n_cores(n_cores)

        def _summarize_kc(item: tuple[str, KCData]) -> Optional[pd.DataFrame]:
            kc_id, kc_data = item
            kc_id_str = str(kc_id)
            gq_kc = posterior_summary_raw.get(kc_id_str)
            if gq_kc is None:
                self.log(
                    f"No generated quantities found for KC '{kc_id_str}' when summarizing predictions.",
                    level=VerbosityLevel.DEBUG,
                )
                return None

            kc_summary = _summarize_single_kc_gq(
                kc_id_str, gq_kc, kc_data, col_mapping, quantiles
            )
            if kc_summary.empty:
                return None
            return kc_summary

        if n_cores == 1:
            kc_items = iter_kc_data(
                data=data,
                col_mapping=col_mapping,
                return_groups=False,
                print_fn=self.log,
            )
            result_frames = [
                kc_summary
                for kc_summary in (_summarize_kc(item) for item in kc_items)
                if kc_summary is not None
            ]
        else:
            result_frames: list[pd.DataFrame] = []
            pending: set[Future[Optional[pd.DataFrame]]] = set()
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                for kc_item in iter_kc_data(
                    data=data,
                    col_mapping=col_mapping,
                    return_groups=False,
                    print_fn=self.log,
                ):
                    pending.add(executor.submit(_summarize_kc, kc_item))

                    if len(pending) >= n_cores:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            kc_summary = future.result()
                            if kc_summary is not None:
                                result_frames.append(kc_summary)

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        kc_summary = future.result()
                        if kc_summary is not None:
                            result_frames.append(kc_summary)

        if not result_frames:
            return self._empty_posterior_summary_frame(col_mapping, quantiles)

        return pd.concat(result_frames, ignore_index=True)

    def _predict_summary_streaming(
        self,
        data: pd.DataFrame,
        gq_model: csp.CmdStanModel,
        column_mapping: dict[str, str],
        quantiles: list[float],
        n_cores: int = 1,
        student_covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run GQ and summarize outputs one KC at a time to minimize peak memory."""
        result_frames: list[pd.DataFrame] = []

        def _consume(kc_id_str: str, gq_fit: Any, kc_data: KCData) -> None:
            kc_summary = _summarize_single_kc_gq(
                kc_id_str, gq_fit, kc_data, column_mapping, quantiles
            )
            if not kc_summary.empty:
                result_frames.append(kc_summary)

        self._predict_generated_quantities(
            data=data,
            gq_model=gq_model,
            column_mapping=column_mapping,
            _per_kc_callback=_consume,
            n_cores=n_cores,
            student_covariates=student_covariates,
        )

        if not result_frames:
            return self._empty_posterior_summary_frame(column_mapping, quantiles)

        return pd.concat(result_frames, ignore_index=True)

    @property
    @abstractmethod
    def _stan_model_filename(self) -> str:
        """Return stan file name inside stanbkt.stan_code."""
        pass

    @property
    @abstractmethod
    def _stan_hidden_filename(self) -> str:
        pass

    @property
    @abstractmethod
    def _stan_smoothed_hidden_filename(self) -> str:
        pass

    @abstractmethod
    def _build_stan_data_dict(
        self, kc_data: KCData, priors: Optional[PriorsBase] = None
    ) -> dict[str, Any]:
        """Build Stan data dictionary for a single KC interaction bundle.

        Parameters
        ----------
        kc_data : KCData
            Preprocessed KC-specific interaction data.
        priors : PriorsBase, optional
            Priors object containing the prior specifications for the model parameters for this KC.
            If None, the priors will not be added to the return dict.


        Returns
        -------
        dict[str, Any]
            Stan-compatible data.
        """
        raise NotImplementedError(
            "Subclasses must implement _build_stan_data_dict to support model estimation and posterior predictions."
        )

    def _compile_model(self, stan_file: str | os.PathLike[str]) -> None:
        """Compile the Stan model and cache it."""

        self._stan_model = compile_stan_model(
            stan_file,
            stanc_options=self.stan_compile_kwargs,
            cpp_options=self.cpp_compile_kwargs,
            print_fn=self.log,
        )
