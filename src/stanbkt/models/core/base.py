"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.

"""

from __future__ import annotations
from natsort import natsort_keygen
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
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_types import CmdStanFit
from stanbkt.fits.fit_types import FitMethod
from stanbkt.fits.core.base import FitBase as BaseFit
from stanbkt.models.model_types import (
    ModelType,
    InitKnowledgeStrategy,
)
from stanbkt.models.error import FitMethodMismatchError
from stanbkt.models.priors import PriorsBase
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import (
    iter_kc_data,
    ColumnNames,
    KCData,
    _DEFAULT_KC_ID,
    _NA_FILL_VALUE,
    _PKNOW,
    _PCORRECT,
)
from stanbkt.utils.model_archive import pack_model_directory
from stanbkt.utils.posterior_utils import gq_to_draws


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
    fits : Optional[BaseFit]
        Object to store fitted model results for each KC.
    """

    def __init__(
        self,
        fit_method: FitMethod | str = FitMethod.MCMC,
        individual_initial_knowledge: bool = False,
        init_knowledge_strategy: InitKnowledgeStrategy = InitKnowledgeStrategy.CORRECTNESS_ONLY,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: Optional[Dict[str, Any]] = None,
        cpp_compile_kwargs: Optional[Dict[str, Any]] = None,
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
        self.fits: Optional[BaseFit] = None
        self._is_fitted: bool = False

    def __str__(self) -> str:
        """Return a user-friendly string representation of the model."""
        class_name = self.__class__.__name__
        fit_status = "fitted" if self._is_fitted else "not fitted"
        num_kcs = self.fits.num_fitted_kcs if self._is_fitted and self.fits else 0

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


        Returns
        -------
        BKTModelBase
            The fitted BKT model instance.

        Raises
        ------
        ValueError
            If data validation fails or invalid method specified.
        """

        # Intialize new fits object if not already initialized.
        if self.fits is None:
            self.fits = self.fit_class()

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
                stan_fit_options, self._fit_method
            )

        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=self._use_groups,
            print_fn=self.log,
        ):
            if self.fits.has_kc(str(kc_id)):
                if not overwrite_kcs:
                    raise ValueError(
                        f"Fit for KC '{kc_id}' already exists. Set 'overwrite=True' to overwrite."
                    )

            self.log(f"Fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)

            # the `priors.get` is valid but `ty` is not currently smart enough, hence the ignore
            kc_priors = (
                priors.get(
                    str(kc_id), self._default_priors()
                )  # ty:ignore[no-matching-overload]
                if isinstance(priors, dict)
                else priors
            )
            data_dict = self._build_stan_data_dict(kc_data, kc_priors)
            fit_result = self._fit_stan_model_using_method(
                data_dict=data_dict, fit_options=stan_fit_options
            )
            self.fits.add_fit(str(kc_id), fit_result, overwrite_kcs=overwrite_kcs)
            self.log(f"Finished fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)
            self._is_fitted = True
        return self

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
        percentiles: Tuple[float, float] = (2.5, 97.5),
        column_mapping: dict[str, str] = {},
        clear_cache: bool = False,
    ) -> Any:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        params : list of str, optional
            List of parameters to summarize. If None, summarizes all.
        percentiles : tuple of float, default=(2.5, 97.5)
            Percentiles to include in summary. Values should be in range [1, 99].
        refresh_cache : bool, default=False
            Whether to refresh the cached summary.

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
        if self.fits is None:
            raise RuntimeError("Model must be fitted before calling summary()")
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

        return self.fits._summary(
            percentiles=percentiles,
            kc_col_name=kc_col_name,
        )

    def _fit_check(self, referrer: Optional[str] = None) -> None:
        """Check if model has been fitted."""
        if not self._is_fitted or self.fits is None or self.fits.num_fitted_kcs == 0:
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
        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling save()."
            )
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
        fitted_kcs: set[str] = set(self.fits.stan_fits.keys()) if self.fits else set()
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
        fitted_kcs: set[str] = set(self.fits.stan_fits.keys()) if self.fits else set()
        return kcs.intersection(fitted_kcs)

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
    ) -> pd.DataFrame:
        """Predict hidden states using point-estimate parameters from fitted posteriors."""
        self._fit_check(referrer="predict")

        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError("'point_estimate' must be 'mean', 'median', or 'mode'.")

        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]

        working_data = data
        if kc_column_name not in working_data.columns:
            working_data = working_data.copy()
            working_data[kc_column_name] = _DEFAULT_KC_ID

        self.check_data_contains_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling prediction methods."
            )

        if filtered_data.empty:
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

        # compile and cache the numba function
        njit_predict_numba: Callable = njit(
            fastmath=fast_math, parallel=parallel, cache=True
        )(type(self)._predict_hidden_states_numba)

        # TODO: need to adjust this based on the model (grouped vs not)
        predictions: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=filtered_data,
            col_mapping=column_mapping,
            return_groups=self._use_groups,
            print_fn=self.log,
        ):
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
                groups=kc_data.groups if self._use_groups else None,
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
                kc_data=kc_data,
                p_correctness=p_correctness,
                correctness_col_name=column_mapping.get(
                    ColumnNames.CORRECTNESS, ColumnNames.CORRECTNESS
                ),
            )
            kc_predictions.insert(0, ColumnNames.KC_ID, str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
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

        return pd.concat(predictions, ignore_index=True)

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
    ) -> pd.DataFrame:
        """Predict smoothed hidden states using point-estimate parameters."""
        self._fit_check(referrer="predict_smoothed")

        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError("'point_estimate' must be 'mean', 'median', or 'mode'.")

        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]

        working_data = data
        if kc_column_name not in working_data.columns:
            working_data = working_data.copy()
            working_data[kc_column_name] = _DEFAULT_KC_ID

        self.check_data_contains_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling prediction methods."
            )

        if filtered_data.empty:
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

        # compile and cache the numba function
        njit_predict_smoothed_numba: Callable = njit(
            fastmath=fast_math, parallel=parallel, cache=True
        )(type(self)._predict_hidden_states_smoothed_numba)

        predictions: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=filtered_data,
            col_mapping=column_mapping,
            return_groups=self._use_groups,
            print_fn=self.log,
        ):
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
                groups=kc_data.groups if self._use_groups else None,
            )
            p_smooth, p_correctness = njit_predict_smoothed_numba(
                correctness=kc_data.correctness,
                prior=prior,
                learn=learn,
                forget=forget,
                guess=guess,
                slip=slip,
                lengths=kc_data.lengths,
            )
            kc_predictions = self._state_arrays_to_long_df(
                p_know=p_smooth,
                kc_data=kc_data,
                p_correctness=p_correctness,
                correctness_col_name=column_mapping.get(
                    ColumnNames.CORRECTNESS, ColumnNames.CORRECTNESS
                ),
            )

            kc_predictions.insert(0, ColumnNames.KC_ID, str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
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

        return pd.concat(predictions, ignore_index=True)

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
        kc_data: KCData,
        p_correctness: Optional[np.ndarray] = None,
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
            if p_correctness is not None:
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
            "student_id": student_ids,
            "problem_id": problem_ids,
            "pKnow": p_know_vals,
        }
        if p_correctness is not None:
            long_data_df["pCorrectness"] = (
                np.concatenate(p_correctness_segs)
                if p_correctness_segs
                else np.empty(0, dtype=np.float64)
            )
        long_data_df[correctness_col_name] = correctness_vals

        return pd.DataFrame(long_data_df)

    @staticmethod
    def _build_prediction_index_frame(kc_data: KCData, n_problems: int) -> pd.DataFrame:
        """Build a lookup frame mapping Stan 1-based (student_idx, problem_idx) to original IDs.

        Used by the posterior prediction path to remap numeric Stan indices back to the
        original student/problem ID strings.
        """
        n_students = len(kc_data.student_inter_dict)
        lengths = np.clip(kc_data.lengths, 0, n_problems)

        flat_student_idx = np.repeat(
            np.arange(1, n_students + 1, dtype=np.int64), n_problems
        )
        flat_problem_idx = np.tile(
            np.arange(1, n_problems + 1, dtype=np.int64), n_students
        )

        student_positions = np.repeat(np.arange(n_students, dtype=np.int64), n_problems)
        problem_positions = flat_problem_idx - 1  # 0-based
        flat_is_valid = problem_positions < lengths[student_positions]

        flat_student_ids = np.repeat(
            np.fromiter(
                (str(sid) for sid in kc_data.student_inter_dict.keys()),
                dtype=object,
                count=n_students,
            ),
            n_problems,
        )

        flat_problem_ids = np.full(n_students * n_problems, "-1", dtype=object)
        for student_index, interaction in enumerate(
            kc_data.student_inter_dict.values()
        ):
            row_start = student_index * n_problems
            for problem_index, pid in enumerate(interaction.problem_ids):
                flat_problem_ids[row_start + problem_index] = str(pid)

        flat_correctness = kc_data.correctness.ravel(order="C").astype(
            np.float64, copy=True
        )
        flat_correctness[flat_correctness < 0] = -1.0

        return pd.DataFrame(
            {
                "student_idx": flat_student_idx,
                "problem_idx": flat_problem_idx,
                "student_id": flat_student_ids,
                "problem_id": flat_problem_ids,
                "is_valid": flat_is_valid,
                "correctness": flat_correctness,
            }
        )

    @abstractmethod
    def _extract_bkt_params_from_fit(
        self,
        fit: CmdStanFit,
        n_students: int,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        groups: Optional[npt.NDArray[np.int32]] = None,
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

    # TODO: evaluate takes list of metrics and returns dataframe with cols: kc, metric_name
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the fitted model.

        Parameters
        ----------
        **kwargs
            Additional evaluation parameters.

        Returns
        -------
        dict
            Dictionary of evaluation metrics.

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        pass

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
        self._fit_check(referrer="predict_posterior_stan")
        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]
        self.check_data_contains_fitted_kcs(
            set(data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(set(data[kc_column_name].unique()))
        data_cp = data.copy().loc[data[kc_column_name].isin(overlapping_kcs)]

        if self._hidden_states_model is None:
            self._hidden_states_model = compile_stan_model(
                self._stan_hidden_filename,
                cpp_options=self.cpp_compile_kwargs,
                stanc_options=self.stan_compile_kwargs,
                print_fn=self.log,
            )

        return self._predict_generated_quantities(
            data=data_cp,
            gq_model=self._hidden_states_model,
            column_mapping=column_mapping,
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

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from KC ID to draw-level DataFrames.  Pass to
            ``stanbkt.utils.posterior_summary`` to obtain summary statistics.
        """
        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]
        self.check_data_contains_fitted_kcs(
            set(data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(set(data[kc_column_name].unique()))
        data_cp = data.copy().loc[data[kc_column_name].isin(overlapping_kcs)]

        if stan_output is None:
            if self._hidden_states_model is None:
                self._hidden_states_model = compile_stan_model(
                    self._stan_hidden_filename,
                    cpp_options=self.cpp_compile_kwargs,
                    stanc_options=self.stan_compile_kwargs,
                    print_fn=self.log,
                )
            stan_output = self._predict_generated_quantities(
                data=data_cp,
                gq_model=self._hidden_states_model,
                column_mapping=column_mapping,
            )

        return self._process_predict_gq(stan_output, data_cp, column_mapping)

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
        self._fit_check(referrer="predict_smoothed_posterior_stan")
        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]
        self.check_data_contains_fitted_kcs(
            set(data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(set(data[kc_column_name].unique()))
        data_cp = data.copy().loc[data[kc_column_name].isin(overlapping_kcs)]

        if self._smoothed_hidden_states_model is None:
            self._smoothed_hidden_states_model = compile_stan_model(
                self._stan_smoothed_hidden_filename,
                cpp_options=self.cpp_compile_kwargs,
                stanc_options=self.stan_compile_kwargs,
                print_fn=self.log,
            )

        return self._predict_generated_quantities(
            data=data_cp,
            gq_model=self._smoothed_hidden_states_model,
            column_mapping=column_mapping,
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

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from KC ID to draw-level DataFrames.  Pass to
            ``stanbkt.utils.posterior_summary`` to obtain summary statistics.
        """
        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]
        self.check_data_contains_fitted_kcs(
            set(data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(set(data[kc_column_name].unique()))
        data_cp = data.copy().loc[data[kc_column_name].isin(overlapping_kcs)]

        if stan_output is None:
            if self._smoothed_hidden_states_model is None:
                self._smoothed_hidden_states_model = compile_stan_model(
                    self._stan_smoothed_hidden_filename,
                    cpp_options=self.cpp_compile_kwargs,
                    stanc_options=self.stan_compile_kwargs,
                    print_fn=self.log,
                )
            stan_output = self._predict_generated_quantities(
                data=data_cp,
                gq_model=self._smoothed_hidden_states_model,
                column_mapping=column_mapping,
            )

        return self._process_predict_gq(stan_output, data_cp, column_mapping)

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
    ) -> dict[str, csp.CmdStanGQ]:

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling prediction methods."
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

        gq_kc_fit: dict[str, csp.CmdStanGQ] = {}
        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=self._use_groups,
            print_fn=self.log,
        ):
            kc_id_str = str(kc_id)

            kc_fit_result = self.fits.get_fit(kc_id)
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

            data_dict = self._build_stan_data_dict(kc_data, kc_priors)
            # type ignore is needed since CmdStanGQ has incorrect type hints for .generate_quantities
            gq_fit = gq_model.generate_quantities(
                data=data_dict,
                previous_fit=kc_fit_result,  # type: ignore[type-var]
            )

            gq_kc_fit[kc_id_str] = gq_fit

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
