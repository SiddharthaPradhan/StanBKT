"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.

"""

from __future__ import annotations
import re
import stanbkt
from stanbkt.fits.fit_factory import FitFactory
import json

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union, Final, get_args, Callable
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
from stanbkt.fits.core.base import BaseFit
from stanbkt.models.model_types import (
    ModelType,
    PriorEstimationType,
    PosteriorPredictionOutput,
)
from stanbkt.models.error import FitMethodMismatchError
from stanbkt.models.priors import BayesianPriors
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import (
    iter_kc_data,
    dict_has_types,
    ColumnNames,
    KCData,
    _DEFAULT_KC_ID,
    _NA_FILL_VALUE,
)
from stanbkt.utils.model_archive import pack_model_directory


class BKTModelBase(VerboseMixin, ABC):
    """Abstract base class for Stan Bayesian Knowledge Tracing (BKT) models.

    This class defines the interface that all Stan BKT model implementations must follow.

    Attributes
    ----------
    fit_method : FitMethod
        The method used for fitting the model (e.g., MCMC, VB, MAP).
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
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: Optional[Dict[str, Any]] = None,
        cpp_compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
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

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
        stan_fit_options: Optional[Union[StanFitOptions, dict[str, Any]]] = None,
    ) -> BKTModelBase:
        """
        Fit the BKT model to data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the training data. Must include columns for:
            Student ID, Problem ID, and Correctness (0/1).
            If the KC column is absent, all interactions are assumed to belong to a single knowledge component.
        column_mapping : dict, optional
            Mapping of expected column names. Keys should be 'student_id', 'problem_id', 'correct', and 'kc_id'.
            If None, default column names are used.
        stan_fit_options : StanFitOptions or dict, optional
            Arguments to pass to the CmdStanPy fit method. It is recommended to use the typed
            :class:`StanFitOptions`, but a raw dictionary of option, value pairs can also be passed. See :externa
            For example:
            - MCMC parameters (e.g., iter_sampling, chains, seed)
            - VB parameters (e.g., iter, tol_rel_obj)


        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If data validation fails, invalid method specified, or invalid fit arguments provided.

        """
        pass

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

    # TODO: implement this, but override if needed.
    # This will really just call the summary method of the fit object.
    def summary(
        self,
        params: Optional[list[str]] = None,
        percentiles: Tuple[float, ...] = (5, 50, 95),
        refresh_cache: bool = False,
    ) -> Any:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        params : list of str, optional
            List of parameters to summarize. If None, summarizes all.
        percentiles : tuple of float, default=(5, 50, 95)
            Percentiles to include in summary.
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
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling summary()")

        # if hasattr(self.fit, "summary"):
        #     if params:
        #         return self.fit.summary(var_names=params, percentiles=percentiles)
        #     return self._fit.summary(percentiles=percentiles)

        raise NotImplementedError("Summary not available for this fit method")

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
        fitted_kcs: set[str] = set(self.fits.kc_fits.keys()) if self.fits else set()
        if not len(self.get_kcs_in_fitted_kcs(kcs)) > 0:
            raise ValueError(
                f"Data contains no KCs that were previously fitted. Given KCs: {kcs}, fitted KCs: {fitted_kcs}"
            )
        kcs_not_fitted = kcs - fitted_kcs
        if kcs_not_fitted:
            self._print(
                f"Data contains {len(kcs_not_fitted)} KCs that were not fitted.",
                level=VerbosityLevel.WARN,
            )
            self._print(
                f"{list(kcs_not_fitted)} do not have fits.",
                level=VerbosityLevel.INFO,
            )

    def get_kcs_in_fitted_kcs(self, kcs: set[str]) -> set[str]:
        """Return the set of KCs in the data that were fitted previously."""
        self._fit_check()
        fitted_kcs: set[str] = set(self.fits.kc_fits.keys()) if self.fits else set()
        return kcs.intersection(fitted_kcs)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[dict[str, str]] = None,
        point_estimate: Literal["mean", "median"] = "mean",
        parallel: bool = True,
        fast_math: bool = True,
    ) -> pd.DataFrame:
        """Predict hidden states using point-estimate parameters from fitted posteriors."""
        self._fit_check(referrer="predict")

        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median"):
            raise ValueError("'point_estimate' must be either 'mean' or 'median'.")

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
            set(working_data[kc_column_name].unique())
        )
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling prediction methods."
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
            return_groups=False,
            print_fn=self._print,
        ):
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
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
            )
            kc_predictions.insert(0, "kc_id", str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
            return pd.DataFrame(
                columns=["kc_id", "parameter", "student_id", "problem_id", "value"]
            )

        return pd.concat(predictions, ignore_index=True)

    def predict_smoothed_states(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[dict[str, str]] = None,
        point_estimate: Literal["mean", "median"] = "mean",
        parallel: bool = True,
        fast_math: bool = True,
    ) -> pd.DataFrame:
        """Predict smoothed hidden states using point-estimate parameters."""
        self._fit_check(referrer="predict_smoothed_states")

        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median"):
            raise ValueError("'point_estimate' must be either 'mean' or 'median'.")

        kc_column_name = ColumnNames.KC_ID
        if column_mapping and ColumnNames.KC_ID in column_mapping:
            kc_column_name = column_mapping[ColumnNames.KC_ID]

        working_data = data
        if kc_column_name not in working_data.columns:
            working_data = working_data.copy()
            working_data[kc_column_name] = _DEFAULT_KC_ID

        self.check_data_contains_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(
            set(working_data[kc_column_name].unique())
        )
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling prediction methods."
            )

        # compile and cache the numba function
        njit_predict_smoothed_numba: Callable = njit(
            fastmath=fast_math, parallel=parallel, cache=True
        )(type(self)._predict_hidden_states_smoothed_numba)

        predictions: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=filtered_data,
            col_mapping=column_mapping,
            return_groups=False,
            print_fn=self._print,
        ):
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
            )
            p_smooth = njit_predict_smoothed_numba(
                correctness=kc_data.correctness,
                prior=prior,
                learn=learn,
                forget=forget,
                guess=guess,
                slip=slip,
                lengths=kc_data.lengths,
            )
            kc_predictions = self._single_state_array_to_long_df(
                p_state=p_smooth,
                parameter_name="pKnow",
                kc_data=kc_data,
            )
            kc_predictions.insert(0, "kc_id", str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
            return pd.DataFrame(
                columns=["kc_id", "parameter", "student_id", "problem_id", "value"]
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
    ) -> np.ndarray:
        """Numba-accelerated forward-backward recursion for smoothed state probabilities."""
        n_students, n_problems = correctness.shape
        p_smooth = np.full((n_students, n_problems), _NA_FILL_VALUE, dtype=np.float64)

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

        return p_smooth

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
        correctness_segs: list[npt.NDArray[np.float64]] = []

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
                kc_data.correctness[student_idx, :length].astype(np.float64)
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
        correctness_vals: npt.NDArray[np.float64] = (
            np.concatenate(correctness_segs)
            if correctness_segs
            else np.empty(0, dtype=np.float64)
        )

        data: dict[str, Any] = {
            "student_id": student_ids,
            "problem_id": problem_ids,
            "pKnow": p_know_vals,
        }
        if p_correctness is not None:
            data["pCorrectness"] = (
                np.concatenate(p_correctness_segs)
                if p_correctness_segs
                else np.empty(0, dtype=np.float64)
            )
        data[correctness_col_name] = correctness_vals

        return pd.DataFrame(data)

    @staticmethod
    def _single_state_array_to_long_df(
        p_state: np.ndarray,
        parameter_name: str,
        kc_data: KCData,
    ) -> pd.DataFrame:
        """Convert a single state matrix to long-form prediction output using only valid entries."""
        student_id_segs: list[np.ndarray] = []
        problem_id_segs: list[np.ndarray] = []
        p_state_segs: list[np.ndarray] = []
        correctness_segs: list[np.ndarray] = []

        for student_idx, (student_id, interaction) in enumerate(
            kc_data.student_inter_dict.items()
        ):
            length = interaction.length
            student_id_segs.append(np.full(length, str(student_id), dtype=object))
            problem_id_segs.append(np.asarray(interaction.problem_ids, dtype=object))
            p_state_segs.append(p_state[student_idx, :length].astype(np.float64))
            correctness_segs.append(
                kc_data.correctness[student_idx, :length].astype(np.float64)
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
        p_state_vals = (
            np.concatenate(p_state_segs)
            if p_state_segs
            else np.empty(0, dtype=np.float64)
        )
        correctness_vals = (
            np.concatenate(correctness_segs)
            if correctness_segs
            else np.empty(0, dtype=np.float64)
        )
        n = len(student_ids)

        return pd.DataFrame(
            {
                "parameter": np.concatenate(
                    [
                        np.repeat(parameter_name, n),
                        np.repeat("true_correctness", n),
                    ]
                ),
                "student_id": np.concatenate([student_ids, student_ids]),
                "problem_id": np.concatenate([problem_ids, problem_ids]),
                "value": np.concatenate([p_state_vals, correctness_vals]),
            }
        )

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
        point_estimate: Literal["mean", "median"] = "mean",
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
    def _extract_param_point_estimate(
        fit: CmdStanFit,
        param_name: str,
        point_estimate: Literal["mean", "median"] = "mean",
    ) -> float:
        draws = BKTModelBase._extract_param_draws(fit, param_name)
        if point_estimate == "mean":
            return float(np.mean(draws))
        return float(np.median(draws))

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

    def predict_posterior(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[dict[str, str]] = None,
        posterior_draws: Optional[dict[str, pd.DataFrame]] = None,
        output: PosteriorPredictionOutput = "default",
        summary_quantiles: list[float] = [0.025, 0.975],
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame], dict[str, csp.CmdStanGQ]]:
        self._check_predict_posterior_args(
            data=data,
            calling_method="predict_posterior",
            posterior_draws=posterior_draws,
            output=output,
        )

        if data is not None and posterior_draws is not None:
            self._print(
                "Both 'data' and 'posterior_draws' are provided. 'posterior_draws' will be used and 'data' will be ignored.",
                level=VerbosityLevel.WARN,
            )
            data = None

        if data is not None:
            self._fit_check()
            if self._hidden_states_model is None:
                self._hidden_states_model = compile_stan_model(
                    self._stan_hidden_filename,
                    cpp_options=self.cpp_compile_kwargs,
                    stanc_options=self.stan_compile_kwargs,
                    print_fn=self._print,
                )

            posterior_draws_stan, kc_data_by_kc = self._predict_generated_quantities(
                data=data,
                gq_model=self._hidden_states_model,
                column_mapping=column_mapping,
            )

            if output == "stan":
                return posterior_draws_stan

            posterior_draws = type(self)._process_predict_gq(posterior_draws_stan)
            posterior_draws = type(self)._apply_kc_data_to_posterior_draws(
                posterior_draws=posterior_draws,
                kc_data_by_kc=kc_data_by_kc,
                drop_index_cols=output != "summary",
            )

        if posterior_draws is not None:
            if output == "summary":
                return type(self)._summarize_gq_state_predictions(
                    posterior_draws, quantiles=summary_quantiles
                )
            return posterior_draws

        raise ValueError(
            "Either 'data' or 'posterior_draws' must be provided. "
            "If you have precomputed posterior draws, pass them via 'posterior_draws'. "
            "Otherwise, provide the input data to generate new predictions."
        )

    def predict_smoothed_states_posterior(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[dict[str, str]] = None,
        posterior_draws: Optional[dict[str, pd.DataFrame]] = None,
        output: PosteriorPredictionOutput = "default",
        summary_quantiles: list[float] = [0.025, 0.975],
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame], dict[str, csp.CmdStanGQ]]:
        self._check_predict_posterior_args(
            data=data,
            calling_method="predict_smoothed_states_posterior",
            posterior_draws=posterior_draws,
            output=output,
        )

        if data is not None and posterior_draws is not None:
            self._print(
                "Both 'data' and 'posterior_draws' are provided. 'posterior_draws' will be used and 'data' will be ignored.",
                level=VerbosityLevel.WARN,
            )
            data = None

        if data is not None:
            self._fit_check()
            if self._smoothed_hidden_states_model is None:
                self._smoothed_hidden_states_model = compile_stan_model(
                    self._stan_smoothed_hidden_filename,
                    cpp_options=self.cpp_compile_kwargs,
                    stanc_options=self.stan_compile_kwargs,
                    print_fn=self._print,
                )

            posterior_draws_stan, kc_data_by_kc = self._predict_generated_quantities(
                data=data,
                gq_model=self._smoothed_hidden_states_model,
                column_mapping=column_mapping,
            )

            if output == "stan":
                return posterior_draws_stan

            posterior_draws = type(self)._process_predict_gq(posterior_draws_stan)
            posterior_draws = type(self)._apply_kc_data_to_posterior_draws(
                posterior_draws=posterior_draws,
                kc_data_by_kc=kc_data_by_kc,
                drop_index_cols=output != "summary",
            )

        if posterior_draws is not None:
            if output == "summary":
                return type(self)._summarize_gq_state_predictions(
                    posterior_draws, quantiles=summary_quantiles
                )
            return posterior_draws

        raise ValueError(
            "Either 'data' or 'posterior_draws' must be provided. "
            "If you have precomputed posterior draws, pass them via 'posterior_draws'. "
            "Otherwise, provide the input data to generate new predictions."
        )

    def _predict_generated_quantities(
        self,
        data: pd.DataFrame,
        gq_model: csp.CmdStanModel,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> tuple[dict[str, csp.CmdStanGQ], dict[str, KCData]]:
        kc_column_name = ColumnNames.KC_ID
        if column_mapping and ColumnNames.KC_ID in column_mapping:
            kc_column_name = column_mapping[ColumnNames.KC_ID]

        self.check_data_contains_fitted_kcs(
            set(data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(set(data[kc_column_name].unique()))
        data = data.copy()
        data = data.loc[data[kc_column_name].isin(overlapping_kcs)]

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model has been "
                "successfully fitted before calling prediction methods."
            )

        gq_kc_fit: dict[str, csp.CmdStanGQ] = {}
        kc_data_by_kc: dict[str, KCData] = {}
        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=False,
            print_fn=self._print,
        ):
            kc_id_str = str(kc_id)
            kc_data_by_kc[kc_id_str] = kc_data

            kc_fit_result = self.fits.get_fit(kc_id)
            if kc_fit_result is None:
                continue

            data_dict = self._build_stan_data_dict(kc_data)
            gq_fit = gq_model.generate_quantities(
                data=data_dict,
                previous_fit=kc_fit_result,
            )

            gq_kc_fit[kc_id_str] = gq_fit

        return gq_kc_fit, kc_data_by_kc

    @staticmethod
    def _process_predict_gq(
        posterior_draws_raw: dict[str, csp.CmdStanGQ],
    ) -> dict[str, pd.DataFrame]:
        expanded_param_names = ["parameter", "student_id", "problem_id"]
        posterior_draws: dict[str, pd.DataFrame] = {}
        for kc_id, gq_kc in posterior_draws_raw.items():
            gq_kc_df = gq_kc.draws_pd()
            id_cols: list[str] = [
                c for c in gq_kc_df.columns.astype(str) if c.endswith("__")
            ]
            long_df = gq_kc_df.melt(
                id_vars=id_cols,
                var_name="parameter_base",
                value_name="value",
            )
            parameter_lookup = pd.DataFrame(
                {
                    "parameter_base": pd.Index(
                        long_df["parameter_base"].unique(), dtype="string"
                    )
                }
            )
            parameter_lookup[expanded_param_names] = parameter_lookup[
                "parameter_base"
            ].str.extract(r"^([^\[]+)\[(\d+)\s*,\s*(\d+)\]$")
            parameter_lookup["parameter"] = parameter_lookup["parameter"].fillna(
                parameter_lookup["parameter_base"]
            )
            parameter_lookup[expanded_param_names[1]] = pd.to_numeric(
                parameter_lookup[expanded_param_names[1]], errors="coerce"
            ).astype("Int64")
            parameter_lookup[expanded_param_names[2]] = pd.to_numeric(
                parameter_lookup[expanded_param_names[2]], errors="coerce"
            ).astype("Int64")

            long_df = long_df.join(
                parameter_lookup.set_index("parameter_base"), on="parameter_base"
            )
            long_df = long_df.drop(columns=["parameter_base"], errors="ignore")
            long_df = long_df[
                expanded_param_names
                + ["value"]
                + [col for col in long_df.columns if col.endswith("__")]
            ]
            posterior_draws[kc_id] = long_df
        return posterior_draws

    @staticmethod
    def _apply_kc_data_to_posterior_draws(
        posterior_draws: dict[str, pd.DataFrame],
        kc_data_by_kc: dict[str, KCData],
        drop_index_cols: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Map posterior index coordinates back to original IDs and invalidate padded cells."""
        remapped_draws: dict[str, pd.DataFrame] = {}

        for kc_id, kc_draws in posterior_draws.items():
            kc_data = kc_data_by_kc.get(kc_id)
            if kc_data is None or kc_draws.empty:
                remapped_draws[kc_id] = kc_draws
                continue

            lookup_df = BKTModelBase._build_prediction_index_frame(
                kc_data=kc_data,
                n_problems=kc_data.correctness.shape[1],
            )
            lookup_df = lookup_df.rename(
                columns={
                    "student_id": "student_id_mapped",
                    "problem_id": "problem_id_mapped",
                    "is_valid": "is_valid_mapped",
                }
            )

            remapped = kc_draws.merge(
                lookup_df[
                    [
                        "student_idx",
                        "problem_idx",
                        "student_id_mapped",
                        "problem_id_mapped",
                        "is_valid_mapped",
                    ]
                ],
                left_on=["student_id", "problem_id"],
                right_on=["student_idx", "problem_idx"],
                how="left",
            )

            remapped["student_id"] = remapped["student_id"].astype(object)
            remapped["problem_id"] = remapped["problem_id"].astype(object)

            has_mapping = remapped["student_id_mapped"].notna()
            remapped.loc[has_mapping, "student_id"] = remapped.loc[
                has_mapping, "student_id_mapped"
            ]
            remapped.loc[has_mapping, "problem_id"] = remapped.loc[
                has_mapping, "problem_id_mapped"
            ]

            invalid_mask = remapped["is_valid_mapped"] == False
            remapped.loc[invalid_mask, "value"] = -1.0

            remapped = BKTModelBase._append_true_correctness_rows(
                prediction_df=remapped,
                kc_data=kc_data,
            )
            if drop_index_cols:
                remapped = remapped.drop(
                    columns=[
                        "student_idx",
                        "problem_idx",
                        "student_id_mapped",
                        "problem_id_mapped",
                        "is_valid_mapped",
                    ],
                    errors="ignore",
                )
            else:
                remapped = remapped.drop(
                    columns=[
                        "student_id_mapped",
                        "problem_id_mapped",
                        "is_valid_mapped",
                    ],
                    errors="ignore",
                )
            remapped_draws[kc_id] = remapped

        return remapped_draws

    @staticmethod
    def _append_true_correctness_rows(
        prediction_df: pd.DataFrame,
        kc_data: KCData,
    ) -> pd.DataFrame:
        """Append observed correctness values using the same long-form schema."""
        index_df = BKTModelBase._build_prediction_index_frame(
            kc_data=kc_data,
            n_problems=kc_data.correctness.shape[1],
        )
        id_cols = [col for col in prediction_df.columns if col.endswith("__")]

        if id_cols:
            draw_ids = prediction_df[id_cols].drop_duplicates(ignore_index=True)
            correctness_rows = draw_ids.merge(index_df, how="cross")
        else:
            correctness_rows = index_df.copy()

        correctness_rows["parameter"] = "true_correctness"
        correctness_rows["value"] = correctness_rows["correctness"].astype(np.float64)

        for base_col in ("student_id", "problem_id"):
            if base_col not in correctness_rows.columns:
                correctness_rows[base_col] = pd.Series(dtype="object")

        result_cols = list(prediction_df.columns)
        for col in result_cols:
            if col not in correctness_rows.columns:
                correctness_rows[col] = pd.NA

        correctness_rows = correctness_rows[result_cols]
        return pd.concat([prediction_df, correctness_rows], ignore_index=True)

    @staticmethod
    def _summarize_gq_state_predictions(
        gq_dict: dict[str, pd.DataFrame],
        quantiles=(0.025, 0.975),
    ) -> pd.DataFrame:
        if not gq_dict:
            raise ValueError("Input Dict is empty.")

        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1.")

        summary_cols = ["mean", "std", "median"] + [
            f"{q * 100:.2f}%" for q in quantiles
        ]
        param_id_cols = ["parameter", "student_id", "problem_id"]
        internal_id_cols = ["student_idx", "problem_idx"]

        result_frames: list[pd.DataFrame] = []
        for kc, gq_kc_df in gq_dict.items():
            if gq_kc_df.empty:
                continue

            group_cols = param_id_cols
            if all(col in gq_kc_df.columns for col in internal_id_cols):
                group_cols = param_id_cols + internal_id_cols

            gq_kc_summary: pd.DataFrame = (
                gq_kc_df.groupby(group_cols, sort=False)["value"]
                .agg(
                    mean="mean",
                    std="std",
                    median="median",
                    **{
                        f"{q * 100:.2f}%": (lambda s, q=q: s.quantile(q))
                        for q in quantiles
                    },
                )
                .reset_index()
            )

            gq_kc_summary = gq_kc_summary.drop(
                columns=internal_id_cols, errors="ignore"
            )

            gq_kc_summary.insert(0, "kc_id", str(kc))
            result_frames.append(gq_kc_summary)

        if not result_frames:
            return pd.DataFrame(columns=["kc_id"] + param_id_cols + summary_cols)

        return pd.concat(result_frames, ignore_index=True)

    @staticmethod
    def _check_predict_posterior_args(
        data: Optional[pd.DataFrame],
        calling_method: Literal[
            "predict_posterior", "predict_smoothed_states_posterior"
        ],
        posterior_draws: Optional[dict[str, pd.DataFrame]],
        output: PosteriorPredictionOutput,
    ):
        if output not in get_args(PosteriorPredictionOutput):
            raise ValueError(
                f"Invalid value for 'output': '{output}'. "
                f"Expected one of {get_args(PosteriorPredictionOutput)}."
            )

        if posterior_draws is None:
            return

        if isinstance(posterior_draws, pd.DataFrame):
            raise TypeError(
                "Invalid type for 'posterior_draws': received a pd.DataFrame. "
                "Expected a dict[str, DataFrame] mapping knowledge component IDs to "
                "posterior draw DataFrames."
            )
        if not isinstance(posterior_draws, dict):
            raise TypeError(
                "'posterior_draws' must be a dict mapping knowledge component IDs to "
                "pd.DataFrame or CmdStanGQ objects. Pass the return value of a previous "
                "call as 'posterior_draws'."
            )

        if calling_method not in (
            "predict_posterior",
            "predict_smoothed_states_posterior",
        ):
            raise ValueError(
                f"Invalid calling method: '{calling_method}'. "
                "Expected 'predict_posterior' or 'predict_smoothed_states_posterior'."
            )

        if output == "stan":
            raise TypeError(
                "'posterior_draws' cannot be used when 'output' is 'stan'. "
                "Precomputed posterior draws are incompatible with raw Stan output; "
                "omit 'posterior_draws' or choose a different 'output' format."
            )

        if not dict_has_types(posterior_draws, str, pd.DataFrame):
            raise TypeError(
                "'posterior_draws' must be a dict[str, pd.DataFrame] when 'output' is 'default' or 'summary'. "
                "Pass the DataFrame return value of a previous prediction call."
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
    def _build_stan_data_dict(self, kc_data: KCData) -> dict[str, Any]:
        raise NotImplementedError(
            "Subclasses must implement _build_stan_data_dict to support posterior predictions."
        )

    def _compile_model(self, stan_file: str | os.PathLike[str]) -> None:
        """Compile the Stan model and cache it."""

        self._stan_model = compile_stan_model(
            stan_file,
            stanc_options=self.stan_compile_kwargs,
            cpp_options=self.cpp_compile_kwargs,
            print_fn=self._print,
        )
