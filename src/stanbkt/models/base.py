"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import cmdstanpy as csp
import pandas as pd
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from stanbkt.fits.fit_types import FitMethod
from stanbkt.models.model_types import ModelType, PriorEstimationType
from stanbkt.models.priors import BayesianPriors
from stanbkt.utils.data_utils import iter_kc_data

# expectations for data
# long format
# columns: student_id, problem_id, correctness (0/1), KC


class BKTModelBase(VerboseMixin, ABC):
    """Abstract base class for Stan Bayesian Knowledge Tracing (BKT) models.

    This class defines the interface that all Stan BKT model implementations must follow.

    Attributes
    ----------
    verbose : VerbosityLevel
        Verbosity level for logging.
    stan_compile_kwargs : dict
        Additional keyword arguments for Stan model compilation.
    cpp_compile_kwargs : dict
        Additional keyword arguments for C++ compilation of the Stan model.
    fits_ : Optional[dict[str, Any]]
        Dictionary to store fitted model results for each KC.
    """

    def __init__(
        self,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: Optional[Dict[str, Any]] = None,
        cpp_compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(verbose=verbose)
        self.stan_compile_kwargs = stan_compile_kwargs or {}
        self.cpp_compile_kwargs = cpp_compile_kwargs or {}
        # Model state attributes
        self._stan_model: Optional[csp.CmdStanModel] = None
        self.fits_: Optional[dict[str, Any]] = (
            None  # TODO replace type with fit base type
        )
        self._is_fitted: bool = False
        self._previous_fit_method: Optional[str] = None  # TODO

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
        method: FitMethod = FitMethod.MCMC,
        stan_fit_kwargs: Optional[dict[str, Any]] = None,
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
        method : FitMethodType, default=FitMethodType.MCMC
            Inference method: FitMethodType.MCMC for MCMC, FitMethodType.VB for VI,
            FitMethodType.OPTIMIZE for MAP, FitMethodType.PATHFINDER for pathfinder.
        stan_fit_kwargs : dict, optional
            Arguments to pass to the CmdStanPy fit method. This depends on the chosen method.
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

    def summary(
        self,
        params: Optional[list[str]] = None,
        percentiles: Tuple[float, ...] = (5, 50, 95),
    ) -> Any:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        params : list of str, optional
            List of parameters to summarize. If None, summarizes all.
        percentiles : tuple of float, default=(5, 50, 95)
            Percentiles to include in summary.

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

    def fit_check(self) -> None:
        """Check if model has been fitted."""
        if not self._is_fitted or self.fits_ is None:
            raise RuntimeError("Model must be fitted before calling this method")

    def check_data_contains_fitted_kcs(self, kcs: set[str]) -> None:
        """Check if data contains any KC that was fitted.
        Raises an error if data contains KCs that were not fitted.
        """
        self.fit_check()
        fitted_kcs = set(self.fits_.keys()) if self.fits_ else set()
        if not kcs.issubset(fitted_kcs):
            raise ValueError(
                f"Data contains KCs {kcs - fitted_kcs} that were not fitted"
            )
        kcs_not_fitted = kcs - fitted_kcs
        if kcs_not_fitted:
            self._print(
                f"Data contains {len(kcs_not_fitted)} KCs that were not fitted.",
                level=VerbosityLevel.INFO,
            )
            self._print(
                f"{list(kcs_not_fitted)} do not have fits.",
                level=VerbosityLevel.WARN,
            )

    def get_kc_in_fitted_kcs(self, kcs: set[str]) -> set[str]:
        """Return the set of KCs in the data that were fitted previously."""
        self.fit_check()
        fitted_kcs = set(self.fits_.keys()) if self.fits_ else set()
        return kcs.intersection(fitted_kcs)

    # TODO FIXME
    #
    def predict(self) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
        """
        Predict hidden knowledge states.

        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame containing the training data. Must include columns:
            'student_id', 'problem_id', and 'correctness' (0/1).
        return_std : bool, default=False
            Whether to return standard deviations of predictions.

        Returns
        -------
        predictions : np.ndarray
            Predicted probability of mastery for each student and problem.
        std : np.ndarray, optional
            Standard deviation of predictions (only if return_std=True).

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")

        raise NotImplementedError("Predict method not implemented for this model")

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

    def _compile_model(self) -> None:
        """Compile the Stan model and cache it."""
        # TODO: add caching of compiled models to avoid recompilation
        # TODO: look into deprecated params argument in CmdStanModel constructor
        model = csp.CmdStanModel(
            stan_file=self._stan_model_filename,
            cpp_options=self.cpp_compile_kwargs,
            stanc_options=self.stan_compile_kwargs,
        )
        self._stan_model = model
