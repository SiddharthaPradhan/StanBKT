"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.

"""

from __future__ import annotations
from stanbkt.fits.fit_factory import FitFactory

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union, Final
import numpy as np
import numpy.typing as npt
import cmdstanpy as csp
import pandas as pd
import os
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_types import CmdStanFit
from stanbkt.fits.fit_types import FitMethod
from stanbkt.fits.core.base import BaseFit
from stanbkt.models.model_types import ModelType, PriorEstimationType
from stanbkt.models.error import FitMethodMismatchError
from stanbkt.models.priors import BayesianPriors
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import iter_kc_data


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
        fit_method: FitMethod = FitMethod.MCMC,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: Optional[Dict[str, Any]] = None,
        cpp_compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(verbose=verbose)
        self._fit_method: Final[FitMethod] = fit_method
        self.fit_class: Final[type[BaseFit]] = FitFactory.get_fit_class_from_method(
            fit_method
        )
        # TODO: NEED defaults for compile kwargs?
        # Does this need to be a dataclass?
        # TODO: catch the Error thrown and re-raise with custom error for invalid
        self.stan_compile_kwargs: dict[str, Any] = stan_compile_kwargs or {}
        self.cpp_compile_kwargs: dict[str, Any] = cpp_compile_kwargs or {}
        # Model is instantiated lazily during first fit and cached
        self._stan_model: Optional[csp.CmdStanModel] = None
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

    def _fit_check(self) -> None:
        """Check if model has been fitted."""
        if not self._is_fitted or self.fits is None or self.fits.num_fitted_kcs == 0:
            raise RuntimeError("Model must be fitted before calling this method")

    def check_data_contains_fitted_kcs(self, kcs: set[str]) -> None:
        """Check if data contains any KC that was fitted.
        Raises an error if data contains KCs that were not fitted.
        """
        self._fit_check()
        fitted_kcs: set[str] = set(self.fits.kc_fits.keys()) if self.fits else set()
        if not kcs.issubset(fitted_kcs):
            raise ValueError(
                f"Data contains no KCs that were previously fitted. Given KCs: {kcs}, fitted KCs: {fitted_kcs}"
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
        self._fit_check()
        fitted_kcs: set[str] = set(self.fits.kc_fits.keys()) if self.fits else set()
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

    def _compile_model(self, stan_file: str | os.PathLike[str]) -> None:
        """Compile the Stan model and cache it."""

        self._stan_model = compile_stan_model(
            stan_file,
            stanc_options=self.stan_compile_kwargs,
            cpp_options=self.cpp_compile_kwargs,
            print_fn=self._print,
        )
