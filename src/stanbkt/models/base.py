"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import cmdstanpy as csp
import pandas as pd
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from stanbkt.utils.data_utils import iter_kc_data

# expectations for data
# long format
# columns: student_id, problem_id, correctness (0/1), KC


class BKTModelBase(VerboseMixin, ABC):
    """Abstract base class for Stan Bayesian Knowledge Tracing (BKT) models.

    This class defines the interface that all Stan BKT model implementations must follow.

    Attributes
    ----------
    model_type : str
        The type of BKT model.
    model_name : str
        Name of the model.
    is_fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        model_type: str,
        verbose: int = 0,
        compile_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(verbose=verbose)
        self.model_name = self.__class__.__name__
        self.stan_compile_kwargs = compile_kwargs or {}
        # Model state attributes
        self._stan_model: Optional[csp.CmdStanModel] = None
        self.fits_: Optional[dict[str, Any]] = (
            None  # TODO replace type with fit base type
        )
        self.is_fitted: bool = False
        self._previous_fit_method: Optional[str] = None  # TODO

    @property
    @abstractmethod
    def _stan_model_filename(self) -> str:
        """Return stan file name inside stanbkt.stan_code."""
        pass


    @property
    @abstractmethod
    def _stan_files_base_location(self) -> str:
        """Return stan file base location inside stanbkt.stan_code."""
        pass

    @abstractmethod
    def fit(
        self, data: pd.DataFrame, column_mapping: Optional[dict[str, str]] = None, method: str = "sample", **kwargs
    ) -> "BKTModelBase":
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
        method : str, default="sample"
            Inference method: "sample" for MCMC, "variational" for VI,
            "optimize" for MAP, "pathfinder" for pathfinder.
        **kwargs
            Additional arguments including:
            - Stan sampling parameters (e.g., iter_sampling, chains, seed)

        Returns
        -------
        self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If data validation fails or invalid method specified.
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
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling summary()")

        if hasattr(self.fit, "summary"):
            if params:
                return self.fit.summary(var_names=params, percentiles=percentiles)
            return self._fit.summary(percentiles=percentiles)

        raise NotImplementedError("Summary not available for this fit method")

    def fit_check(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling this method")
        
    def check_data_contains_fitted_kcs(self, kcs: set[str]) -> None:
        """Check if data contains any KC that was fitted.
            Raises an error if data contains KCs that were not fitted.
        """
        self.fit_check()
        fitted_kcs = set(self.fits_.keys())
        if not kcs.issubset(fitted_kcs):
            raise ValueError(
                f"Data contains KCs {kcs - fitted_kcs} that were not fitted"
            )
        kcs_not_fitted = kcs - fitted_kcs
        if kcs_not_fitted:
            self._print(f"Data contains {len(kcs_not_fitted)} KCs that were not fitted.", level=VerbosityLevel.INFO)
            self._print(f"{list(kcs_not_fitted)} do not have fits.", level=VerbosityLevel.WARNING)

    def get_kc_in_fitted_kcs(self, kcs: set[str]) -> set[str]:
        """Return the set of KCs in the data that were fitted previously."""
        self.fit_check()
        fitted_kcs = set(self.fits_.keys())
        return kcs.intersection(fitted_kcs)

    def predict(
        self,
        correctness: Optional[npt.NDArray[np.int_]] = None,
        return_std: bool = False,
    ) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
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

        predictions = self._extract_predictions()

        if return_std:
            std = self._extract_prediction_std()
            return predictions, std
        return predictions

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
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling evaluate()")
        pass

    def _compile_model(self) -> None:
        """Compile the Stan model and cache it."""
        # TODO: add caching of compiled models to avoid recompilation
        # TODO: look into deprecated params argument in CmdStanModel constructor
        model = csp.CmdStanModel(
            stan_file=self.stan_filename,
            cpp_options=self.stan_compile_kwargs,
        )
        self._stan_model = model

    @abstractmethod
    def _extract_predictions(self) -> npt.NDArray:
        """Extract predictions from fitted model."""
        pass

    @abstractmethod
    def _extract_prediction_std(self) -> npt.NDArray:
        """Extract prediction standard deviations from fitted model."""
        pass
