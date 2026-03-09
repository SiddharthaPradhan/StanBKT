"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union
from enum import Enum
import numpy as np
import numpy.typing as npt
import cmdstanpy as csp
import pandas as pd
from stanbkt.utils.verbose import VerboseMixin, VerbosityLevel
from stanbkt.utils.data_utils import iter_kc_data

# expectations for data
# long format
# columns: student_id, problem_id, correctness (0/1), KC


# Types (enums) for model configuration
class ModelType(str, Enum):
    STANDARD = "standard"
    GROUPED = "grouped"
    NESTED = "nested"


class FitMethodType(str, Enum):
    MCMC = "sample"
    VB = "variational"
    MLE = "optimize"
    PF = "pathfinder"


class PriorEstimationType(str, Enum):
    JOINT = "joint"
    DEFAULT = "default"


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
        method: FitMethodType = FitMethodType.MCMC,
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


# Parameter Classes
class BayesianPriors(str, Enum):
    PI_KNOW_MU = "pi_know_mu"
    PI_KNOW_STD = "pi_know_std"
    LEARN_MU = "learn_mu"
    LEARN_STD = "learn_std"
    FORGET_MU = "forget_mu"
    FORGET_STD = "forget_std"
    GUESS_MU = "guess_mu"
    GUESS_STD = "guess_std"
    SLIP_MU = "slip_mu"
    SLIP_STD = "slip_std"

    @staticmethod
    def _default_scalar_priors() -> dict[BayesianPriors, float]:
        return {
            BayesianPriors.PI_KNOW_MU: -2.0,
            BayesianPriors.PI_KNOW_STD: 5.0,
            BayesianPriors.LEARN_MU: 0.0,
            BayesianPriors.LEARN_STD: 5.0,
            BayesianPriors.FORGET_MU: -2.0,
            BayesianPriors.FORGET_STD: 5.0,
            BayesianPriors.GUESS_MU: -1.0,
            BayesianPriors.GUESS_STD: 5.0,
            BayesianPriors.SLIP_MU: -1.0,
            BayesianPriors.SLIP_STD: 5.0,
        }

    # TODO: post initial release: add features to customize the number of groups for each of the parameters.
    # for example we can fix learning and forgetting to be the same across groups,
    # but allow guess and slip to vary by group.
    @staticmethod
    def _expand_grouped_priors(
        scalar_priors: dict[BayesianPriors, float],
        n_groups: int,
    ) -> dict[BayesianPriors, list[float]]:
        return {prior: [value] * n_groups for prior, value in scalar_priors.items()}

    # TODO: add additional priors for advanced models
    @staticmethod
    def get_default_priors(
        model_type: ModelType,
        estimation_type: PriorEstimationType,
        n_groups: Optional[int] = None,
    ) -> Union[dict[BayesianPriors, float], dict[BayesianPriors, list[float]]]:
        """Return default priors used for BKT parameters.

        Notes
        -----
        Priors are modeled as Normal distributions with means and standard deviations specified on the logit scale
        for probability parameters.
        """
        if estimation_type == PriorEstimationType.JOINT:
            raise NotImplementedError(
                "Joint prior estimation defaults are not implemented yet"
            )

        if estimation_type != PriorEstimationType.DEFAULT:
            raise ValueError(f"Unsupported prior estimation type: {estimation_type}")

        scalar_priors = BayesianPriors._default_scalar_priors()

        if model_type in [ModelType.STANDARD, ModelType.NESTED]:
            return scalar_priors

        if model_type == ModelType.GROUPED:
            if not isinstance(n_groups, int):
                raise ValueError(
                    "n_groups must be an integer for default grouped model priors"
                )
            if n_groups <= 0:
                raise ValueError("n_groups must be > 0 for grouped model priors")
            return BayesianPriors._expand_grouped_priors(scalar_priors, n_groups)

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def add_missing_priors(
        values: Mapping[BayesianPriors | str, float | list[float]],
        model_type: ModelType,
        estimation_type: PriorEstimationType,
        n_groups: Optional[int] = None,
    ) -> Union[dict[BayesianPriors, float], dict[BayesianPriors, list[float]]]:
        """Fill missing priors with defaults and return a normalized dictionary.

        Parameters
        ----------
        values : Mapping[BayesianPriors | str, float | list[float]]
            Partial prior values keyed by `BayesianPriors` or by string names.
            Missing keys are filled from defaults.
        model_type : ModelType
            BKT model type for selecting prior structure.
        estimation_type : PriorEstimationType
            Prior estimation mode.
        n_groups : int, optional
            Number of groups for grouped models.
        """
        defaults = dict(
            BayesianPriors.get_default_priors(
                model_type=model_type,
                estimation_type=estimation_type,
                n_groups=n_groups,
            )
        )

        normalized_values: dict[BayesianPriors, float | list[float]] = {}
        for key, value in values.items():
            if isinstance(key, BayesianPriors):
                prior_key = key
            elif isinstance(key, str):
                # ensure string keys are valid and members of the BayesianPriors enum
                try:
                    prior_key = BayesianPriors(key)
                except ValueError as exc:
                    raise ValueError(f"Unsupported prior key: {key}") from exc
            else:
                raise ValueError(f"Unsupported prior key type: {type(key).__name__}")

            normalized_values[prior_key] = value

        defaults.update(normalized_values)
        return defaults
