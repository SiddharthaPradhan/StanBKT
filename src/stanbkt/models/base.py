"""
Base abstract class for BKT models using Stan.

This module provides the abstract base class that all BKT model implementations
should inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, Tuple
import numpy as np
import numpy.typing as npt
import cmdstanpy as csp



PARAMETERS = Literal["learn", "forget", "guess", "slip"]
# Meta model types
ModelType = Literal[
    "population",  # parameters are shared across all students, complete pooling
    "grouped",  # parameters vary by a single grouping factor, no pooling
    # The following two require a group and/or student-level identifier
    "nested-population",  # group parameters are drawn from global parameters, partial pooling
    "nested-grouped" # individuals parameters are drawn from group paramters
]



class BKTModelBase(ABC):
    """Abstract base class for Stan Bayesian Knowledge Tracing (BKT) models.

    This class defines the interface that all Stan BKT model implementations must follow.

    Attributes
    ----------
    model_type: ModelType
        The type of BKT model (e.g., "global", "nested", "grouped").
    model_name : str
        Name of the model.
    stan_file : str or Pathlike or None
        Path to the Stan model file.
    model_ : CmdStanModel
        The compiled Stan model.
    fits_ : Dict[str, FitType]
        Dictionary that maps Knowledge Components to it's corresponding Fit Object.
    is_fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(
        self, model_type: ModelType, compile_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type
        self.model_name = self.__class__.__name__
        self.compile_kwargs = compile_kwargs or {}
        self.stan_file = None
        self.params = {}

        # Model state attributes
        # Compiled Stan Model
        self._stan_model: Optional[csp.CmdStanModel] = None
        # Fitted model results
        self._fits: Optional[Any] = None
        # Fitted flag
        self._is_fitted: bool = False
        # Data used for fitting (this is built before fitting)
        self._data: Optional[Dict[str, Any]] = None

    def compile(self, force_recompile: bool = False) -> "BKTModelBase":
        """
        Compile the Stan model.

        Parameters
        ----------
        force_recompile : bool, default=False
            Whether to force recompilation even if model is already compiled.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if self._stan_model is None or force_recompile:
            if not self._stan_file.exists():
                raise FileNotFoundError(f"Stan file not found: {self._stan_file}")

            self._stan_model = csp.CmdStanModel(
                stan_file=str(self._stan_file), **self.compile_kwargs
            )
        return self

    def pin_parameters(self, **params: Dict[PARAMETERS, Union[float, npt.ArrayLike]]) -> "BKTModelBase":
        """
        Pin model parameters to fixed values.

        Parameters
        ----------
        **params
            Model parameters to pin (e.g., learn=0.2, forget=0.1).

        Returns
        -------
        self
            Returns self for method chaining.
        """
        # TODO do some sort of checking here.
        self.params.update(params)
        return self

    def fit(
        self, correctness: npt.NDArray[np.int_], method: str = "sample", **kwargs
    ) -> "BKTModelBase":
        """
        Fit the BKT model to data.

        Parameters
        ----------
        correctness : np.ndarray
            Binary correctness matrix of shape (n_students, n_problems).
            Values should be 0 (incorrect) or 1 (correct).
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
        # Validate data
        self._validate_data(correctness, **kwargs)

        # Prepare data for Stan
        self.data_ = self._build_data_json(correctness, **kwargs)

        # Compile model if needed
        if self.model_ is None:
            self.compile()

        # Extract Stan-specific kwargs
        stan_kwargs = self._extract_stan_kwargs(kwargs, method)

        # Fit model
        if method == "sample":
            self.fit_ = self.model_.sample(data=self.data_, **stan_kwargs)
        elif method == "variational":
            self.fit_ = self.model_.variational(data=self.data_, **stan_kwargs)
        elif method == "optimize":
            self.fit_ = self.model_.optimize(data=self.data_, **stan_kwargs)
        elif method == "pathfinder":
            self.fit_ = self.model_.pathfinder(data=self.data_, **stan_kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Choose from 'sample', 'variational', 'optimize', 'pathfinder'."
            )

        self.is_fitted_ = True
        return self

    # prepare input data for Stan model
    def _build_data_json(
        self, correctness: npt.NDArray[np.int_], **kwargs
    ) -> Dict[str, Any]:
        """
        Build data dictionary for Stan model.

        Parameters
        ----------
        correctness : np.ndarray
            Binary correctness matrix of shape (n_students, n_problems).
        **kwargs
            Additional arguments for data preparation.

        Returns
        -------
        dict
            SelfoStan model.
        """
        n_students, n_problems = correctness.shape
        data = {
            "N_students": n_students,
            "N_problems": n_problems,
            "correctness": correctness.tolist(),
        }
        # Add pinned parameters if any
        for param, value in self.params.items():
            data[f"pin_{param}"] = 1
            data[f"{param}_value"] = value
        return data

    def predict(
        self,
        correctness: Optional[npt.NDArray[np.int_]] = None,
        return_std: bool = False,
    ) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
        """
        Predict hidden knowledge states.

        Parameters
        ----------
        correctness : np.ndarray, optional
            Binary correctness matrix. If None, uses training data.
        return_std : bool, default=False
            Whether to return standard deviations of predictions.

        Returns
        -------
        predictions : np.ndarray
            Predicted probability of mastery for each student and problem.
            Shape: (n_students, n_problems, n_states) or mean over states.
        std : np.ndarray, optional
            Standard deviation of predictions (only if return_std=True).

        Raises
        ------
        RuntimeError
            If model has not been fitted yet.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict()")

        predictions = self._extract_predictions()

        if return_std:
            std = self._extract_prediction_std()
            return predictions, std
        return predictions

    # This is heavily dependent on the method that was used to fit the model.
    # I should abstract out the fit to another class and add it to this class.
    # TODO
    # and I can just call the fits .summary method.
    
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
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")

        if hasattr(self.fit_, "summary"):
            if params:
                return self.fit_.summary(var_names=params, percentiles=percentiles)
            return self.fit_.summary(percentiles=percentiles)

        raise NotImplementedError("Summary not available for this fit method")

    def predict():
        pass
    
    def get_smoothed_state_probs():
        pass
    
    def get_PSIS_LOO():
        pass
    
    # call GQ from CmdStanPy
    def _generate_quantity():
        pass
    
    
    # check if the object has been fitted
    def _fitted_check():
        pass


    # only available for those that were fit with MCMC
    def diagnose(self) -> Dict[str, Any]:
        """
        Run diagnostics on the fitted model.

        Returns
        -------
        dict
            Dictionary of diagnostic information.

        Raises
        ------
        RuntimeError
            If model has not been fitted yet or not using MCMC.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before running diagnostics")

        if not hasattr(self.fit_, "diagnose"):
            raise NotImplementedError("Diagnostics only available for MCMC sampling")

        diagnostics = {}

        # Get divergences
        if hasattr(self.fit_, "divergences"):
            diagnostics["n_divergences"] = self.fit_.divergences

        # Get Rhat
        summary = self.fit_.summary()
        if "R_hat" in summary.columns:
            diagnostics["max_rhat"] = summary["R_hat"].max()
            diagnostics["n_high_rhat"] = (summary["R_hat"] > 1.05).sum()

        return diagnostics

    def __repr__(self) -> str:
        """String representation of the model."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.model_type}({fitted_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


# Move this to a stan helper file that loads,
# compiles, and stores exe logic
def build_stan_file(model_type: ModelType, **kwargs):
    """Build a Stan file for the specified model type.
       Assembles the necessary components based model type and kwargs and
       writes them to a .stan file.

    Args:
        model_type (ModelType): The type of the model to build.
    """
    # For now, we assume a fixed stan files for each model type.
    # TODO: dynamically build stan files using string templates.
    # Requires the .stan file to be modularized and template-ified.
    base_mapping = {
        "global": "../stan_code/bkt_global.stan",
        "nested": "../stan_code/bkt_nested.stan",
        "grouped": "../stan_code/bkt_grouped.stan",
    }
    if model_type not in base_mapping:
        raise ValueError(f"Unknown model type: {model_type}")

    stan_file = base_mapping[model_type]
    with open(stan_file, "r") as f:
        return f.read()
