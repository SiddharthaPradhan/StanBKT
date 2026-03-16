from __future__ import annotations
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_factory import FitFactory
from stanbkt.fits.core.base import BaseFit

from importlib.resources import files
from typing import Any, Optional, Union

import cmdstanpy as csp
import numpy as np
import numpy.typing as npt
import pandas as pd

from stanbkt.models.core.base import BKTModelBase, FitMethod
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import (
    ColumnNames,
    iter_kc_data,
    summarize_state_predictions,
)
from stanbkt.utils.verbose import VerbosityLevel


class StandardBKT(BKTModelBase):
    """
    Bayesian Knowledge Tracing (BKT) model implementation.

    This class implements the standard BKT model using Bayesian inference.
    It estimates population-level parameters pT, pF, pG, and pS for each knowledge component.
    For each KC, a separate Stan model is fitted independently.


    Parameters
    ----------
    fit_method : FitMethod, default=FitMethod.MCMC
        The method to use for fitting the Stan model. Supported methods include:
        - FitMethod.MCMC: Markov Chain Monte Carlo sampling for full posterior inference.
        - FitMethod.MLE: Maximum Likelihood Estimation for point estimates of parameters.
        - FitMethod.VB: Variational Bayes for faster approximate inference.
        - FitMethod.PATHFINDER: Pathfinder variational approximation for fast inference.
    verbose : VerbosityLevel, default=VerbosityLevel.INFO
        Verbosity level for logging during fitting and prediction.
    stan_compile_kwargs : dict, optional
        Additional Stan compile options forwarded as ``stanc_options`` to
        :external+cmdstanpy:py:class:`cmdstanpy.CmdStanModel`.
    cpp_compile_kwargs : dict, optional
        Additional C++ compile options forwarded as ``cpp_options`` to
        :external+cmdstanpy:py:class:`cmdstanpy.CmdStanModel`.

    Attributes
    ----------
    model_ : CmdStanModel
        The compiled Stan model.
    fits_ : dict[str, McmcFit | VbFit | MleFit]
        The fitted models after training. Each KC will have its own fitted model.
    """

    def __init__(
        self,
        fit_method: FitMethod = FitMethod.MCMC,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: dict | None = None,
        cpp_compile_kwargs: dict | None = None,
    ):
        super().__init__(
            verbose=verbose,
            fit_method=fit_method,
            stan_compile_kwargs=stan_compile_kwargs,
            cpp_compile_kwargs=cpp_compile_kwargs,
        )
        self._hidden_states_model: Optional[csp.CmdStanModel] = None
        self._smoothed_hidden_states_model: Optional[csp.CmdStanModel] = None

    @property
    def _stan_model_filename(self) -> str:
        return str(files("stanbkt").joinpath("stan_code", "BKT", "BKT_model.stan"))

    @property
    def _stan_hidden_filename(self) -> str:
        return str(files("stanbkt").joinpath("stan_code", "BKT", "hidden_states.stan"))

    @property
    def _stan_smoothed_hidden_filename(self) -> str:
        return str(
            files("stanbkt").joinpath("stan_code", "BKT", "smoothed_hidden_states.stan")
        )

    def fit(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
        stan_fit_options: Optional[Union[StanFitOptions, dict[str, Any]]] = None,
    ) -> StandardBKT:
        """
        Fit the BKT model to data. Each KC is fitted independently with its own model.
        Additional KCs can be fitted by calling fit again with new data.

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
                Additional keyword arguments to pass to the Stan fitting method. If a dict is passed, it will be forwarded as-is to the CmdStanPy fit method.
                It is recommended to use the typed :class:`StanFitOptions` for better type checking and validation. The accepted options depend on the chosen fit method. For example:
                - MCMC parameters (e.g., iter_sampling, chains, seed)
                - VB parameters (e.g., iter, tol_rel_obj)
                If None, default fitting options for the chosen fit method will be used.


        Returns
        -------
        StandardBKT
            The fitted StandardBKT model instance.

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
                "Failed to compile and load the Stan model. Cannot fit the model."
            )

        # check stan_fit_options
        if stan_fit_options is None:
            stan_fit_options = FitFactory.create_default_fit_options(self._fit_method)
        else:
            # convert to StanFitOptions if it is a dict, mainly for better type checking and validation, but also to ensure compatibility with the FitFactory verification method
            if isinstance(stan_fit_options, dict):
                stan_fit_options = FitFactory.create_fit_options_from_dict(
                    stan_fit_options, self._fit_method
                )
            # verify compatibility of provided options with the fit method
            FitFactory.verify_fit_options_compatibility(
                stan_fit_options, self._fit_method
            )
        # convert to dict for CmdStanPy
        fit_options_dict = stan_fit_options.to_dict()

        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=False,
            print_fn=self._print,
        ):
            self._print(f"Fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)

            correctness = kc_data.correctness
            n_students, n_problems = correctness.shape
            # TODO move this to data helper
            data_dict = {
                "nStudents": int(n_students),
                "nProblems": int(n_problems),
                "correctness": correctness,
                "nGroups": 1,  # standard BKT does not use groups
                "groups": np.ones(n_students, dtype=np.int32),
            }
            fit_result = self._fit_stan_model_using_method(
                data_dict=data_dict, fit_options=stan_fit_options
            )
            self.fits.add_fit(str(kc_id), fit_result)

        self._is_fitted = True
        return self

    def _build_stan_data_dict(self, correctness: np.ndarray) -> dict[str, Any]:
        """Build data dictionary for the stan models.

        Parameters
        ----------
        correctness : np.ndarray
            Array of correctness values with shape
            ``(n_students, n_problems)``.

        Returns
        -------
        dict[str, Any]
            Dictionary containing data for the Stan model.
        """
        n_students, n_problems = correctness.shape
        return {
            "nStudents": int(n_students),
            "nProblems": int(n_problems),
            "correctness": correctness,
            "nGroups": 1,
            "groups": np.ones(n_students, dtype=np.int32),
        }

    def _predict_generated_quantities(
        self,
        data: pd.DataFrame,
        gq_model: csp.CmdStanModel,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:

        kc_column_name = ColumnNames.KC_ID
        if column_mapping and ColumnNames.KC_ID in column_mapping:
            kc_column_name = column_mapping[ColumnNames.KC_ID]

        self.check_data_contains_fitted_kcs(set(data[kc_column_name].unique()))
        overlapping_kcs = self.get_kc_in_fitted_kcs(set(data[kc_column_name].unique()))
        data = data.copy()  # avoid modifying original data
        data = data.loc[data[kc_column_name].isin(overlapping_kcs)]

        if self.fits is None:
            raise RuntimeError(
                "Fits object is not initialized. Cannot call generated quantities for prediction."
            )

        result_frames: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=False,
            print_fn=self._print,
        ):
            fit_result = self.fits.get_fit(kc_id)
            if fit_result is None:
                continue

            data_dict = self._build_stan_data_dict(kc_data.correctness)
            gq_fit = gq_model.generate_quantities(
                data=data_dict,
                previous_fit=fit_result,
            )

            # make pretty and store in dict

            # return the dict

            # TODO: we do not want to summarize this directly in the predict method
            # instead we should create a base class for generated quantity predictions
            # and have it support summarize_state_predictions as a method that can
            # be called on the object.
            # Currently working on summarize_state_predictions_test version. Complete that!!

            summary_df = summarize_state_predictions(gq_fit.draws_pd().T).reset_index()
            summary_df = summary_df.rename(columns={"index": "variable"})
            summary_df[kc_column_name] = kc_id
            result_frames.append(summary_df)

        return pd.concat(result_frames, ignore_index=True)

    # TODO: add numba implementation here
    def predict():
        pass

    # TODO: add numba implementation here
    def predict_smoothed_states():
        pass

    # TODO for posterior functions, check that the fit method was MCMC, VB or Pathfinder

    def predict_posterior(
        self,
        data: pd.DataFrame,
        posterior=True,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Predict probability of hidden-states (:math:`p(hidden_{t} \\mid correct_{1:t})`) and .


        """
        self._fit_check()

        if self._hidden_states_model is None:
            self._hidden_states_model = compile_stan_model(
                self._stan_hidden_filename,
                cpp_options=self.cpp_compile_kwargs,
                stanc_options=self.stan_compile_kwargs,
                print_fn=self._print,
            )

        return self._predict_generated_quantities(
            data=data,
            gq_model=self._hidden_states_model,
            column_mapping=column_mapping,
        )

    def predict_smoothed_states_posterior(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Generate the posterior of the smoothed probability of hidden-states."""
        self._fit_check()

        if self._smoothed_hidden_states_model is None:
            self._smoothed_hidden_states_model = compile_stan_model(
                self._stan_smoothed_hidden_filename,
                cpp_options=self.cpp_compile_kwargs,
                stanc_options=self.stan_compile_kwargs,
                print_fn=self._print,
            )

        return self._predict_generated_quantities(
            data=data,
            gq_model=self._smoothed_hidden_states_model,
            column_mapping=column_mapping,
        )

    def evaluate(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("evaluate is not implemented for StandardBKT yet")
