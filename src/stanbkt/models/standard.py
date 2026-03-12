from __future__ import annotations

from importlib.resources import files
from typing import Any, Optional

import cmdstanpy as csp
import numpy as np
import numpy.typing as npt
import pandas as pd

from stanbkt.models.base import BKTModelBase, FitMethod
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
    compile_kwargs : dict, optional
        Additional C++ compile options forwarded as ``cpp_options`` to
        :external+cmdstanpy:py:class:`cmdstanpy.CmdStanModel`.
        See :external+cmdstanpy:py:meth:`cmdstanpy.CmdStanModel.__init__`
        for accepted values.

    Attributes
    ----------
    model_ : CmdStanModel
        The compiled Stan model.
    fits_ : dict[str, McmcFit | VbFit | MleFit]
        The fitted models after training. Each KC will have its own fitted model.
    """

    def __init__(
        self,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: dict | None = None,
        cpp_compile_kwargs: dict | None = None,
    ):
        super().__init__(
            verbose=verbose,
            stan_compile_kwargs=stan_compile_kwargs,
            cpp_compile_kwargs=cpp_compile_kwargs,
        )
        self._hidden_states_model: Optional[csp.CmdStanModel] = None
        self._smoothed_hidden_states_model: Optional[csp.CmdStanModel] = None

    def fit(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
        method: FitMethod = FitMethod.MCMC,
        stan_fit_kwargs: Optional[dict[str, Any]] = None,
    ) -> StandardBKT:
        """
        Fit the BKT model to data. Each KC is fitted independently with its own model.
        Additional KCs can be fitted by calling fit again with new data. However, refitting
        with a different method is not allowed on the same model instance.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the training data. Must include columns for:
            Student ID, Problem ID, and Correctness (0/1).
            If the KC column is absent, all interactions are assumed to belong to a single knowledge component.
        column_mapping : dict, optional
            Mapping of expected column names. Keys should be 'student_id', 'problem_id', 'correct', and 'kc_id'.
            If None, default column names are used.
        method : FitMethod, default='sample'
            Method for fitting the model. Options are 'sample' for MCMC sampling,
            'vb' for variational inference, and 'optimize' for MAP estimation.
        **kwargs:
            Method-specific keyword arguments forwarded to :mod:`cmdstanpy`.

            Supported by ``method``:

            - ``method='sample'``: passed to :py:meth:`cmdstanpy.CmdStanModel.sample`
                (for example: ``iter_sampling``, ``iter_warmup``, ``chains``,
                ``parallel_chains``, ``thin``, ``seed``, ``show_progress``).
            - ``method='vb'``: passed to :py:meth:`cmdstanpy.CmdStanModel.variational`
                when implemented.
            - ``method='optimize'``: passed to :py:meth:`cmdstanpy.CmdStanModel.optimize`
                when implemented.

        Returns
        -------
        StandardBKT
            The fitted StandardBKT model instance.

        Raises
        ------
        ValueError
            If data validation fails or invalid method specified.
        """
        if self._previous_fit_method is None:
            self._previous_fit_method = method
        elif method != self._previous_fit_method:
            raise ValueError(
                f"Model was previously fitted with method '{self._previous_fit_method}'. "
                f"Refitting with a different method '{method}' is not allowed. "
                "Please create a new model instance to fit with a different method."
            )

        # TODO: remove after implementing VB and MLE fitting methods
        if method != "sample":
            raise ValueError(
                "Only method='sample' is currently implemented for StandardBKT."
            )

        # TODO need caching logic here, create and use a util method here.
        # Move logic to __init__
        # no need for check everytime when fit is called
        # verification of the stan and cpp options
        if self._stan_model is None:
            self._stan_model = csp.CmdStanModel(
                stan_file=self._stan_model_filename,
                cpp_options=self.cpp_compile_kwargs,
                stanc_options=self.stan_compile_kwargs,
            )

        fits: dict[str, Any] = {}
        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=False,
            print_fn=self._print,
        ):
            self._print(f"Fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)

            correctness = kc_data.correctness
            n_students, n_problems = correctness.shape
            data_dict = {
                "nStudents": int(n_students),
                "nProblems": int(n_problems),
                "correctness": correctness,
                "nGroups": 1,
                "groups": np.ones(n_students, dtype=np.int32),
            }

            fit_result = self._stan_model.sample(
                data=data_dict,
                seed=[1, 2, 3, 4],
                chains=4,
                threads_per_chain=4,
                parallel_chains=4,
                iter_sampling=1500,
                iter_warmup=2000,
                thin=2,
            )
            fits[kc_id] = fit_result

        self.fits_ = fits
        self._is_fitted = True

        return self

    # TODO complete this method, no
    def _fit_using_method(
        self, method: FitMethod, data_dict: dict[str, Any], **kwargs
    ) -> Any:

        if method == FitMethod.MCMC:
            return self._stan_model.sample(data=data_dict, **kwargs)
        elif method == FitMethod.VB:
            raise NotImplementedError(
                "Variational inference fitting is not implemented yet."
            )
        elif method == FitMethod.MLE:
            raise NotImplementedError("MAP estimation fitting is not implemented yet.")
        else:
            raise ValueError(
                f"Invalid fitting method '{method}'. Supported methods are '{FitMethod.MCMC}', '{FitMethod.VB}', and '{FitMethod.MLE}'."
            )

    def _build_stan_data_dict(self, correctness: np.ndarray) -> dict[str, Any]:
        """Build data dictionary for the stan models.

        Args:
            correctness (np.ndarray): Array of correctness values with shape (n_students, n_problems).

        Returns:
            dict[str, Any]: Dictionary containing data for the Stan model.
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

        result_frames: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=False,
            print_fn=self._print,
        ):
            fit_result = self.fits_.get(kc_id)
            if fit_result is None:
                continue

            data_dict = self._build_stan_data_dict(kc_data.correctness)
            gq_fit = gq_model.generate_quantities(
                data=data_dict,
                previous_fit=fit_result,
            )

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
        self.fit_check()

        if self._hidden_states_model is None:
            self._hidden_states_model = csp.CmdStanModel(
                stan_file=self._stan_hidden_filename,
                cpp_options=self.stan_compile_kwargs,
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
        self.fit_check()

        if self._smoothed_hidden_states_model is None:
            self._smoothed_hidden_states_model = csp.CmdStanModel(
                stan_file=self._stan_smoothed_hidden_filename,
                cpp_options=self.stan_compile_kwargs,
            )

        return self._predict_generated_quantities(
            data=data,
            gq_model=self._smoothed_hidden_states_model,
            column_mapping=column_mapping,
        )

    def evaluate(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("evaluate is not implemented for StandardBKT yet")

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
