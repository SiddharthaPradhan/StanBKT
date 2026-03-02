from importlib.resources import files
from typing import Any, Optional

import cmdstanpy as csp
import numpy as np
import numpy.typing as npt
import pandas as pd

from stanbkt.models.base import BKTModelBase
from stanbkt.utils.data_utils import ColumnNames, iter_kc_data, summarize_state_predictions
from stanbkt.utils.verbose import VerbosityLevel


class StandardBKT(BKTModelBase):
    """
    Bayesian Knowledge Tracing (BKT) model implementation.

    This class implements the standard BKT model using Bayesian inference.
    It extends the BKTModelBase class and provides methods for fitting the model,
    predicting student knowledge states, and generating additional quantities.

    Parameters
    ----------
    stan_file : str or Path, optional
        Path to the Stan model file. If None, uses the default BKT Stan model.
    compile_kwargs : dict, optional
        Additional keyword arguments for compiling the Stan model.

    Attributes
    ----------
    model_ : CmdStanModel
        The compiled Stan model.
    fit_ : CmdStanMCMC | CmdStanVB | CmdStanOptimize
        The fitted model after training.
    """
    def __init__(self, verbose: VerbosityLevel = VerbosityLevel.INFO, compile_kwargs: dict | None = None):
        super().__init__(
            model_type="standard",
            verbose=verbose,
            compile_kwargs=compile_kwargs,
        )
        self._hidden_states_model: Optional[csp.CmdStanModel] = None
        self._smoothed_hidden_states_model: Optional[csp.CmdStanModel] = None

    # TODO: check if I can use _stan_files_base_location to get the other filenames
    @property
    def _stan_files_base_location(self) -> str:
        """Return stan file base location inside stanbkt.stan_code."""
        return str(files("stanbkt").joinpath("stan_code", "BKT"))

    @property
    def _stan_model_filename(self) -> str:
        return str(files("stanbkt").joinpath("stan_code", "BKT", "BKT_model.stan"))

    @property
    def _stan_hidden_filename(self) -> str:
        return str(
            files("stanbkt").joinpath("stan_code", "BKT", "hidden_states.stan")
        )

    @property
    def _stan_smoothed_hidden_filename(self) -> str:
        return str(
            files("stanbkt").joinpath(
                "stan_code", "BKT", "smoothed_hidden_states.stan"
            )
        )

    def fit(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
        method: str = "sample",
        **kwargs,
    ) -> "StandardBKT":
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
        method : str, default='sample'
            Method for fitting the model. Options are 'sample' for MCMC sampling,
            'vb' for variational inference, and 'optimize' for MAP estimation.
        **kwargs
            Additional keyword arguments to pass to the underlying CmdStanModel fitting method.

        Returns
        -------
        StandardBKT
            The fitted StandardBKT model instance.

        Raises
        ------
        ValueError
            If data validation fails or invalid method specified.
        """
        if method != "sample":
            raise ValueError(
                "Only method='sample' is currently implemented for StandardBKT."
            )

        if self._stan_model is None:
            self._stan_model = csp.CmdStanModel(
                stan_file=self._stan_model_filename,
                cpp_options=self.stan_compile_kwargs,
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
        self.is_fitted = True
        self._previous_fit_method = method
        return self

    def _build_stan_data_dict(self, correctness: np.ndarray) -> dict[str, Any]:
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

            summary_df = summarize_state_predictions(gq_fit.draws_pd().T).reset_index()
            summary_df = summary_df.rename(columns={"index": "variable"})
            summary_df[kc_column_name] = kc_id
            result_frames.append(summary_df)

        if not result_frames:
            return pd.DataFrame(
                columns=[kc_column_name, "variable", "mean", "std", "median", "2.5%", "97.5%"]
            )

        return pd.concat(result_frames, ignore_index=True)

    def predict(
        self,
        data: pd.DataFrame,
        posterior=True,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        r"""
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

    def predict_smoothed_states(
        self,
        data: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Generate smoothed hidden-state summaries as one long DataFrame across KCs."""
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

    def _extract_predictions(self) -> npt.NDArray:
        raise NotImplementedError(
            "Prediction extraction is not implemented for StandardBKT yet"
        )

    def _extract_prediction_std(self) -> npt.NDArray:
        raise NotImplementedError(
            "Prediction std extraction is not implemented for StandardBKT yet"
        )

