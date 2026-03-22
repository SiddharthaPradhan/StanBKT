from __future__ import annotations
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_factory import FitFactory
from importlib.resources import files
from typing import Any, Optional, Union, Literal
import numpy as np
import numpy.typing as npt
import pandas as pd
from stanbkt.models.core.base import BKTModelBase, FitMethod
from stanbkt.utils.data_utils import (
    iter_kc_data,
    KCData,
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
                "Stan model compilation failed. The model object is None after compilation. "
                "Ensure the Stan source file is valid and that CmdStanPy is correctly installed."
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
            if self.fits.has_kc(str(kc_id)):
                raise ValueError(
                    f"Fit for KC '{kc_id}' already exists. Set 'overwrite_kcs=True' to overwrite."
                )

            self._print(f"Fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)

            data_dict = self._build_stan_data_dict(kc_data)
            fit_result = self._fit_stan_model_using_method(
                data_dict=data_dict, fit_options=stan_fit_options
            )
            self.fits.add_fit(str(kc_id), fit_result)

        self._is_fitted = True
        return self

    def _build_stan_data_dict(self, kc_data: KCData) -> dict[str, Any]:
        """Build data dictionary for the stan models.

        Parameters
        ----------
        kc_data : KCData
            KCData object containing the correctness matrix and student interaction information for a single knowledge component.

        Returns
        -------
        dict[str, Any]
            Dictionary containing data for the Stan model.
        """
        correctness = kc_data.correctness
        n_students, n_problems = correctness.shape

        return {
            "nStudents": int(n_students),
            "nProblems": int(n_problems),
            "correctness": correctness,
            "interaction_lengths": kc_data.lengths,
            "nGroups": 1,
            "groups": np.ones(n_students, dtype=np.int32),
        }

    def _extract_bkt_params_from_fit(
        self,
        fit,
        n_students: int,
        point_estimate: Literal["mean", "median"] = "mean",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prior = StandardBKT._extract_param_point_estimate(
            fit, "pi_know", point_estimate
        )
        learn = StandardBKT._extract_param_point_estimate(fit, "learn", point_estimate)
        forget = StandardBKT._extract_param_point_estimate(
            fit, "forget", point_estimate
        )
        guess = StandardBKT._extract_param_point_estimate(fit, "guess", point_estimate)
        slip = StandardBKT._extract_param_point_estimate(fit, "slip", point_estimate)

        return (
            np.full(n_students, prior, dtype=np.float64),
            np.full(n_students, learn, dtype=np.float64),
            np.full(n_students, forget, dtype=np.float64),
            np.full(n_students, guess, dtype=np.float64),
            np.full(n_students, slip, dtype=np.float64),
        )

    def evaluate(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            "'evaluate' is not yet implemented for StandardBKT. "
            "This method will be available in a future release."
        )
