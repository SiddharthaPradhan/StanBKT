from __future__ import annotations
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_factory import FitFactory
from stanbkt.models.priors import PriorsBase, StandardPriors
from stanbkt.models.model_types import InitKnowledgeStrategy
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
        individual_initial_knowledge: bool = False,
        init_knowledge_strategy: InitKnowledgeStrategy = InitKnowledgeStrategy.CORRECTNESS_ONLY,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: dict | None = None,
        cpp_compile_kwargs: dict | None = None,
    ):
        """Initialize a StandardBKT model instance.

        Attributes
        ----------
        fit_method : FitMethod
            The method used for fitting the model (e.g., MCMC, VB, MAP).
        individual_initial_knowledge : bool
            Whether to initial know states are individualized to the student. If False,
            a single initial knowledge parameter is estimated for all students.
        initital_knowledge_strategy : InitKnowledgeStrategy
            Strategy for estimating initial knowledge. This is only applicable if `individual_initial_knowledge` is True.
            When  this is set to `CORRECTNESS_ONLY`, only the correctness data is used to estimate initial knowledge.
            When `JOINT`, model requires student level covariate (e.g. pretest) to jointly estimate initial knowledge (uses correctness and covariate).
            For example, `CORRECTNESS_ONLY` uses only the correctness of the first interaction for each student to inform
            initial knowledge estimates, while `FIRST_INTERACTION` uses the correctness of the first interaction with each KC for each student.
        verbose : VerbosityLevel
            Verbosity level for logging.
        stan_compile_kwargs : dict
            Additional keyword arguments for Stan model compilation.
        cpp_compile_kwargs : dict
            Additional keyword arguments for C++ compilation of the Stan model.
        fits : Optional[BaseFit]
            Object to store fitted model results for each KC.
        """

        super().__init__(
            verbose=verbose,
            fit_method=fit_method,
            individual_initial_knowledge=individual_initial_knowledge,
            init_knowledge_strategy=init_knowledge_strategy,
            stan_compile_kwargs=stan_compile_kwargs,
            cpp_compile_kwargs=cpp_compile_kwargs,
        )

    @property
    def _stan_model_filename(self) -> str:
        """Get path to the main Stan model file.

        Returns
        -------
        str
            Path to BKT_model.stan.
        """
        return str(files("stanbkt").joinpath("stan_code", "BKT", "BKT_model.stan"))

    @property
    def _stan_hidden_filename(self) -> str:
        """Get path to the Stan hidden states model file.

        Returns
        -------
        str
            Path to hidden_states.stan for generating hidden state estimates.
        """
        return str(files("stanbkt").joinpath("stan_code", "BKT", "hidden_states.stan"))

    @property
    def _stan_smoothed_hidden_filename(self) -> str:
        """Get path to the Stan smoothed hidden states model file.

        Returns
        -------
        str
            Path to smoothed_hidden_states.stan for generating smoothed state estimates.
        """
        return str(
            files("stanbkt").joinpath("stan_code", "BKT", "smoothed_hidden_states.stan")
        )

    def _default_priors(self):
        return StandardPriors(use_defaults=True)

    def _default_priors_class(self):
        return StandardPriors

    def _build_stan_data_dict(
        self, kc_data: KCData, priors: Optional[PriorsBase] = None
    ) -> dict[str, Any]:
        """Build data dictionary for the stan models.

        Parameters
        ----------
        kc_data : KCData
            KCData object containing the correctness matrix and student interaction information for a single knowledge component.
        priors : PriorsBase, optional
            PriorsBase object containing the prior specifications for the model parameters for this KC.
            If None, priors are not added to the return dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing data for the Stan model.
        """
        correctness = kc_data.correctness
        n_students, n_problems = correctness.shape

        data_dict = {
            "nStudents": int(n_students),
            "nProblems": int(n_problems),
            "correctness": correctness,
            "interaction_lengths": kc_data.lengths,
            "nGroups": 1,
            "groups": np.ones(n_students, dtype=np.int32),
        }
        if priors is not None:
            for prior_param_key, value in priors.to_dict(
                self.init_knowledge_strategy
            ).items():
                base_key = prior_param_key.removesuffix("_mu").removesuffix("_std")
                if isinstance(value, list):
                    data_dict["prior_" + prior_param_key] = value
                    data_dict["unif_prior_" + base_key] = 0
                elif isinstance(value, type(None)):
                    data_dict["unif_prior_" + base_key] = 1
                elif np.isscalar(value):  # scalar but stan expects array
                    data_dict["prior_" + prior_param_key] = [value]
                    data_dict["unif_prior_" + base_key] = 0
                else:
                    raise ValueError(
                        f"Unsupported prior value type for parameter '{prior_param_key}': {type(value)}. "
                        "Expected list, or scalar."
                    )

            data_dict.update(priors.to_dict(self.init_knowledge_strategy))
        return data_dict

    def _extract_bkt_params_from_fit(
        self,
        fit,
        n_students: int,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract BKT parameters from a fitted Stan model.

        Extracts point estimates for each BKT parameter (`prior` knowledge, learn,
        forget, guess, slip) and expands them to match the student dimension.

        Parameters
        ----------
        fit : CmdStanFit
            Fitted Stan model object.
        n_students : int
            Number of students (used to broadcast scalar parameters to array form).
        point_estimate : Literal["mean", "median", "mode"], default "mean"
            Which point estimate to use (posterior mean or median).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple of (prior_know, learn, forget, guess, slip) arrays, each of shape (n_students,).
        """
        init_know = StandardBKT._extract_param_point_estimate(
            fit, "pi_know", point_estimate
        )
        learn = StandardBKT._extract_param_point_estimate(fit, "learn", point_estimate)
        forget = StandardBKT._extract_param_point_estimate(
            fit, "forget", point_estimate
        )
        guess = StandardBKT._extract_param_point_estimate(fit, "guess", point_estimate)
        slip = StandardBKT._extract_param_point_estimate(fit, "slip", point_estimate)

        return (
            np.full(n_students, init_know, dtype=np.float64),
            np.full(n_students, learn, dtype=np.float64),
            np.full(n_students, forget, dtype=np.float64),
            np.full(n_students, guess, dtype=np.float64),
            np.full(n_students, slip, dtype=np.float64),
        )

    def evaluate(self, **kwargs) -> dict[str, Any]:
        """Evaluate model performance (not yet implemented).

        This method will provide evaluation metrics for the fitted model
        in a future release.

        Returns
        -------
        dict[str, Any]
            Evaluation results (implementation pending).

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError(
            "'evaluate' is not yet implemented for StandardBKT. "
            "This method will be available in a future release."
        )
