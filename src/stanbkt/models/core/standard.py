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
    individual_initial_knowledge : bool, default False
        Whether to estimate individualized initial knowledge parameters for each student (True) or a single population-level initial knowledge parameter (False).
    init_knowledge_strategy : InitKnowledgeStrategy, default=InitKnowledgeStrategy.CORRECTNESS_ONLY
        Strategy for estimating initial knowledge when `individual_initial_knowledge` is True. This determines how the initial knowledge parameters are informed by the data:
        - InitKnowledgeStrategy.CORRECTNESS_ONLY: Initial knowledge is informed solely by the correctness of the first interaction for each student.
        - InitKnowledgeStrategy.JOINT: Initial knowledge is informed by both the correctness
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
        # TODO: this depends on init_knowledge_strategy
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
            "individual_pi_know": int(self.individual_initial_knowledge),
        }
        if priors is not None:
            raw_priors = priors.to_dict(self.init_knowledge_strategy)
            for param in ("pi_know", "learn", "forget", "guess", "slip"):
                mu_key = f"{param}_mu"
                std_key = f"{param}_std"
                mu_value = raw_priors.get(mu_key)
                std_value = raw_priors.get(std_key)

                if mu_value is None or std_value is None:
                    # Stan still requires prior arrays even when the model uses
                    # the corresponding uniform-prior flag to ignore them.
                    data_dict[f"prior_{mu_key}"] = [0.0]
                    data_dict[f"prior_{std_key}"] = [1.0]
                    data_dict[f"unif_prior_{param}"] = 1
                elif np.isscalar(mu_value) and np.isscalar(std_value):
                    data_dict[f"prior_{mu_key}"] = [mu_value]
                    data_dict[f"prior_{std_key}"] = [std_value]
                    data_dict[f"unif_prior_{param}"] = 0
                else:
                    raise ValueError(
                        f"Unsupported prior value type for parameter '{param}': "
                        f"mu={type(mu_value).__name__}, std={type(std_value).__name__}. "
                        "Expected scalar values or None."
                    )
        return data_dict

    def _extract_bkt_params_from_fit(
        self,
        fit,
        n_students: int,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        groups: Optional[np.ndarray] = None,
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
        groups : np.ndarray, optional
            Group indices (unused for StandardBKT, only for MultiBKT).

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


class StandardBKTTest(StandardBKT):
    """StandardBKT variant that uses the test Stan model file."""

    def __init__(
        self,
        fit_method: FitMethod = FitMethod.MCMC,
        individual_initial_knowledge: bool = False,
        init_knowledge_strategy: InitKnowledgeStrategy = InitKnowledgeStrategy.CORRECTNESS_ONLY,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: dict | None = None,
        cpp_compile_kwargs: dict | None = None,
        multi_init_stu: bool = False,
        multi_trans_stu: bool = False,
        multi_emis_stu: bool = False,
        multi_trans_prob: bool = False,
        multi_emis_prob: bool = False,
    ):
        super().__init__(
            fit_method=fit_method,
            individual_initial_knowledge=individual_initial_knowledge,
            init_knowledge_strategy=init_knowledge_strategy,
            verbose=verbose,
            stan_compile_kwargs=stan_compile_kwargs,
            cpp_compile_kwargs=cpp_compile_kwargs,
        )
        self.multi_init_stu = multi_init_stu
        self.multi_trans_stu = multi_trans_stu
        self.multi_emis_stu = multi_emis_stu
        self.multi_trans_prob = multi_trans_prob
        self.multi_emis_prob = multi_emis_prob

    @property
    def _stan_model_filename(self) -> str:
        return str(files("stanbkt").joinpath("stan_code", "BKT", "BKT_model_test.stan"))

    def _build_stan_data_dict(
        self, kc_data: KCData, priors: Optional[PriorsBase] = None
    ) -> dict[str, Any]:
        correctness = kc_data.correctness
        n_students, n_problems = correctness.shape

        if kc_data.problem_sequence is not None and kc_data.problem_sequence.shape == (
            n_students,
            n_problems,
        ):
            problem_sequence = kc_data.problem_sequence.astype(np.int32, copy=False)
        else:
            problem_sequence = np.tile(
                np.arange(1, n_problems + 1, dtype=np.int32), (n_students, 1)
            )

        def _resolve_student_groups(
            values: Optional[npt.NDArray[np.int32]],
            enabled: bool,
        ) -> npt.NDArray[np.int32]:
            if enabled:
                if values is None:
                    raise ValueError(
                        "Role-specific student grouping requested but no data provided. "
                        "Ensure the data column is populated."
                    )
                return values.astype(np.int32, copy=False)
            return np.ones(n_students, dtype=np.int32)

        def _resolve_problem_groups(
            values: Optional[npt.NDArray[np.int32]],
            enabled: bool,
        ) -> npt.NDArray[np.int32]:
            if enabled:
                if values is None:
                    raise ValueError(
                        "Role-specific problem grouping requested but no data provided. "
                        "Ensure the data column is populated."
                    )
                return values.astype(np.int32, copy=False)
            return np.ones(max(1, n_problems), dtype=np.int32)

        student_groups_init = _resolve_student_groups(
            kc_data.student_groups_init,
            self.multi_init_stu,
        )
        student_groups_transition = _resolve_student_groups(
            kc_data.student_groups_transition,
            self.multi_trans_stu,
        )
        student_groups_emission = _resolve_student_groups(
            kc_data.student_groups_emission,
            self.multi_emis_stu,
        )
        problem_groups_transition = _resolve_problem_groups(
            kc_data.problem_groups_transition,
            self.multi_trans_prob,
        )
        problem_groups_emission = _resolve_problem_groups(
            kc_data.problem_groups_emission,
            self.multi_emis_prob,
        )

        n_student_groups_init = int(np.max(student_groups_init))
        n_student_groups_transition = int(np.max(student_groups_transition))
        n_student_groups_emission = int(np.max(student_groups_emission))
        n_problem_groups_transition = int(np.max(problem_groups_transition))
        n_problem_groups_emission = int(np.max(problem_groups_emission))

        data_dict: dict[str, Any] = {
            "nProblems": int(n_problems),
            "nStudents": int(n_students),
            "nStudentGroupsInit": n_student_groups_init,
            "nStudentGroupsTransition": n_student_groups_transition,
            "nStudentGroupsEmission": n_student_groups_emission,
            "nProblemGroupsTransition": n_problem_groups_transition,
            "nProblemGroupsEmission": n_problem_groups_emission,
            "studentGroupsInit": student_groups_init,
            "studentGroupsTransition": student_groups_transition,
            "studentGroupsEmission": student_groups_emission,
            "problemGroupsTransition": problem_groups_transition,
            "problemGroupsEmission": problem_groups_emission,
            "correctness": correctness,
            "problem_sequence": problem_sequence,
            "interaction_lengths": kc_data.lengths,
        }

        if priors is None:
            priors = self._default_priors()

        raw_priors = priors.to_dict(self.init_knowledge_strategy)

        def _to_scalar(x: Any) -> float:
            arr = np.asarray(x, dtype=np.float64).ravel()
            if arr.size == 0:
                raise ValueError("Prior value cannot be empty.")
            return float(arr[0])

        def _vector_prior(param: str, size: int) -> tuple[list[float], list[float], int]:
            mu_value = raw_priors.get(f"{param}_mu")
            std_value = raw_priors.get(f"{param}_std")
            if mu_value is None or std_value is None:
                return [0.0] * size, [1.0] * size, 1
            mu_scalar = _to_scalar(mu_value)
            std_scalar = _to_scalar(std_value)
            return [mu_scalar] * size, [std_scalar] * size, 0

        def _matrix_prior(
            param: str, n_rows: int, n_cols: int
        ) -> tuple[list[list[float]], list[list[float]], int]:
            mu_value = raw_priors.get(f"{param}_mu")
            std_value = raw_priors.get(f"{param}_std")
            if mu_value is None or std_value is None:
                return (
                    [[0.0] * n_cols for _ in range(n_rows)],
                    [[1.0] * n_cols for _ in range(n_rows)],
                    1,
                )
            mu_scalar = _to_scalar(mu_value)
            std_scalar = _to_scalar(std_value)
            return (
                [[mu_scalar] * n_cols for _ in range(n_rows)],
                [[std_scalar] * n_cols for _ in range(n_rows)],
                0,
            )

        pi_mu, pi_std, pi_unif = _vector_prior("pi_know", n_student_groups_init)
        learn_mu, learn_std, learn_unif = _matrix_prior(
            "learn", n_student_groups_transition, n_problem_groups_transition
        )
        forget_mu, forget_std, forget_unif = _matrix_prior(
            "forget", n_student_groups_transition, n_problem_groups_transition
        )
        guess_mu, guess_std, guess_unif = _matrix_prior(
            "guess", n_student_groups_emission, n_problem_groups_emission
        )
        slip_mu, slip_std, slip_unif = _matrix_prior(
            "slip", n_student_groups_emission, n_problem_groups_emission
        )

        data_dict.update(
            {
                "prior_pi_know_mu": pi_mu,
                "prior_pi_know_std": pi_std,
                "prior_learn_mu": learn_mu,
                "prior_learn_std": learn_std,
                "prior_forget_mu": forget_mu,
                "prior_forget_std": forget_std,
                "prior_guess_mu": guess_mu,
                "prior_guess_std": guess_std,
                "prior_slip_mu": slip_mu,
                "prior_slip_std": slip_std,
                "unif_prior_pi_know": pi_unif,
                "unif_prior_learn": learn_unif,
                "unif_prior_forget": forget_unif,
                "unif_prior_guess": guess_unif,
                "unif_prior_slip": slip_unif,
            }
        )

        return data_dict
