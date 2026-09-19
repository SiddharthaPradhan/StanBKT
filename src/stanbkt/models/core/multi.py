from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

from stanbkt.fits.fit_factory import FitFactory
from stanbkt.fits.fit_options import StanFitOptions
from stanbkt.fits.fit_types import CmdStanFit
from stanbkt.models.core.base import BKTModelBase, FitMethod
from stanbkt.models.model_types import InitKnowledgeStrategy
from stanbkt.models.priors import MultiPriors, PriorsBase
from stanbkt.utils.data_utils import (
    ColumnNames,
    KCData,
    _DEFAULT_KC_ID,
    iter_kc_data,
)
from stanbkt.utils.verbose import VerbosityLevel


def _is_all_none(val: Any) -> bool:
    """Return True if val is None or a list where every element is None."""
    if val is None:
        return True
    if isinstance(val, list):
        return all(v is None for v in val)
    return False


class MultiBKT(BKTModelBase):
    """Grouped Bayesian Knowledge Tracing model.

    Extends the standard BKT model to allow group-specific parameters.
    Each student is assigned to a group via a ``group_id`` column in the
    data, and each group receives its own BKT parameters
    (``pi_know``, ``learn``, ``forget``, ``guess``, ``slip``).

    The same Stan model (``BKT_model.stan``) is reused. ``StandardBKT``
    collapses it to a single group; ``MultiBKT`` lets it run with the
    full group structure present in the data.

    Parameters
    ----------
    fit_method : FitMethod, default FitMethod.MCMC
        The method to use for fitting the Stan model.
    verbose : VerbosityLevel, default VerbosityLevel.INFO
        Verbosity level for logging.
    stan_compile_kwargs : dict | None, optional
        Additional Stan compilation options.
    cpp_compile_kwargs : dict | None, optional
        Additional C++ compilation options.
    individual_initial_knowledge : bool, default False
        Whether to estimate individualized initial knowledge parameters for each student (True) or a single population-level initial knowledge parameter (False).
    init_knowledge_strategy : InitKnowledgeStrategy, default=InitKnowledgeStrategy.CORRECTNESS_ONLY
        Strategy for estimating initial knowledge when `individual_initial_knowledge` is True. This determines how the initial knowledge parameters are informed by the data:
        - InitKnowledgeStrategy.CORRECTNESS_ONLY: Initial knowledge is informed solely by the correctness of the first interaction for each student.
        - InitKnowledgeStrategy.JOINT: Initial knowledge is informed by both the correctness and a supplied student-level covariate.
    low_memory : bool
        Whether to evict each KC's fit to disk after fitting, reloading it lazily on
        next access. Reduces peak memory usage when fitting many KCs at the cost of
        some performance.
    """

    def __init__(
        self,
        fit_method: FitMethod = FitMethod.MCMC,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: dict | None = None,
        cpp_compile_kwargs: dict | None = None,
        individual_initial_knowledge: bool = False,
        init_knowledge_strategy: InitKnowledgeStrategy = InitKnowledgeStrategy.CORRECTNESS_ONLY,
        low_memory: bool = False,
    ):
        super().__init__(
            verbose=verbose,
            fit_method=fit_method,
            individual_initial_knowledge=individual_initial_knowledge,
            init_knowledge_strategy=init_knowledge_strategy,
            stan_compile_kwargs=stan_compile_kwargs,
            cpp_compile_kwargs=cpp_compile_kwargs,
            low_memory=low_memory,
        )
        self._use_groups = True

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

    def _default_priors(self) -> MultiPriors:
        return MultiPriors(use_defaults=True)

    def _default_priors_class(self) -> type[MultiPriors]:
        return MultiPriors

    def _build_stan_data_dict(
        self, kc_data: KCData, priors: Optional[PriorsBase] = None
    ) -> dict[str, Any]:
        """Build the data dictionary for the Stan grouped BKT model.

        Parameters
        ----------
        kc_data : KCData
            Preprocessed KC data.  Must have ``groups`` and ``group_2_index``
            populated (i.e. produced with ``return_groups=True``).
        priors : PriorsBase, optional
            Per-group prior specifications.  If ``None``, default priors are used.

        Returns
        -------
        dict[str, Any]
            Data dict ready to pass to CmdStanPy.
        """
        if kc_data.groups is None or kc_data.group_2_index is None:
            raise ValueError(
                "KCData must have groups populated for MultiBKT. "
                "Ensure the data contains a 'group_id' column and that "
                "fit() / predict() are called correctly."
            )

        correctness = kc_data.correctness
        n_students, n_problems = correctness.shape
        n_groups: int = len(kc_data.group_2_index)
        n_covariates = kc_data.covariates.shape[1] if kc_data.covariates is not None else 0
        is_joint = self.init_knowledge_strategy == InitKnowledgeStrategy.JOINT

        data_dict: dict[str, Any] = {
            "nStudents": int(n_students),
            "nProblems": int(n_problems),
            "correctness": correctness,
            "interaction_lengths": kc_data.lengths,
            "nGroups": n_groups,
            "groups": kc_data.groups,
            "individual_pi_know": int(self.individual_initial_knowledge),
            "joint_pi_know": int(is_joint),
            "nTrainStudents": int(n_students),
            "nCovariates": int(n_covariates),
            "train_student_idx": np.arange(1, n_students + 1, dtype=np.int32),
            "covariates": (
                kc_data.covariates
                if kc_data.covariates is not None
                else np.zeros((n_students, 0), dtype=np.float64)
            ),
        }

        if priors is None:
            priors = MultiPriors(use_defaults=True)

        raw_priors = priors.to_dict(self.init_knowledge_strategy)
        # Expand scalar priors to per-group lists (pi_b0/b1/sigma stay scalar/per-covariate)
        expanded_priors = MultiPriors._expand_grouped_priors(
            raw_priors, n_groups=n_groups
        )

        if is_joint:
            data_dict["prior_pi_know_mu"] = [0.0] * n_groups
            data_dict["prior_pi_know_std"] = [1.0] * n_groups
            data_dict["unif_prior_pi_know"] = 1

            b0_mu = raw_priors.get("pi_b0_know_mu")
            b0_std = raw_priors.get("pi_b0_know_std")
            data_dict["prior_pi_b0_know_mu"] = b0_mu if b0_mu is not None else 0.0
            data_dict["prior_pi_b0_know_std"] = b0_std if b0_std is not None else 5.0
            data_dict["unif_prior_pi_b0_know"] = int(b0_mu is None or b0_std is None)

            b1_mu, b1_std = PriorsBase._expand_covariate_priors(
                raw_priors.get("pi_b1_know_mu"),
                raw_priors.get("pi_b1_know_std"),
                n_covariates,
            )
            data_dict["prior_pi_b1_know_mu"] = (
                b1_mu if b1_mu is not None else [0.0] * n_covariates
            )
            data_dict["prior_pi_b1_know_std"] = (
                b1_std if b1_std is not None else [5.0] * n_covariates
            )
            data_dict["unif_prior_pi_b1_know"] = int(b1_mu is None or b1_std is None)

            sigma_lambda = raw_priors.get("pi_sigma_lambda")
            data_dict["prior_pi_sigma_lambda"] = (
                sigma_lambda if sigma_lambda is not None else 0.5
            )
            data_dict["unif_prior_pi_sigma"] = int(sigma_lambda is None)
        else:
            data_dict["prior_pi_b0_know_mu"] = 0.0
            data_dict["prior_pi_b0_know_std"] = 5.0
            data_dict["prior_pi_b1_know_mu"] = [0.0] * n_covariates
            data_dict["prior_pi_b1_know_std"] = [5.0] * n_covariates
            data_dict["prior_pi_sigma_lambda"] = 0.5
            data_dict["unif_prior_pi_b0_know"] = 1
            data_dict["unif_prior_pi_b1_know"] = 1
            data_dict["unif_prior_pi_sigma"] = 1

        for param in ("pi_know", "learn", "forget", "guess", "slip"):
            if is_joint and param == "pi_know":
                continue
            mu_key = f"{param}_mu"
            std_key = f"{param}_std"
            mu_val = expanded_priors.get(mu_key)
            std_val = expanded_priors.get(std_key)

            if _is_all_none(mu_val) or _is_all_none(std_val):
                # Non-informative uniform prior — provide dummy values
                # (Stan ignores prior_* when unif_prior_{param} == 1)
                data_dict[f"prior_{mu_key}"] = [0.0] * n_groups
                data_dict[f"prior_{std_key}"] = [1.0] * n_groups  # must be > 0
                data_dict[f"unif_prior_{param}"] = 1
            else:
                mu_list = mu_val if isinstance(mu_val, list) else [mu_val] * n_groups
                std_list = (
                    std_val if isinstance(std_val, list) else [std_val] * n_groups
                )
                data_dict[f"prior_{mu_key}"] = mu_list
                data_dict[f"prior_{std_key}"] = std_list
                data_dict[f"unif_prior_{param}"] = 0

        return data_dict

    @staticmethod
    def _extract_group_param_estimates(
        fit: CmdStanFit,
        param_name: str,
        n_groups: int,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
    ) -> npt.NDArray[np.float64]:
        """Extract per-group point estimates for a single BKT parameter.

        Parameters
        ----------
        fit : CmdStanFit
            Fitted Stan model.
        param_name : str
            Name of the Stan parameter (e.g. ``"pi_know"``).
        n_groups : int
            Number of groups.
        point_estimate : Literal["mean", "median", "mode"], default "mean"
            Statistic to compute across MCMC/VB draws.

        Returns
        -------
        np.ndarray
            Shape ``(n_groups,)`` array of point estimates.
        """
        stan_var_fn = getattr(fit, "stan_variable", None)
        if callable(stan_var_fn):
            arr = np.asarray(stan_var_fn(param_name), dtype=np.float64)
            if arr.ndim == 1:
                # MLE/MAP: single value per group
                return arr
            # MCMC/VB: shape (n_samples, n_groups) — reduce over samples axis
            if point_estimate == "mean":
                return np.mean(arr, axis=0)
            if point_estimate == "median":
                return np.median(arr, axis=0)
            return np.array(
                [BKTModelBase._modal_estimate(arr[:, i]) for i in range(arr.shape[1])],
                dtype=np.float64,
            )

        # Fallback via generic draw extractor (ravelled 1-D array)
        draws = BKTModelBase._extract_param_draws(fit, param_name)
        if draws.size % n_groups == 0:
            draws_2d = draws.reshape(-1, n_groups)
            if point_estimate == "mean":
                return np.mean(draws_2d, axis=0)
            if point_estimate == "median":
                return np.median(draws_2d, axis=0)
            return np.array(
                [
                    BKTModelBase._modal_estimate(draws_2d[:, i])
                    for i in range(draws_2d.shape[1])
                ],
                dtype=np.float64,
            )
        # Last resort: broadcast the global mean to all groups
        return np.full(n_groups, float(np.mean(draws)), dtype=np.float64)

    def _extract_bkt_params_from_fit(
        self,
        fit: CmdStanFit,
        n_students: int,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        groups: Optional[npt.NDArray[np.int32]] = None,
        kc_data: Optional[KCData] = None,
        kc_id: Optional[str] = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Extract per-student BKT parameter arrays from a grouped fit.

        Extracts group-level point estimates for each BKT parameter, then maps
        each student to their group's value using the ``groups`` index array.

        Parameters
        ----------
        fit : CmdStanFit
            Fitted Stan model.
        n_students : int
            Number of students.
        point_estimate : Literal["mean", "median", "mode"], default "mean"
            Statistic to compute across posterior draws.
        groups : np.ndarray, optional
            1-based group index per student, shape ``(n_students,)``.
            If ``None``, broadcasts the first group's value to all students.

        Returns
        -------
        tuple of np.ndarray
            ``(prior, learn, forget, guess, slip)``, each of shape
            ``(n_students,)``.
        """
        n_groups = int(np.max(groups)) if groups is not None else 1

        def _to_student_array(param_name: str) -> npt.NDArray[np.float64]:
            group_params = MultiBKT._extract_group_param_estimates(
                fit, param_name, n_groups, point_estimate
            )
            if groups is not None:
                return group_params[groups - 1].astype(np.float64)
            return np.full(n_students, float(group_params[0]), dtype=np.float64)

        if self.individual_initial_knowledge:
            init_know = self._extract_individual_pi_know_point_estimate(
                fit, kc_data=kc_data, kc_id=kc_id, point_estimate=point_estimate
            )
        else:
            init_know = _to_student_array("pi_know")

        return (
            init_know,
            _to_student_array("learn"),
            _to_student_array("forget"),
            _to_student_array("guess"),
            _to_student_array("slip"),
        )

    def evaluate(self, **kwargs) -> dict[str, Any]:
        """Evaluate model performance (not yet implemented).

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
            "'evaluate' is not yet implemented for MultiBKT. "
            "This method will be available in a future release."
        )
