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
        self._use_groups = True
        # For MultiBKT, we use student grouping for transitions (the traditional grouping)
        self.multi_trans_stu = True

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
            Preprocessed KC data. Must have ``student_groups_transition``
            populated (i.e., produced with ``multi_trans_stu=True``).
        priors : PriorsBase, optional
            Per-group prior specifications.  If ``None``, default priors are used.

        Returns
        -------
        dict[str, Any]
            Data dict ready to pass to CmdStanPy.
        """
        if kc_data.student_groups_transition is None:
            raise ValueError(
                "KCData must have student_groups_transition populated for MultiBKT. "
                "Ensure the data contains a group column and that "
                "fit() / predict() are called correctly."
            )

        correctness = kc_data.correctness
        n_students, n_problems = correctness.shape
        groups = kc_data.student_groups_transition
        n_groups: int = int(np.max(groups))

        data_dict: dict[str, Any] = {
            "nStudents": int(n_students),
            "nProblems": int(n_problems),
            "correctness": correctness,
            "interaction_lengths": kc_data.lengths,
            "nGroups": n_groups,
            "groups": groups,
            "individual_pi_know": int(self.individual_initial_knowledge),
        }

        if priors is None:
            priors = MultiPriors(use_defaults=True)

        raw_priors = priors.to_dict(self.init_knowledge_strategy)
        # Expand scalar priors to per-group lists
        expanded_priors = MultiPriors._expand_grouped_priors(
            raw_priors, n_groups=n_groups
        )

        for param in ("pi_know", "learn", "forget", "guess", "slip"):
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

        return (
            _to_student_array("pi_know"),
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


class MultiBKTTest(MultiBKT):
    """MultiBKT variant that uses the test Stan model file."""

    def __init__(
        self,
        fit_method: FitMethod = FitMethod.MCMC,
        verbose: VerbosityLevel = VerbosityLevel.INFO,
        stan_compile_kwargs: dict | None = None,
        cpp_compile_kwargs: dict | None = None,
        multi_init_stu: bool = True,
        multi_trans_stu: bool = True,
        multi_emis_stu: bool = True,
        multi_trans_prob: bool = True,
        multi_emis_prob: bool = True,
    ):
        super().__init__(
            fit_method=fit_method,
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

        def _expand_vector(value: Any, size: int) -> list[float]:
            if value is None:
                return []
            arr = np.asarray(value, dtype=np.float64).ravel()
            if arr.size == 0:
                return []
            if arr.size == 1:
                return [float(arr[0])] * size
            if arr.size >= size:
                return [float(v) for v in arr[:size]]
            out = [float(v) for v in arr]
            out.extend([float(arr[-1])] * (size - arr.size))
            return out

        def _vector_prior(param: str, size: int) -> tuple[list[float], list[float], int]:
            mu_vec = _expand_vector(raw_priors.get(f"{param}_mu"), size)
            std_vec = _expand_vector(raw_priors.get(f"{param}_std"), size)
            if len(mu_vec) == 0 or len(std_vec) == 0:
                return [0.0] * size, [1.0] * size, 1
            return mu_vec, std_vec, 0

        def _matrix_prior(
            param: str, n_rows: int, n_cols: int
        ) -> tuple[list[list[float]], list[list[float]], int]:
            mu_rows = _expand_vector(raw_priors.get(f"{param}_mu"), n_rows)
            std_rows = _expand_vector(raw_priors.get(f"{param}_std"), n_rows)
            if len(mu_rows) == 0 or len(std_rows) == 0:
                return (
                    [[0.0] * n_cols for _ in range(n_rows)],
                    [[1.0] * n_cols for _ in range(n_rows)],
                    1,
                )
            return (
                [[mu_rows[row_idx]] * n_cols for row_idx in range(n_rows)],
                [[std_rows[row_idx]] * n_cols for row_idx in range(n_rows)],
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
