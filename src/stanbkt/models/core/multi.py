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

        data_dict: dict[str, Any] = {
            "nStudents": int(n_students),
            "nProblems": int(n_problems),
            "correctness": correctness,
            "interaction_lengths": kc_data.lengths,
            "nGroups": n_groups,
            "groups": kc_data.groups,
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

    def fit(
        self,
        data: pd.DataFrame,
        priors: Optional[dict[str, PriorsBase] | PriorsBase] = None,
        column_mapping: Optional[Mapping[str, str]] = None,
        stan_fit_options: Optional[Union[StanFitOptions, dict[str, Any]]] = None,
        overwrite_kcs: bool = False,
    ) -> MultiBKT:
        """Fit the grouped BKT model to data.

        Each KC is fitted independently.  Students are grouped according to the
        ``group_id`` column; each group receives its own set of BKT parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format interaction data.  Must include ``student_id``,
            ``problem_id``, ``correct``, and ``group_id`` columns (or their
            mapped equivalents).
        column_mapping : dict, optional
            Mapping to non-default column names.
        stan_fit_options : StanFitOptions or dict, optional
            Arguments forwarded to the CmdStanPy fitting method.
        overwrite_kcs : bool, default False
            Whether to overwrite existing fits when re-fitting a KC.

        Returns
        -------
        MultiBKT
            The fitted model instance.

        Raises
        ------
        ValueError
            If data validation fails or ``group_id`` column is absent.
        """
        if self.fits is None:
            self.fits = self.fit_class()

        if self._stan_model is None:
            self._compile_model(self._stan_model_filename)

        if self._stan_model is None:
            raise RuntimeError(
                "Stan model compilation failed. The model object is None after "
                "compilation. Ensure the Stan source file is valid and that "
                "CmdStanPy is correctly installed."
            )

        if stan_fit_options is None:
            stan_fit_options = FitFactory.create_default_fit_options(self._fit_method)
        else:
            if isinstance(stan_fit_options, dict):
                stan_fit_options = FitFactory.create_fit_options_from_dict(
                    stan_fit_options, self._fit_method
                )
            FitFactory.verify_fit_options_compatibility(
                stan_fit_options, self._fit_method
            )

        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=column_mapping,
            return_groups=True,  # groups required for MultiBKT
            print_fn=self.log,
        ):
            if self.fits.has_kc(str(kc_id)):
                if not overwrite_kcs:
                    raise ValueError(
                        f"Fit for KC '{kc_id}' already exists. "
                        "Set 'overwrite_kcs=True' to overwrite."
                    )

            self.log(f"Fitting KC: {kc_id}", level=VerbosityLevel.DEBUG)

            data_dict = self._build_stan_data_dict(kc_data)
            fit_result = self._fit_stan_model_using_method(
                data_dict=data_dict, fit_options=stan_fit_options
            )
            self.fits.add_fit(str(kc_id), fit_result, overwrite_kcs=overwrite_kcs)

        self._is_fitted = True
        return self

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[Mapping[str, str]] = None,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        parallel: bool = True,
        fast_math: bool = True,
    ) -> pd.DataFrame:
        """Predict hidden states using group-specific point-estimate parameters.

        Overrides the base-class implementation to pass per-student group
        indices when extracting BKT parameters from the fit, ensuring each
        student uses their group's parameter estimates.

        Parameters
        ----------
        data : pd.DataFrame
            Interaction data.  Must include a ``group_id`` column so that each
            student can be assigned to a group.
        column_mapping : dict, optional
            Mapping to non-default column names.
        point_estimate : Literal["mean", "median", "mode"], default "mean"
            Statistic to use when collapsing posterior draws.
        parallel : bool, default True
            Numba parallelism flag.
        fast_math : bool, default True
            Numba fast-math flag.

        Returns
        -------
        pd.DataFrame
            Long-format predictions with columns ``kc_id``, ``student_id``,
            ``problem_id``, ``pKnow``, ``pCorrectness``, and ``correct``.
        """
        self._fit_check(referrer="predict")

        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError("'point_estimate' must be 'mean', 'median', or 'mode'.")

        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]

        working_data = data
        if kc_column_name not in working_data.columns:
            working_data = working_data.copy()
            working_data[kc_column_name] = _DEFAULT_KC_ID

        self.check_data_contains_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(
            set(working_data[kc_column_name].unique())
        )
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model "
                "has been successfully fitted before calling prediction methods."
            )

        if filtered_data.empty:
            return pd.DataFrame(
                columns=[
                    ColumnNames.KC_ID,
                    ColumnNames.STUDENT_ID,
                    ColumnNames.PROBLEM_ID,
                    "pKnow",
                    "pCorrectness",
                    ColumnNames.CORRECTNESS,
                ]
            )

        njit_predict_numba: Callable = njit(
            fastmath=fast_math, parallel=parallel, cache=True
        )(type(self)._predict_hidden_states_numba)

        predictions: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=filtered_data,
            col_mapping=column_mapping,
            return_groups=True,
            print_fn=self.log,
        ):
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
                groups=kc_data.groups,
            )
            p_know, p_correctness = njit_predict_numba(
                correctness=kc_data.correctness,
                prior=prior,
                learn=learn,
                forget=forget,
                guess=guess,
                slip=slip,
                lengths=kc_data.lengths,
            )
            kc_predictions = self._state_arrays_to_long_df(
                p_know=p_know,
                kc_data=kc_data,
                p_correctness=p_correctness,
                correctness_col_name=column_mapping.get(
                    ColumnNames.CORRECTNESS, ColumnNames.CORRECTNESS
                ),
            )
            kc_predictions.insert(0, ColumnNames.KC_ID, str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
            return pd.DataFrame(
                columns=[
                    ColumnNames.KC_ID,
                    ColumnNames.STUDENT_ID,
                    ColumnNames.PROBLEM_ID,
                    "pKnow",
                    "pCorrectness",
                    ColumnNames.CORRECTNESS,
                ]
            )

        return pd.concat(predictions, ignore_index=True)

    def predict_smoothed_states(
        self,
        data: Optional[pd.DataFrame] = None,
        column_mapping: Optional[Mapping[str, str]] = None,
        point_estimate: Literal["mean", "median", "mode"] = "mean",
        parallel: bool = True,
        fast_math: bool = True,
    ) -> pd.DataFrame:
        """Predict smoothed hidden states using group-specific point-estimate parameters.

        Overrides the base-class implementation to pass per-student group
        indices when extracting BKT parameters from the fit.

        Parameters
        ----------
        data : pd.DataFrame
            Interaction data.  Must include a ``group_id`` column.
        column_mapping : dict, optional
            Mapping to non-default column names.
        point_estimate : Literal["mean", "median", "mode"], default "mean"
            Statistic to use when collapsing posterior draws.
        parallel : bool, default True
            Numba parallelism flag.
        fast_math : bool, default True
            Numba fast-math flag.

        Returns
        -------
        pd.DataFrame
            Long-format smoothed hidden-state estimates.
        """
        self._fit_check(referrer="predict_smoothed_states")

        if data is None:
            raise ValueError("'data' must be provided for point-estimate prediction.")

        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError("'point_estimate' must be 'mean', 'median', or 'mode'.")

        column_mapping = ColumnNames.apply_default_mapping(column_mapping)
        kc_column_name = column_mapping[ColumnNames.KC_ID]

        working_data = data
        if kc_column_name not in working_data.columns:
            working_data = working_data.copy()
            working_data[kc_column_name] = _DEFAULT_KC_ID

        self.check_data_contains_fitted_kcs(
            set(working_data[kc_column_name].astype(str).unique())
        )
        overlapping_kcs = self.get_kcs_in_fitted_kcs(
            set(working_data[kc_column_name].unique())
        )
        filtered_data = working_data.loc[
            working_data[kc_column_name].isin(overlapping_kcs)
        ].copy()

        if self.fits is None:
            raise RuntimeError(
                "The fits container has not been initialized. Ensure the model "
                "has been successfully fitted before calling prediction methods."
            )

        if filtered_data.empty:
            return pd.DataFrame(
                columns=[
                    ColumnNames.KC_ID,
                    ColumnNames.STUDENT_ID,
                    ColumnNames.PROBLEM_ID,
                    "pKnow",
                    "pCorrectness",
                    ColumnNames.CORRECTNESS,
                ]
            )

        njit_predict_smoothed_numba: Callable = njit(
            fastmath=fast_math, parallel=parallel, cache=True
        )(type(self)._predict_hidden_states_smoothed_numba)

        predictions: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=filtered_data,
            col_mapping=column_mapping,
            return_groups=True,
            print_fn=self.log,
        ):
            kc_fit = self.fits.get_fit(kc_id)
            prior, learn, forget, guess, slip = self._extract_bkt_params_from_fit(
                kc_fit,
                n_students=kc_data.correctness.shape[0],
                point_estimate=point_estimate,
                groups=kc_data.groups,
            )
            p_smooth = njit_predict_smoothed_numba(
                correctness=kc_data.correctness,
                prior=prior,
                learn=learn,
                forget=forget,
                guess=guess,
                slip=slip,
                lengths=kc_data.lengths,
            )
            kc_predictions = self._state_arrays_to_long_df(
                p_smooth,
                kc_data,
                correctness_col_name=column_mapping.get(
                    ColumnNames.CORRECTNESS, ColumnNames.CORRECTNESS
                ),
            )
            kc_predictions.insert(0, ColumnNames.KC_ID, str(kc_id))
            predictions.append(kc_predictions)

        if not predictions:
            return pd.DataFrame(
                columns=[
                    ColumnNames.KC_ID,
                    ColumnNames.STUDENT_ID,
                    ColumnNames.PROBLEM_ID,
                    "pKnow",
                    "pCorrectness",
                    ColumnNames.CORRECTNESS,
                ]
            )

        return pd.concat(predictions, ignore_index=True)

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
