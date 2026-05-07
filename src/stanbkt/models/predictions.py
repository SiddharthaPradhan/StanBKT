"""
This module handles posterior prediction and exposes a public wrapper function that can be reused by model methods.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Optional, Union, overload

import cmdstanpy as csp
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

from stanbkt.fits.fit_types import CmdStanFit
from stanbkt.utils.compilation import compile_stan_model
from stanbkt.utils.data_utils import ColumnNames, KCData, iter_kc_data
from stanbkt.utils.posterior_utils import posterior_summary

if TYPE_CHECKING:
    from stanbkt.models.core.base import BKTModelBase


_ColumnMappingInput = Optional[
    Union[
        Mapping[ColumnNames, str],
        Mapping[str, str],
        Mapping[ColumnNames | str, str],
    ]
]


def _prepare_posterior_prediction_inputs(
    model: "BKTModelBase",
    data: pd.DataFrame,
    column_mapping: _ColumnMappingInput = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    resolved_mapping = ColumnNames.apply_default_mapping(column_mapping)
    kc_column_name = resolved_mapping[ColumnNames.KC_ID]
    model.check_data_contains_fitted_kcs(set(data[kc_column_name].astype(str).unique()))
    overlapping_kcs = model.get_kcs_in_fitted_kcs(set(data[kc_column_name].unique()))
    data_cp = data.copy().loc[data[kc_column_name].isin(overlapping_kcs)]
    return data_cp, resolved_mapping


def _get_or_compile_gq_model(model: "BKTModelBase", smoothed: bool) -> csp.CmdStanModel:
    attr_name = "_smoothed_hidden_states_model" if smoothed else "_hidden_states_model"
    model_filename = (
        model._stan_smoothed_hidden_filename
        if smoothed
        else model._stan_hidden_filename
    )

    gq_model = getattr(model, attr_name)
    if gq_model is None:
        gq_model = compile_stan_model(
            model_filename,
            cpp_options=model.cpp_compile_kwargs,
            stanc_options=model.stan_compile_kwargs,
            print_fn=model.log,
        )
        setattr(model, attr_name, gq_model)

    return gq_model


def _extract_param_draw_matrix(
    model: "BKTModelBase",
    fit: CmdStanFit,
    param_name: str,
    n_students: int,
    groups: Optional[npt.NDArray[np.int32]],
) -> npt.NDArray[np.float64]:
    stan_variable_fn = getattr(fit, "stan_variable", None)
    if callable(stan_variable_fn):
        values = np.asarray(stan_variable_fn(param_name), dtype=np.float64)
    else:
        values = np.asarray(
            model._extract_param_draws(fit, param_name), dtype=np.float64
        )

    if values.ndim == 0:
        values = values.reshape(1, 1)

    if groups is None:
        if values.ndim == 1:
            return np.broadcast_to(
                values.reshape(-1, 1), (values.shape[0], n_students)
            ).copy()
        if values.ndim == 2 and values.shape[1] == 1:
            return np.broadcast_to(values, (values.shape[0], n_students)).copy()
        if values.ndim == 2:
            return np.broadcast_to(values[:, :1], (values.shape[0], n_students)).copy()
        raise ValueError(
            f"Unexpected parameter draw shape for '{param_name}': {values.shape}."
        )

    n_groups = int(np.max(groups))
    group_indices = groups.astype(np.int64) - 1
    if np.any(group_indices < 0):
        raise ValueError("Group indices must be 1-based positive integers.")

    if values.ndim == 1:
        if values.shape[0] == n_groups:
            group_draws = values.reshape(1, n_groups)
        else:
            group_draws = np.broadcast_to(
                values.reshape(-1, 1), (values.shape[0], n_groups)
            ).copy()
    elif values.ndim == 2:
        if values.shape[1] == n_groups:
            group_draws = values
        elif values.shape[1] == 1:
            group_draws = np.broadcast_to(values, (values.shape[0], n_groups)).copy()
        else:
            raise ValueError(
                f"Cannot map parameter '{param_name}' draws with shape {values.shape} to {n_groups} groups."
            )
    else:
        raise ValueError(
            f"Unexpected parameter draw shape for '{param_name}': {values.shape}."
        )

    return group_draws[:, group_indices]


def _flatten_student_matrix_by_lengths(
    matrix: npt.NDArray[np.float64], lengths: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    total_obs = int(np.sum(lengths))
    flat = np.empty(total_obs, dtype=np.float64)
    cursor = 0
    for student_index, interaction_len in enumerate(lengths):
        interaction_len_int = int(interaction_len)
        if interaction_len_int <= 0:
            continue
        next_cursor = cursor + interaction_len_int
        flat[cursor:next_cursor] = matrix[student_index, :interaction_len_int]
        cursor = next_cursor
    return flat


def _build_observation_index_arrays(
    kc_data: KCData,
) -> tuple[
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    npt.NDArray[np.int8],
    npt.NDArray[np.int64],
]:
    student_ids: list[object] = []
    problem_ids: list[object] = []
    correctness_vals: list[int] = []
    order_vals: list[int] = []

    for student_index, (student_id, interaction) in enumerate(
        kc_data.student_inter_dict.items()
    ):
        interaction_len = int(interaction.length)
        for problem_index in range(interaction_len):
            student_ids.append(str(student_id))
            problem_ids.append(str(interaction.problem_ids[problem_index]))
            correctness_vals.append(
                int(kc_data.correctness[student_index, problem_index])
            )
            order_vals.append(problem_index)

    return (
        np.asarray(student_ids, dtype=object),
        np.asarray(problem_ids, dtype=object),
        np.asarray(correctness_vals, dtype=np.int8),
        np.asarray(order_vals, dtype=np.int64),
    )


def _predict_posterior_draws_numba(
    model: "BKTModelBase",
    data: pd.DataFrame,
    column_mapping: dict[str, str],
    *,
    smoothed: bool,
) -> dict[str, pd.DataFrame]:
    state_predictor = (
        type(model)._predict_hidden_states_smoothed_numba
        if smoothed
        else type(model)._predict_hidden_states_numba
    )
    njit_predictor = njit(fastmath=True, parallel=False, cache=True)(state_predictor)

    kc_col = column_mapping[ColumnNames.KC_ID]
    student_col = column_mapping[ColumnNames.STUDENT_ID]
    problem_col = column_mapping[ColumnNames.PROBLEM_ID]
    correctness_col = column_mapping[ColumnNames.CORRECTNESS]

    posterior_draws: dict[str, pd.DataFrame] = {}

    for kc_id, kc_data in iter_kc_data(
        data=data,
        col_mapping=column_mapping,
        return_groups=model._use_groups,
        print_fn=model.log,
    ):
        kc_id_str = str(kc_id)
        kc_data = model._align_kc_group_indices_with_fit_metadata(kc_id_str, kc_data)
        kc_fit = model.fits.get_fit(kc_id_str)
        if kc_fit is None:
            continue

        n_students = int(kc_data.correctness.shape[0])
        groups = kc_data.groups if model._use_groups else None

        prior_draws = _extract_param_draw_matrix(
            model, kc_fit, "pi_know", n_students, groups
        )
        learn_draws = _extract_param_draw_matrix(
            model, kc_fit, "learn", n_students, groups
        )
        forget_draws = _extract_param_draw_matrix(
            model, kc_fit, "forget", n_students, groups
        )
        guess_draws = _extract_param_draw_matrix(
            model, kc_fit, "guess", n_students, groups
        )
        slip_draws = _extract_param_draw_matrix(
            model, kc_fit, "slip", n_students, groups
        )

        n_draws = prior_draws.shape[0]
        draw_sizes = {
            n_draws,
            learn_draws.shape[0],
            forget_draws.shape[0],
            guess_draws.shape[0],
            slip_draws.shape[0],
        }
        if len(draw_sizes) != 1:
            raise ValueError(
                f"Inconsistent posterior draw counts for KC '{kc_id_str}': {sorted(draw_sizes)}"
            )

        student_ids, problem_ids, correctness_vals, order_vals = (
            _build_observation_index_arrays(kc_data)
        )
        total_obs = len(student_ids)
        if total_obs == 0:
            posterior_draws[kc_id_str] = pd.DataFrame(
                columns=[
                    "draw__",
                    kc_col,
                    student_col,
                    problem_col,
                    correctness_col,
                    "_order",
                    "pKnow",
                    "pCorrectness",
                ]
            )
            continue

        pknow_all = np.empty(n_draws * total_obs, dtype=np.float64)
        pcorr_all = np.empty(n_draws * total_obs, dtype=np.float64)
        lengths = kc_data.lengths.astype(np.int64)

        for draw_index in range(n_draws):
            p_know, p_correct = njit_predictor(
                correctness=kc_data.correctness,
                prior=prior_draws[draw_index],
                learn=learn_draws[draw_index],
                forget=forget_draws[draw_index],
                guess=guess_draws[draw_index],
                slip=slip_draws[draw_index],
                lengths=lengths,
            )
            start = draw_index * total_obs
            end = start + total_obs
            pknow_all[start:end] = _flatten_student_matrix_by_lengths(p_know, lengths)
            pcorr_all[start:end] = _flatten_student_matrix_by_lengths(
                p_correct, lengths
            )

        kc_df = pd.DataFrame(
            {
                "draw__": np.repeat(
                    np.arange(1, n_draws + 1, dtype=np.int64), total_obs
                ),
                kc_col: np.repeat(kc_id_str, n_draws * total_obs),
                student_col: np.tile(student_ids, n_draws),
                problem_col: np.tile(problem_ids, n_draws),
                correctness_col: np.tile(correctness_vals, n_draws),
                "_order": np.tile(order_vals, n_draws),
                "pKnow": pknow_all,
                "pCorrectness": pcorr_all,
            }
        )
        posterior_draws[kc_id_str] = kc_df

    return posterior_draws


@overload
def predict_posterior(
    model: "BKTModelBase",
    data: pd.DataFrame,
    column_mapping: _ColumnMappingInput = None,
    *,
    smoothed: bool = False,
    backend: Literal["stan", "numba"] = "stan",
    output: Literal["stan"],
    quantiles: Optional[list[float]] = None,
    stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
    n_cores: int = 1,
) -> dict[str, csp.CmdStanGQ]: ...


@overload
def predict_posterior(
    model: "BKTModelBase",
    data: pd.DataFrame,
    column_mapping: _ColumnMappingInput = None,
    *,
    smoothed: bool = False,
    backend: Literal["stan", "numba"] = "stan",
    output: Literal["draws"] = "draws",
    quantiles: Optional[list[float]] = None,
    stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
    n_cores: int = 1,
) -> dict[str, pd.DataFrame]: ...


@overload
def predict_posterior(
    model: "BKTModelBase",
    data: pd.DataFrame,
    column_mapping: _ColumnMappingInput = None,
    *,
    smoothed: bool = False,
    backend: Literal["stan", "numba"] = "stan",
    output: Literal["summary"],
    quantiles: Optional[list[float]] = None,
    stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
    n_cores: int = 1,
) -> pd.DataFrame: ...


def predict_posterior(
    model: "BKTModelBase",
    data: pd.DataFrame,
    column_mapping: _ColumnMappingInput = None,
    *,
    smoothed: bool = False,
    backend: Literal["stan", "numba"] = "stan",
    output: Literal["stan", "draws", "summary"] = "draws",
    quantiles: Optional[list[float]] = None,
    stan_output: Optional[dict[str, csp.CmdStanGQ]] = None,
    n_cores: int = 1,
) -> dict[str, csp.CmdStanGQ] | dict[str, pd.DataFrame] | pd.DataFrame:
    """Public wrapper for posterior prediction workflows.

    Parameters
    ----------
    model : BKTModelBase
        Fitted model instance.
    data : pd.DataFrame
        Interaction data.
    column_mapping : dict, optional
        Mapping from expected column names to source data names.
    smoothed : bool, default=False
        Whether to use smoothed posterior hidden-state prediction models.
    backend : {"stan", "numba"}, default="stan"
        Prediction backend. ``stan`` uses generated quantities. ``numba`` runs
        deterministic hidden-state recursion for each posterior parameter draw.
    output : {"stan", "draws", "summary"}, default="draws"
        Desired output type.
    quantiles : list[float], optional
        Quantiles used for summary output.
    stan_output : dict[str, CmdStanGQ], optional
        Precomputed Stan generated quantities output.
    n_cores : int, default=1
        Number of cores used for summary processing.
    """
    if output == "stan":
        referrer = (
            "predict_smoothed_posterior_stan" if smoothed else "predict_posterior_stan"
        )
        model._fit_check(referrer=referrer)
    elif output == "summary":
        referrer = (
            "predict_smoothed_posterior_summary"
            if smoothed
            else "predict_posterior_summary"
        )
        model._fit_check(referrer=referrer)

    data_cp, resolved_mapping = _prepare_posterior_prediction_inputs(
        model=model,
        data=data,
        column_mapping=column_mapping,
    )

    if backend == "numba" and output == "stan":
        raise ValueError(
            "'backend=numba' does not support output='stan'. Use output='draws' or 'summary'."
        )

    if output == "stan":
        gq_model = _get_or_compile_gq_model(model, smoothed=smoothed)
        return model._predict_generated_quantities(
            data=data_cp,
            gq_model=gq_model,
            column_mapping=resolved_mapping,
        )

    if output == "draws":
        if backend == "numba":
            if stan_output is not None:
                raise ValueError(
                    "'stan_output' cannot be provided when backend='numba'."
                )
            return _predict_posterior_draws_numba(
                model=model,
                data=data_cp,
                column_mapping=resolved_mapping,
                smoothed=smoothed,
            )

        resolved_stan_output = stan_output
        if resolved_stan_output is None:
            gq_model = _get_or_compile_gq_model(model, smoothed=smoothed)
            resolved_stan_output = model._predict_generated_quantities(
                data=data_cp,
                gq_model=gq_model,
                column_mapping=resolved_mapping,
            )
        return model._process_predict_gq(
            resolved_stan_output, data_cp, resolved_mapping
        )

    resolved_quantiles = quantiles if quantiles is not None else [0.025, 0.975]
    if not all(0 <= q <= 1 for q in resolved_quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")
    resolved_n_cores = model._resolve_n_cores(n_cores)

    if backend == "numba":
        if stan_output is not None:
            raise ValueError("'stan_output' cannot be provided when backend='numba'.")
        draws = _predict_posterior_draws_numba(
            model=model,
            data=data_cp,
            column_mapping=resolved_mapping,
            smoothed=smoothed,
        )
        return posterior_summary(
            draws,
            col_mapping=resolved_mapping,
            quantiles=resolved_quantiles,
        )

    if stan_output is not None:
        return model._process_predict_summary_gq(
            posterior_summary_raw=stan_output,
            data=data_cp,
            col_mapping=resolved_mapping,
            quantiles=resolved_quantiles,
            n_cores=resolved_n_cores,
        )

    gq_model = _get_or_compile_gq_model(model, smoothed=smoothed)
    return model._predict_summary_streaming(
        data=data_cp,
        gq_model=gq_model,
        column_mapping=resolved_mapping,
        quantiles=resolved_quantiles,
        n_cores=resolved_n_cores,
    )
