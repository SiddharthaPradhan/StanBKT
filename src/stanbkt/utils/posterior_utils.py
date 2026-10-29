"""Utilities for summarising posterior prediction draw DataFrames."""

from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Optional, Sequence, Union, Mapping, cast

import cmdstanpy as csp
import numpy as np
import pandas as pd
from numba import njit, prange

from stanbkt.utils.numba_utils import PARALLEL_SAFE
from stanbkt.utils.data_utils import (
    ColumnNames,
    KCData,
    _PKNOW,
    _PCORRECT,
    constant_categorical,
    iter_kc_data,
    smallest_int_dtype,
    tile_categorical,
)

_STAT_LABELS = ("mean", "std", "median")


# parallel only when the threading layer tolerates concurrent launches from KC worker threads
@njit(parallel=PARALLEL_SAFE, cache=True)
def _compute_posterior_stats(
    arr: np.ndarray,  # (n_obs, n_draws), float64,
    quantile_fracs: np.ndarray,  # (n_q,), float64
) -> tuple:
    """Compute mean, std, median, and quantiles over draws for each observation in parallel."""
    n_obs, n_draws = arr.shape
    n_q = len(quantile_fracs)

    means = np.empty(n_obs, np.float64)
    stds = np.empty(n_obs, np.float64)
    medians = np.empty(n_obs, np.float64)
    quants = np.empty((n_obs, n_q), np.float64)

    for i in prange(n_obs):  # ty:ignore[not-iterable]
        buf = arr[i].copy()

        # mean
        s = 0.0
        for j in range(n_draws):
            s += buf[j]
        mean_i = s / n_draws
        means[i] = mean_i

        # std (ddof=1)
        ss = 0.0
        for j in range(n_draws):
            d = buf[j] - mean_i
            ss += d * d
        stds[i] = np.sqrt(ss / (n_draws - 1)) if n_draws > 1 else 0.0

        # sort once for median + quantiles
        buf.sort()

        # median
        half = n_draws >> 1
        if n_draws & 1:
            medians[i] = buf[half]
        else:
            medians[i] = 0.5 * (buf[half - 1] + buf[half])

        # quantiles
        for qi in range(n_q):
            virtual = quantile_fracs[qi] * (n_draws - 1)
            lo = int(virtual)
            hi = lo + 1
            if hi >= n_draws:
                hi = n_draws - 1
            frac = virtual - lo
            quants[i, qi] = buf[lo] + frac * (buf[hi] - buf[lo])

    return means, stds, medians, quants


def _quantile_label(q: float) -> str:
    return f"{q * 100:.2f}%"


def _observation_index_arrays(
    kc_data: KCData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Student IDs, problem IDs, correctness and position within the student of every observation.

    Observations are ordered student by student in interaction order, the same order as the
    flat generated quantities written by the Stan programs.
    """
    lengths = np.asarray(kc_data.lengths, dtype=np.int64)
    interactions = kc_data.student_inter_dict
    student_ids = np.repeat(np.asarray(list(interactions.keys()), dtype=object), lengths)
    problem_ids = np.asarray(
        list(chain.from_iterable(inter.problem_ids for inter in interactions.values())),
        dtype=object,
    )
    valid_mask = np.arange(kc_data.correctness.shape[1])[np.newaxis, :] < lengths[:, np.newaxis]
    correctness = np.asarray(kc_data.correctness[valid_mask], dtype=np.int8)
    order = np.arange(len(student_ids), dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    return student_ids, problem_ids, correctness, order


def _flat_gq_slice(col_names: Sequence[str], name: str, n_obs: int) -> Optional[slice]:
    """Columns of the flat ``name[1]..name[n_obs]`` generated quantity, None when it is absent."""
    col_names = tuple(col_names)
    try:
        start = col_names.index(f"{name}[1]")
    except ValueError:
        if f"{name}[1,1]" in col_names:
            raise ValueError(
                f"Generated quantity '{name}' uses the padded (student, problem) layout of an older "
                "StanBKT version. Regenerate the Stan output with this version."
            ) from None
        return None
    stop = start + n_obs
    if (
        stop > len(col_names)
        or col_names[stop - 1] != f"{name}[{n_obs}]"
        or (stop < len(col_names) and col_names[stop] == f"{name}[{n_obs + 1}]")
    ):
        raise ValueError(
            f"Generated quantity '{name}' does not have one value per observation ({n_obs}). "
            "The Stan output does not match the data passed with it."
        )
    return slice(start, stop)


def _summary_frame_from_arrays(
    kc_id_str: str,
    obs_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pknow_arr: np.ndarray,  # (n_obs, n_draws)
    pcorr_arr: Optional[np.ndarray],  # (n_obs, n_draws)
    col_mapping: dict[str, str],
    quantiles: list[float],
) -> pd.DataFrame:
    """Per-observation summary statistics of one KC, sorted by student then interaction order."""
    obs_student_ids, obs_problem_ids, obs_correctness, obs_order = obs_arrays
    q_arr = np.asarray(quantiles, dtype=np.float64)
    result: dict[str, Any] = {
        col_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID): np.repeat(kc_id_str, len(obs_student_ids)),
        col_mapping[ColumnNames.STUDENT_ID]: pd.Categorical(obs_student_ids),
        col_mapping[ColumnNames.PROBLEM_ID]: pd.Categorical(obs_problem_ids),
        col_mapping[ColumnNames.CORRECTNESS]: obs_correctness,
        "_order": obs_order,
    }
    for name, arr in ((_PKNOW, pknow_arr), (_PCORRECT, pcorr_arr)):
        if arr is None:
            continue
        means, stds, medians, quants = _compute_posterior_stats(
            np.ascontiguousarray(arr, dtype=np.float64), q_arr
        )
        result[f"{name}_mean"] = means
        result[f"{name}_std"] = stds
        result[f"{name}_median"] = medians
        for qi, q in enumerate(quantiles):
            result[f"{name}_{_quantile_label(q)}"] = quants[:, qi]

    kc_summary = pd.DataFrame(result)
    kc_summary.sort_values(by=[col_mapping[ColumnNames.STUDENT_ID], "_order"], inplace=True)
    kc_summary.drop(columns=["_order"], inplace=True)
    return kc_summary


def _process_single_kc_gq(
    kc_id_str: str,
    gq_kc: Any,
    kc_data: KCData,
    col_mapping: dict,
) -> pd.DataFrame:
    """Convert a single KC's CmdStanGQ output to a long-form draw DataFrame.

    Parameters
    ----------
    kc_id_str : str
        KC identifier string.
    gq_kc : CmdStanGQ
        Raw generated-quantities fit for this KC.
    kc_data : KCData
        Preprocessed KC interaction data (used for student/problem ID remapping).
    col_mapping : dict
        Resolved column name mapping.

    Returns
    -------
    pd.DataFrame
        Draw-level DataFrame for this KC.
    """
    student_col = col_mapping[ColumnNames.STUDENT_ID]
    problem_col = col_mapping[ColumnNames.PROBLEM_ID]
    correctness_col = col_mapping[ColumnNames.CORRECTNESS]
    kc_col = col_mapping[ColumnNames.KC_ID]
    id_cols = ["chain__", "iter__", "draw__"]

    obs_student_ids, obs_problem_ids, obs_correctness, obs_order = _observation_index_arrays(
        kc_data
    )
    n_obs = len(obs_student_ids)
    col_names = gq_kc.column_names
    pknow_cols = _flat_gq_slice(col_names, _PKNOW, n_obs) if n_obs else None
    pcorr_cols = _flat_gq_slice(col_names, _PCORRECT, n_obs) if n_obs else None
    if pknow_cols is None:
        empty_cols = id_cols + [
            kc_col,
            student_col,
            problem_col,
            correctness_col,
            "_order",
            _PKNOW,
        ]
        if pcorr_cols is not None:
            empty_cols.append(_PCORRECT)
        return pd.DataFrame(columns=empty_cols)

    # chains are stacked one after another, as in CmdStanGQ.draws_pd
    raw_draws = gq_kc.draws(concat_chains=True)
    n_draws = raw_draws.shape[0]
    n_chains = max(1, int(getattr(gq_kc, "chains", 1)))
    per_chain = n_draws // n_chains
    id_values = {
        "chain__": np.repeat(np.arange(1, n_chains + 1), per_chain),
        "iter__": np.tile(np.arange(1, per_chain + 1), n_chains),
        "draw__": np.arange(1, n_draws + 1),
    }

    result: dict[str, Any] = {}
    for col, col_values in id_values.items():
        result[col] = np.repeat(
            col_values.astype(smallest_int_dtype(int(col_values.max()))), n_obs
        )
    result[kc_col] = constant_categorical(kc_id_str, n_draws * n_obs)
    result[student_col] = tile_categorical(obs_student_ids, n_draws)
    result[problem_col] = tile_categorical(obs_problem_ids, n_draws)
    result[correctness_col] = np.tile(obs_correctness, n_draws)
    result["_order"] = np.tile(
        obs_order.astype(smallest_int_dtype(int(obs_order.max()))), n_draws
    )
    # rows are draw major, each draw holds every observation
    result[_PKNOW] = raw_draws[:, pknow_cols].ravel()
    if pcorr_cols is not None:
        result[_PCORRECT] = raw_draws[:, pcorr_cols].ravel()

    return pd.DataFrame(result, copy=False)


def gq_to_draws(
    stan_output: dict[str, csp.CmdStanGQ],
    data: pd.DataFrame,
    col_mapping: Optional[
        Union[
            Mapping[ColumnNames, str],
            Mapping[str, str],
            Mapping[ColumnNames | str, str],
        ]
    ] = None,
    print_fn: Optional[Callable[..., None]] = None,
) -> dict[str, pd.DataFrame]:
    """Convert raw CmdStanGQ outputs into long-form draw DataFrames with remapped IDs.

    Parameters
    ----------
    stan_output : dict[str, CmdStanGQ]
        Mapping from KC ID to raw CmdStanGQ objects, as returned by
        ``predict_posterior_stan`` or ``predict_smoothed_posterior_stan``.
    data : pd.DataFrame
        The original student interaction data used for the Stan GQ call.
        Required to remap Stan integer indices back to actual student/problem IDs.
    col_mapping : dict, optional
        Column name mapping.  If ``None``, the standard ``ColumnNames`` defaults
        are used.
    print_fn : callable, optional
        Optional logging callable.  Receives a message string as the first
        positional argument.  Defaults to ``None`` (no logging).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from KC ID to draw-level DataFrames suitable for
        ``posterior_summary``.
    """
    col_mapping = ColumnNames.apply_default_mapping(col_mapping)

    posterior_draws: dict[str, pd.DataFrame] = {}
    for kc_id, kc_data in iter_kc_data(
        data=data,
        col_mapping=col_mapping,
        return_groups=False,
        print_fn=print_fn,
    ):
        kc_id_str = str(kc_id)
        gq_kc = stan_output.get(kc_id_str)
        if gq_kc is None:
            if print_fn is not None:
                print_fn(
                    f"No generated quantities found for KC '{kc_id_str}' when post processing predictions."
                )
            continue

        posterior_draws[kc_id_str] = _process_single_kc_gq(
            kc_id_str, gq_kc, kc_data, col_mapping
        )
    return posterior_draws


def _summarize_single_kc_draws(
    kc: str,
    kc_df: pd.DataFrame,
    col_mapping: dict[str, str],
    quantiles: list[float],
) -> pd.DataFrame:
    """Summarise one KC's draw-level DataFrame into per-observation statistics."""
    if kc_df.empty:
        return pd.DataFrame()

    param_id_cols = [
        col_mapping.get(ColumnNames.STUDENT_ID),
        col_mapping.get(ColumnNames.PROBLEM_ID),
        col_mapping.get(ColumnNames.CORRECTNESS),
    ]
    agg_cols = [c for c in [_PKNOW, _PCORRECT] if c in kc_df.columns]
    has_order_col = "_order" in kc_df.columns
    groupby_cols = param_id_cols + (["_order"] if has_order_col else [])
    grouped = kc_df.groupby(groupby_cols, sort=False, observed=True)[agg_cols]
    # built in aggregations, a python callable per quantile would run once per group
    stats = grouped.agg(list(_STAT_LABELS))
    quants = grouped.quantile(quantiles).unstack(level=-1).reindex(stats.index)
    columns: dict[str, Any] = {}
    for col in agg_cols:
        for stat in _STAT_LABELS:
            columns[f"{col}_{stat}"] = stats[(col, stat)].to_numpy()
        for q in quantiles:
            columns[f"{col}_{_quantile_label(q)}"] = quants[(col, q)].to_numpy()
    kc_summary = pd.DataFrame(columns, index=stats.index).reset_index()
    student_col_name = col_mapping.get(ColumnNames.STUDENT_ID)
    sort_cols = [student_col_name] + (["_order"] if has_order_col else [])
    kc_summary.sort_values(by=sort_cols, inplace=True)
    if has_order_col:
        kc_summary.drop(columns=["_order"], inplace=True)
    kc_summary.insert(0, col_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID), str(kc))
    return kc_summary


def _summarize_single_kc_gq(
    kc_id_str: str,
    gq_kc: Any,
    kc_data: KCData,
    col_mapping: dict[str, str],
    quantiles: list[float],
) -> pd.DataFrame:
    """Summarise single KC's CmdStanGQ output directly from raw numpy draws.

    Bypasses pandas DataFrame construction by slicing the flat ``pKnow`` and
    ``pCorrectness`` columns out of ``draws(concat_chains=True)``.
    Stats (mean, std, median, quantiles) are computed by a Numba parallel kernel.
    """
    obs_arrays = _observation_index_arrays(kc_data)
    n_obs = len(obs_arrays[0])
    if n_obs == 0:
        return pd.DataFrame()
    col_names = gq_kc.column_names
    pknow_cols = _flat_gq_slice(col_names, _PKNOW, n_obs)
    if pknow_cols is None:
        return pd.DataFrame()
    pcorr_cols = _flat_gq_slice(col_names, _PCORRECT, n_obs)

    raw_draws = gq_kc.draws(concat_chains=True)
    # transposed to (n_obs, n_draws) for cache-friendly Numba
    return _summary_frame_from_arrays(
        kc_id_str,
        obs_arrays,
        raw_draws[:, pknow_cols].T,
        raw_draws[:, pcorr_cols].T if pcorr_cols is not None else None,
        col_mapping,
        quantiles,
    )


def posterior_summary(
    draws: Union[dict[str, pd.DataFrame], dict[str, csp.CmdStanGQ]],
    col_mapping: Optional[
        Union[
            Mapping[ColumnNames, str],
            Mapping[str, str],
            Mapping[ColumnNames | str, str],
        ]
    ] = None,
    quantiles: list[float] = [0.025, 0.975],
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Summarise posterior predictions into per-observation statistics.

    Parameters
    ----------
    draws : dict[str, pd.DataFrame] or dict[str, CmdStanGQ]
        Either draw-level DataFrames (as returned by ``predict_posterior_draws``
        or ``predict_smoothed_posterior_draws``), or raw CmdStanGQ objects (as
        returned by ``predict_posterior_stan`` or
        ``predict_smoothed_posterior_stan``).  When passing CmdStanGQ objects,
        ``data`` must also be supplied.
    col_mapping : dict, optional
        Column name mapping.  If ``None``, the standard ``ColumnNames``
        defaults are used.
    quantiles : list[float], default [0.025, 0.975]
        Credible-interval quantiles to include in the summary.  Each value
        must be in ``[0, 1]``.
    data : pd.DataFrame, optional
        Original student interaction data.  Required when ``draws`` contains
        CmdStanGQ objects; ignored otherwise.

    Returns
    -------
    pd.DataFrame
        Long-form summary with mean, std, median, and the requested quantiles
        for pKnow and pCorrectness.
    """

    col_mapping = ColumnNames.apply_default_mapping(col_mapping)

    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")

    stat_labels = list(_STAT_LABELS) + [_quantile_label(q) for q in quantiles]
    summary_cols = [f"{_PKNOW}_{stat}" for stat in stat_labels]
    summary_cols += [f"{_PCORRECT}_{stat}" for stat in stat_labels]

    param_id_cols = [
        col_mapping.get(ColumnNames.STUDENT_ID),
        col_mapping.get(ColumnNames.PROBLEM_ID),
        col_mapping.get(ColumnNames.CORRECTNESS),
    ]

    if not draws:
        return pd.DataFrame(columns=[col_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID)] + param_id_cols + summary_cols)

    # get first value to check if draws are CmdStanGQ objects or already-processed DataFrames
    first_val = next(iter(draws.values()))

    if not isinstance(first_val, pd.DataFrame):
        if data is None:
            raise ValueError(
                "'data' must be provided when 'draws' contains CmdStanGQ objects. "
                "Pass the student interaction data used to generate the Stan output."
            )
        gq_draws = cast(dict[str, csp.CmdStanGQ], draws)
        result_frames: list[pd.DataFrame] = []
        for kc_id, kc_data in iter_kc_data(
            data=data,
            col_mapping=col_mapping,
            return_groups=False,
            print_fn=None,
        ):
            kc_id_str = str(kc_id)
            gq_kc = gq_draws.get(kc_id_str)
            if gq_kc is None:
                continue

            kc_summary = _summarize_single_kc_gq(
                kc_id_str,
                gq_kc,
                kc_data,
                col_mapping,
                quantiles,
            )
            if not kc_summary.empty:
                result_frames.append(kc_summary)

        if not result_frames:
            return pd.DataFrame(columns=[col_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID)] + param_id_cols + summary_cols)
        return pd.concat(result_frames, ignore_index=True)
    else:
        draw_frames = cast(dict[str, pd.DataFrame], draws)

    result_frames: list[pd.DataFrame] = []
    for kc, kc_df in draw_frames.items():
        kc_summary = _summarize_single_kc_draws(str(kc), kc_df, col_mapping, quantiles)
        if not kc_summary.empty:
            result_frames.append(kc_summary)

    if not result_frames:
        return pd.DataFrame(columns=[col_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID)] + param_id_cols + summary_cols)

    return pd.concat(result_frames, ignore_index=True)
