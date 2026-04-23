"""Utilities for summarising posterior prediction draw DataFrames."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Union, Mapping, cast

import cmdstanpy as csp
import numpy as np
import pandas as pd
from numba import njit, prange

from stanbkt.utils.data_utils import (
    ColumnNames,
    KCData,
    _PKNOW,
    _PCORRECT,
    iter_kc_data,
)


_COL_PAT = re.compile(r"^([^\[]+)\[(\d+)\s*,\s*(\d+)\]$")


@njit(parallel=True, cache=True)
def _compute_posterior_stats(
    arr: np.ndarray,  # (n_obs, n_draws), float64, C-contiguous rows
    quantile_fracs: np.ndarray,  # (n_q,), float64
) -> tuple:
    """Compute mean, std, median, and quantiles over draws for each observation in parallel."""
    n_obs, n_draws = arr.shape
    n_q = len(quantile_fracs)

    means = np.empty(n_obs, np.float64)
    stds = np.empty(n_obs, np.float64)
    medians = np.empty(n_obs, np.float64)
    quants = np.empty((n_obs, n_q), np.float64)

    for i in prange(n_obs):
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

        # quantiles (linear interpolation, matches numpy's default method)
        for qi in range(n_q):
            virtual = quantile_fracs[qi] * (n_draws - 1)
            lo = int(virtual)
            hi = lo + 1
            if hi >= n_draws:
                hi = n_draws - 1
            frac = virtual - lo
            quants[i, qi] = buf[lo] + frac * (buf[hi] - buf[lo])

    return means, stds, medians, quants


def _process_single_kc_gq(
    kc_id_str: str,
    gq_kc_df: pd.DataFrame,
    kc_data: KCData,
    col_mapping: dict,
    id_cols: Optional[list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Convert a single KC's CmdStanGQ output to a long-form draw DataFrame.

    Parameters
    ----------
    kc_id_str : str
        KC identifier string.
    gq_kc_df : pd.DataFrame
        Raw generated-quantities fit for this KC.
    kc_data : KCData
        Preprocessed KC interaction data (used for student/problem ID remapping).
    col_mapping : dict
        Resolved column name mapping.
    id_cols : list[str] or None
        Meta-columns ending in ``__`` (chain__, iter__, draw__). Shared across KCs;
        pass ``None`` on the first call and reuse the returned list for subsequent KCs.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Draw-level DataFrame for this KC, and the detected ``id_cols``.
    """
    student_col = col_mapping[ColumnNames.STUDENT_ID]
    problem_col = col_mapping[ColumnNames.PROBLEM_ID]
    correctness_col = col_mapping[ColumnNames.CORRECTNESS]
    kc_col = col_mapping[ColumnNames.KC_ID]

    if id_cols is None:
        id_cols = [c for c in gq_kc_df.columns.astype(str) if c.endswith("__")]

    pknow_cols: dict[tuple[int, int], str] = {}
    pcorr_cols: dict[tuple[int, int], str] = {}
    for col in gq_kc_df.columns:
        m = _COL_PAT.match(str(col))
        if m:
            param, s, p = m.group(1), int(m.group(2)), int(m.group(3))
            if param == _PKNOW:
                pknow_cols[(s, p)] = col
            elif param == _PCORRECT:
                pcorr_cols[(s, p)] = col

    n_draws = len(gq_kc_df)

    obs_keys: list[tuple[int, int]] = []
    obs_order: list[int] = []
    obs_student_ids: list[str] = []
    obs_problem_ids: list[str] = []
    obs_correctness: list[int] = []
    pknow_obs_cols: list[str] = []

    for s_idx, (student_id, student_inter) in enumerate(
        kc_data.student_inter_dict.items(), start=1
    ):
        for p_idx, problem_id in enumerate(student_inter.problem_ids, start=1):
            key = (s_idx, p_idx)
            pknow_col = pknow_cols.get(key)
            if pknow_col is None:
                continue
            obs_keys.append(key)
            pknow_obs_cols.append(pknow_col)
            obs_order.append(p_idx - 1)
            obs_student_ids.append(student_id)
            obs_problem_ids.append(problem_id)
            obs_correctness.append(int(kc_data.correctness[s_idx - 1, p_idx - 1]))

    n_obs = len(obs_keys)
    if n_obs == 0:
        empty_cols = list(id_cols) + [
            kc_col,
            student_col,
            problem_col,
            correctness_col,
            "_order",
            _PKNOW,
        ]
        if pcorr_cols:
            empty_cols.append(_PCORRECT)
        return pd.DataFrame(columns=empty_cols), id_cols

    obs_student_ids_arr = np.asarray(obs_student_ids, dtype=object)
    obs_problem_ids_arr = np.asarray(obs_problem_ids, dtype=object)
    obs_correctness_arr = np.asarray(obs_correctness, dtype=np.int8)
    obs_order_arr = np.asarray(obs_order, dtype=np.int64)

    pknow_values = gq_kc_df[pknow_obs_cols].to_numpy().ravel()

    result: dict[str, Any] = {
        col: np.repeat(gq_kc_df[col].to_numpy(), n_obs) for col in id_cols
    }
    result[kc_col] = np.repeat(kc_id_str, n_draws * n_obs)
    result[student_col] = np.tile(obs_student_ids_arr, n_draws)
    result[problem_col] = np.tile(obs_problem_ids_arr, n_draws)
    result[correctness_col] = np.tile(obs_correctness_arr, n_draws)
    result["_order"] = np.tile(obs_order_arr, n_draws)
    result[_PKNOW] = pknow_values
    if pcorr_cols:
        result[_PCORRECT] = (
            gq_kc_df[[pcorr_cols[k] for k in obs_keys]].to_numpy().ravel()
        )

    return pd.DataFrame(result), id_cols


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
    id_cols: list[str] | None = None

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

        kc_df, id_cols = _process_single_kc_gq(
            kc_id_str, gq_kc.draws_pd(), kc_data, col_mapping, id_cols
        )
        posterior_draws[kc_id_str] = kc_df
    return posterior_draws


def _summarize_single_kc_draws(
    kc: str,
    kc_df: pd.DataFrame,
    col_mapping: dict[str, str],
    stat_fns: list[tuple[str, Any]],
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
    agg_kwargs = {
        f"{col}_{stat}": pd.NamedAgg(column=col, aggfunc=fn)
        for col in agg_cols
        for stat, fn in stat_fns
    }
    has_order_col = "_order" in kc_df.columns
    groupby_cols = param_id_cols + (["_order"] if has_order_col else [])
    kc_summary: pd.DataFrame = (
        kc_df.groupby(groupby_cols, sort=False).agg(**agg_kwargs).reset_index()
    )
    student_col_name = col_mapping.get(ColumnNames.STUDENT_ID)
    sort_cols = [student_col_name] + (["_order"] if has_order_col else [])
    kc_summary.sort_values(by=sort_cols, inplace=True)
    if has_order_col:
        kc_summary.drop(columns=["_order"], inplace=True)
    kc_summary.insert(0, "kc_id", str(kc))
    return kc_summary


def _summarize_single_kc_gq(
    kc_id_str: str,
    gq_kc: Any,  # duck-typed: needs .column_names and .draws(concat_chains=True)
    kc_data: KCData,
    col_mapping: dict[str, str],
    quantiles: list[float],
) -> pd.DataFrame:
    """Summarise single KC's CmdStanGQ output directly from raw numpy draws.

    Bypasses pandas DataFrame construction by using ``column_names`` for index
    parsing and ``draws(concat_chains=True)`` for the raw numpy array.
    Stats (mean, std, median, quantiles) are computed by a Numba parallel kernel.
    """
    student_col = col_mapping[ColumnNames.STUDENT_ID]
    problem_col = col_mapping[ColumnNames.PROBLEM_ID]
    correctness_col = col_mapping[ColumnNames.CORRECTNESS]

    # Parse column indices from the lightweight column_names tuple — no DataFrame needed
    col_names: tuple[str, ...] = gq_kc.column_names
    pknow_col_indices: dict[tuple[int, int], int] = {}
    pcorr_col_indices: dict[tuple[int, int], int] = {}
    for i, col in enumerate(col_names):
        m = _COL_PAT.match(col)
        if m:
            param, s, p = m.group(1), int(m.group(2)), int(m.group(3))
            if param == _PKNOW:
                pknow_col_indices[(s, p)] = i
            elif param == _PCORRECT:
                pcorr_col_indices[(s, p)] = i

    if not pknow_col_indices:
        return pd.DataFrame()

    # Build obs list from kc_data, collecting raw column indices in the same pass
    obs_keys: list[tuple[int, int]] = []
    obs_order: list[int] = []
    obs_student_ids: list[str] = []
    obs_problem_ids: list[str] = []
    obs_correctness: list[int] = []
    pknow_raw_col_indices: list[int] = []

    for s_idx, (student_id, student_inter) in enumerate(
        kc_data.student_inter_dict.items(), start=1
    ):
        for p_idx, problem_id in enumerate(student_inter.problem_ids, start=1):
            key = (s_idx, p_idx)
            ci = pknow_col_indices.get(key)
            if ci is None:
                continue
            obs_keys.append(key)
            pknow_raw_col_indices.append(ci)
            obs_order.append(p_idx - 1)
            obs_student_ids.append(student_id)
            obs_problem_ids.append(problem_id)
            obs_correctness.append(int(kc_data.correctness[s_idx - 1, p_idx - 1]))

    if not obs_keys:
        return pd.DataFrame()

    # Raw numpy draws: (n_draws, n_all_cols) — avoids pandas DataFrame construction
    raw_draws = gq_kc.draws(concat_chains=True)

    # Extract pKnow columns and transpose to (n_obs, n_draws) for cache-friendly Numba
    pknow_idx_arr = np.asarray(pknow_raw_col_indices, dtype=np.intp)
    pknow_arr = np.ascontiguousarray(
        raw_draws[:, pknow_idx_arr].T, dtype=np.float64
    )  # (n_obs, n_draws)

    q_arr = np.asarray(quantiles, dtype=np.float64)
    pknow_means, pknow_stds, pknow_medians, pknow_quants = _compute_posterior_stats(
        pknow_arr, q_arr
    )

    result: dict[str, Any] = {
        "kc_id": np.repeat(kc_id_str, len(obs_keys)),
        student_col: np.asarray(obs_student_ids, dtype=object),
        problem_col: np.asarray(obs_problem_ids, dtype=object),
        correctness_col: np.asarray(obs_correctness, dtype=np.int8),
        "_order": np.asarray(obs_order, dtype=np.int64),
        f"{_PKNOW}_mean": pknow_means,
        f"{_PKNOW}_std": pknow_stds,
        f"{_PKNOW}_median": pknow_medians,
    }
    for qi, q in enumerate(quantiles):
        result[f"{_PKNOW}_{q * 100:.2f}%"] = pknow_quants[:, qi]

    if pcorr_col_indices:
        pcorr_raw_col_indices = [pcorr_col_indices.get(k) for k in obs_keys]
        if all(ci is not None for ci in pcorr_raw_col_indices):
            pcorr_idx_arr = np.asarray(pcorr_raw_col_indices, dtype=np.intp)
            pcorr_arr = np.ascontiguousarray(
                raw_draws[:, pcorr_idx_arr].T, dtype=np.float64
            )
            pcorr_means, pcorr_stds, pcorr_medians, pcorr_quants = (
                _compute_posterior_stats(pcorr_arr, q_arr)
            )
            result[f"{_PCORRECT}_mean"] = pcorr_means
            result[f"{_PCORRECT}_std"] = pcorr_stds
            result[f"{_PCORRECT}_median"] = pcorr_medians
            for qi, q in enumerate(quantiles):
                result[f"{_PCORRECT}_{q * 100:.2f}%"] = pcorr_quants[:, qi]

    kc_summary = pd.DataFrame(result)
    kc_summary.sort_values(by=[student_col, "_order"], inplace=True)
    kc_summary.drop(columns=["_order"], inplace=True)
    return kc_summary


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

    stat_fns: list[tuple[str, Any]] = [
        ("mean", "mean"),
        ("std", "std"),
        ("median", "median"),
    ] + [(f"{q * 100:.2f}%", (lambda s, q=q: s.quantile(q))) for q in quantiles]

    summary_cols = [f"{_PKNOW}_{stat}" for stat, _ in stat_fns]
    summary_cols += [f"{_PCORRECT}_{stat}" for stat, _ in stat_fns]

    param_id_cols = [
        col_mapping.get(ColumnNames.STUDENT_ID),
        col_mapping.get(ColumnNames.PROBLEM_ID),
        col_mapping.get(ColumnNames.CORRECTNESS),
    ]

    if not draws:
        return pd.DataFrame(columns=["kc_id"] + param_id_cols + summary_cols)

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
            return pd.DataFrame(columns=["kc_id"] + param_id_cols + summary_cols)
        return pd.concat(result_frames, ignore_index=True)
    else:
        draw_frames = cast(dict[str, pd.DataFrame], draws)

    result_frames: list[pd.DataFrame] = []
    for kc, kc_df in draw_frames.items():
        kc_summary = _summarize_single_kc_draws(str(kc), kc_df, col_mapping, stat_fns)
        if not kc_summary.empty:
            result_frames.append(kc_summary)

    if not result_frames:
        return pd.DataFrame(columns=["kc_id"] + param_id_cols + summary_cols)

    return pd.concat(result_frames, ignore_index=True)
