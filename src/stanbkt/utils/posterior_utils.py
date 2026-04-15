"""Utilities for summarising posterior prediction draw DataFrames."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Union, Mapping

import cmdstanpy as csp
import numpy as np
import pandas as pd

from stanbkt.utils.data_utils import ColumnNames, _PKNOW, _PCORRECT, iter_kc_data


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
    student_col = col_mapping[ColumnNames.STUDENT_ID]
    problem_col = col_mapping[ColumnNames.PROBLEM_ID]
    correctness_col = col_mapping[ColumnNames.CORRECTNESS]
    kc_col = col_mapping[ColumnNames.KC_ID]
    id_cols: list[str] | None = None
    col_pat = re.compile(r"^([^\[]+)\[(\d+)\s*,\s*(\d+)\]$")

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

        draws_df = gq_kc.draws_pd()
        if id_cols is None:
            id_cols = [c for c in draws_df.columns.astype(str) if c.endswith("__")]

        pknow_cols: dict[tuple[int, int], str] = {}
        pcorr_cols: dict[tuple[int, int], str] = {}
        for col in draws_df.columns:
            m = col_pat.match(str(col))
            if m:
                param, s, p = m.group(1), int(m.group(2)), int(m.group(3))
                if param == _PKNOW:
                    pknow_cols[(s, p)] = col
                elif param == _PCORRECT:
                    pcorr_cols[(s, p)] = col

        n_draws = len(draws_df)

        # Build observation-level arrays in kc_data sequence order (student-major,
        # then temporal order within student) in a single pass.
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
                # Per-student temporal index from kc_data ordering.
                obs_order.append(p_idx - 1)
                obs_student_ids.append(student_id)
                obs_problem_ids.append(problem_id)
                obs_correctness.append(int(kc_data.correctness[s_idx - 1, p_idx - 1]))

        n_obs = len(obs_keys)

        obs_student_ids_arr = np.asarray(obs_student_ids, dtype=object)
        obs_problem_ids_arr = np.asarray(obs_problem_ids, dtype=object)
        obs_correctness_arr = np.asarray(obs_correctness, dtype=np.int8)
        obs_order_arr = np.asarray(obs_order, dtype=np.int64)

        pknow_values = draws_df[pknow_obs_cols].to_numpy().ravel()

        result: dict[str, Any] = {
            col: np.repeat(draws_df[col].to_numpy(), n_obs) for col in (id_cols or [])
        }
        result[kc_col] = np.repeat(kc_id_str, n_draws * n_obs)
        result[student_col] = np.tile(obs_student_ids_arr, n_draws)
        result[problem_col] = np.tile(obs_problem_ids_arr, n_draws)
        result[correctness_col] = np.tile(obs_correctness_arr, n_draws)
        # Keep the same observation order values across draws so posterior_summary
        # can aggregate draw samples for the same observation.
        result["_order"] = np.tile(obs_order_arr, n_draws)
        result[_PKNOW] = pknow_values
        if pcorr_cols:
            result[_PCORRECT] = (
                draws_df[[pcorr_cols[k] for k in obs_keys]].to_numpy().ravel()
            )

        posterior_draws[kc_id] = pd.DataFrame(result)
    return posterior_draws


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
        for pKnow (and pCorrectness when present).
    """
    if not draws:
        raise ValueError("'draws' is empty.")

    # If the caller passed Stan GQ output, convert to draws first.
    first_val = next(iter(draws.values()))
    if not isinstance(first_val, pd.DataFrame):
        if data is None:
            raise ValueError(
                "'data' must be provided when 'draws' contains CmdStanGQ objects. "
                "Pass the original student interaction data used to generate the Stan output."
            )
        draws = gq_to_draws(draws, data=data, col_mapping=col_mapping)  # type: ignore[arg-type]

    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")

    col_mapping = ColumnNames.apply_default_mapping(col_mapping)

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

    result_frames: list[pd.DataFrame] = []
    for kc, kc_df in draws.items():
        if kc_df.empty:
            continue

        agg_cols = [c for c in [_PKNOW, _PCORRECT] if c in kc_df.columns]
        agg_kwargs = {
            f"{col}_{stat}": pd.NamedAgg(column=col, aggfunc=fn)
            for col in agg_cols
            for stat, fn in stat_fns
        }
        # Include "_order" when available to preserve temporal sequence from
        # kc_data-derived draws. Some callers pass hand-crafted draws without
        # this internal column.
        has_order_col = "_order" in kc_df.columns
        groupby_cols = param_id_cols + (["_order"] if has_order_col else [])
        kc_summary: pd.DataFrame = (
            kc_df.groupby(groupby_cols, sort=False).agg(**agg_kwargs).reset_index()
        )
        # Sort by student first, then by order within that student when present.
        student_col_name = col_mapping.get(ColumnNames.STUDENT_ID)
        sort_cols = [student_col_name] + (["_order"] if has_order_col else [])
        kc_summary.sort_values(by=sort_cols, inplace=True)
        if has_order_col:
            kc_summary.drop(columns=["_order"], inplace=True)
        kc_summary.insert(0, "kc_id", str(kc))
        result_frames.append(kc_summary)

    if not result_frames:
        return pd.DataFrame(columns=["kc_id"] + param_id_cols + summary_cols)

    return pd.concat(result_frames, ignore_index=True)
