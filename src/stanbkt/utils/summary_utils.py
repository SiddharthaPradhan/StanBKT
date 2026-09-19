from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd


def summary_parameter_names(column_names: Sequence[str]) -> list[str]:
    """Return parameter names to include in fit summaries.

    Mirrors CmdStan's MCMC summary behavior by retaining ``lp__`` and
    excluding method variables (which conventionally end with ``__``).
    """
    return [
        name for name in column_names if name == "lp__" or not name.endswith("__")
    ]


def summarize_draws(
    draws: np.ndarray,
    parameter_names: Sequence[str],
    percentiles: tuple[float, float],
) -> pd.DataFrame:
    """Summarize posterior draws for each parameter.

    Returns a DataFrame with one row per parameter and a stable set of
    descriptive columns aligned with the fit-level summary API.
    """
    if draws.ndim != 2:
        raise ValueError(
            f"Expected 2-D draws array (draws x parameters), found shape {draws.shape}."
        )

    if draws.shape[1] != len(parameter_names):
        raise ValueError(
            "Draw matrix column count does not match parameter names length: "
            f"{draws.shape[1]} != {len(parameter_names)}."
        )

    lower_label = f"{percentiles[0]}%"
    upper_label = f"{percentiles[1]}%"

    if draws.shape[0] == 0:
        return pd.DataFrame(
            {
                "parameter": list(parameter_names),
                "mean": np.nan,
                "sd": np.nan,
                lower_label: np.nan,
                "50%": np.nan,
                upper_label: np.nan,
            }
        )

    return pd.DataFrame(
        {
            "parameter": list(parameter_names),
            "mean": np.mean(draws, axis=0),
            "sd": np.std(draws, axis=0, ddof=0),
            lower_label: np.percentile(draws, percentiles[0], axis=0),
            "50%": np.percentile(draws, 50, axis=0),
            upper_label: np.percentile(draws, percentiles[1], axis=0),
        }
    )


_INDEXED_PARAM = re.compile(r"^(\w+)\[(\d+)\]$")
_GROUP_PARAMS = {
    "learn",
    "forget",
    "guess",
    "slip",
    "logit_learn_group",
    "logit_forget_group",
    "logit_guess_group",
    "logit_slip_group",
}
_PI_KNOW_PARAMS = {"pi_know", "logit_pi_know_group"}
_JOINT_SCALAR_PARAMS = {"pi_b0_know_param", "pi_sigma_param"}


def label_summary_index(
    summary_df: pd.DataFrame,
    group2index: dict[str, dict[str, int] | None],
    student2index: dict[str, dict[str, int] | None],
    covariate_columns: dict[str, tuple[str, ...] | None],
    group_col_name: str,
    student_col_name: str,
    covariate_axis_name: str = "covariate",
) -> pd.DataFrame:
    """Replace Stan integer indexes in a summary with the labels they represent.

    Expects a frame indexed by ``[kc, "parameter"]`` whose parameter names look like
    ``learn[2]``. Returns a frame indexed by ``[kc, "parameter", "axis", "label"]`` where
    ``parameter`` is the bare name (``learn``), ``axis`` names what the index counts
    (group column, student column or covariate) and ``label`` is the original group ID,
    student ID or covariate name. Scalar parameters get empty ``axis`` and ``label``.
    Indexes with no stored mapping keep the raw integer as the label.

    The per-KC mappings are keyed by KC and typically come from the persisted fit metadata.
    """
    kc_level, param_level = summary_df.index.names
    kcs = summary_df.index.get_level_values(kc_level)
    names = summary_df.index.get_level_values(param_level)

    base_names: list[str] = []
    axes: list[str] = []
    labels: list[str] = []
    for kc, name in zip(kcs, names):
        match = _INDEXED_PARAM.match(str(name))
        if match is None:
            base_names.append(str(name))
            axes.append("")
            labels.append("")
            continue
        base, idx = match.group(1), int(match.group(2))
        if base in _JOINT_SCALAR_PARAMS:
            base_names.append(base)
            axes.append("")
            labels.append("")
            continue
        student_map = student2index.get(str(kc))
        group_map = group2index.get(str(kc))
        cov_cols = covariate_columns.get(str(kc))

        axis, lookup = "", None
        if base in _GROUP_PARAMS:
            axis, lookup = group_col_name, group_map
        elif base in _PI_KNOW_PARAMS:
            if student_map:
                axis, lookup = student_col_name, student_map
            else:
                axis, lookup = group_col_name, group_map
        elif base == "logit_pi_know_z":
            axis, lookup = student_col_name, student_map
        elif base == "pi_b1_know_param" and cov_cols is not None:
            axis = covariate_axis_name
            lookup = {col: i + 1 for i, col in enumerate(cov_cols)}

        index2label = {v: k for k, v in lookup.items()} if lookup else {}
        base_names.append(base)
        axes.append(axis)
        labels.append(str(index2label.get(idx, idx)))

    labeled = summary_df.copy()
    labeled.index = pd.MultiIndex.from_arrays(
        [kcs, base_names, axes, labels],
        names=[kc_level, param_level, "axis", "label"],
    )
    return labeled
