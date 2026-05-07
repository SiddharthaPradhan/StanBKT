from __future__ import annotations

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
