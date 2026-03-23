from stanbkt.utils.data_utils import validate_data, _PCORRECT
import natsort
from stanbkt.utils.data_utils import format_kc_data, iter_kc_data, KCData, ColumnNames
from stanbkt.models.core.base import BKTModelBase
from stanbkt.fits.fit_types import FitMethod
from stanbkt.fits.core.base import BaseFit
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Literal
import numpy as np
import numpy.typing as npt


def plot_posterior_correctness(
    posterior_preds: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    kc: str,
    type: Literal["probs", "preds"],
    point_estimate: Literal["mean", "median"] = "mean",
    percentiles: tuple[float, float] = (0.025, 0.975),
    col_mapping: dict[str, str] = {},
    offset: float = 0,
) -> plt.Axes:
    """Plot posterior predictions of correctness for a given KC.

    Parameters
    ----------
    posterior_preds : dict[str, pd.DataFrame]
        Dictionary containing posterior predictions for each KC.
    data : pandas.DataFrame
        Input data containing student interactions.
    kc : str
        The KC for which to plot posterior predictions.
    type : Literal["probs", "preds"]
        Whether the posterior predictions are probabilities ("probs") or binary predictions ("preds").
    point_estimate : Literal["mean", "median"], optional, default "mean"
        The point estimate to display for each problem. Can be either "mean" or "median".
    percentiles : tuple[float, float], optional, Default is (0.025, 0.975) for a 95% credible interval.
        The lower and upper percentiles to display as error bars. Values should be between 0 and 1.
    col_mapping : dict, optional
        Mapping of expected column names. Keys should be 'student_id', 'problem_id', 'correct', and 'kc_id'.
        If None, default column names are used.
    offset : float, default 0.0.
        Horizontal offset to apply to the posterior predictions for better visibility when plotted alongside data points.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if kc not in posterior_preds:
        raise ValueError(f"KC '{kc}' not found in posterior predictions.")
    col_mapping = ColumnNames.apply_default_mapping(col_mapping)
    if kc not in data[col_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID)].unique():
        raise ValueError(f"KC '{kc}' not found in input data.")
    if point_estimate not in ["mean", "median"]:
        raise ValueError("point_estimate must be either 'mean' or 'median'.")
    if type not in ["probs", "preds"]:
        raise ValueError(f"type must be either 'probs' or 'preds'. Got {type}.")
    if len(percentiles) != 2 or not all(0 <= p <= 1 for p in percentiles):
        raise ValueError("percentiles must be a tuple of two values between 0 and 1.")
    if percentiles[0] >= percentiles[1]:
        raise ValueError(
            f"First percentile must be less than second percentile. Got {percentiles}."
        )
    y_axis_label = (
        "Prob/Prop of Correctness" if type == "probs" else "Proportion Correct"
    )

    validate_data(data, col_mapping)

    # subset data to the kc
    data_kc: pd.DataFrame = data[data[col_mapping[ColumnNames.KC_ID]] == kc]
    correctness_by_problem, problem_ids = _point_estimate_correctness_per_problem(
        data_kc, col_mapping, point_estimate
    )
    posterior_kc = posterior_preds[kc]
    posterior_kc: pd.DataFrame = posterior_kc.loc[
        posterior_kc[col_mapping[ColumnNames.KC_ID]] == kc
    ]

    if type == "preds":
        posterior_kc[_PCORRECT] = np.random.binomial(
            n=1, p=posterior_kc["pCorrectness"].values, size=len(posterior_kc)
        )

    posterior_kc: pd.DataFrame = posterior_kc.groupby(["problem_id", "draw__"])[
        _PCORRECT
    ].agg("mean")

    posterior_kc: pd.DataFrame = posterior_kc.groupby("problem_id").agg(
        pe=point_estimate,
        lower=lambda x: x.quantile(percentiles[0]),
        upper=lambda x: x.quantile(percentiles[1]),
    )
    posterior_kc: pd.DataFrame = posterior_kc.reindex(problem_ids)

    x_positions = np.arange(len(problem_ids))

    fig, ax = plt.subplots()

    ax.scatter(
        x_positions - offset,
        correctness_by_problem,
        marker="x",
        color="black",
        label="Proportion Correct (Data)",
        zorder=3,
    )

    ax.scatter(
        x_positions,
        posterior_kc["pe"].values,
        marker="o",
        color="steelblue",
        zorder=2,
    )

    yerr = np.array(
        [
            posterior_kc["pe"].values - posterior_kc["lower"].values,
            posterior_kc["upper"].values - posterior_kc["pe"].values,
        ]
    )
    ax.errorbar(
        x_positions,
        posterior_kc["pe"].values,
        yerr=yerr,
        fmt="none",
        color="steelblue",
        capsize=3,
        label=f"{round(percentiles[0] * 100, 2)}"
        f"–{round(percentiles[1] * 100,2)}%"
        f" {('C.I.' if type == 'probs' else 'P.I.')}",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(problem_ids, rotation=90)
    ax.set_xlabel("Problem ID")
    ax.set_ylabel(y_axis_label)
    ax.set_title(f"Posterior Correctness — {kc}")
    ax.legend()
    plt.tight_layout()
    return ax


def _point_estimate_correctness_per_problem(
    data: pd.DataFrame,
    col_mapping: dict[str, str],
    agg_func: Literal["mean", "median"] = "mean",
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Helper function to compute average correctness per problem for a given data."""
    data = data.copy()
    if data[col_mapping[ColumnNames.KC_ID]].nunique() > 1:
        raise ValueError("Data contains multiple KCs. Please subset to a single KC.")
    data[col_mapping[ColumnNames.PROBLEM_ID]] = data[
        col_mapping[ColumnNames.PROBLEM_ID]
    ].astype(str)
    correctness_by_problem: pd.Series = (
        data[col_mapping[ColumnNames.CORRECTNESS]]
        .groupby(data[col_mapping[ColumnNames.PROBLEM_ID]])
        .agg(agg_func)
    ).sort_index(key=natsort.natsort_keygen())
    return correctness_by_problem.values, correctness_by_problem.index.to_list()
