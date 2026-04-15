from collections.abc import Mapping
from stanbkt.utils.data_utils import validate_data, _PCORRECT
import natsort
from stanbkt.utils.data_utils import format_kc_data, iter_kc_data, KCData, ColumnNames
from stanbkt.models.core.base import BKTModelBase
from stanbkt.fits.fit_types import FitMethod
from stanbkt.fits.core.base import FitBase
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Literal, Union
import numpy as np
import numpy.typing as npt


def plot_posterior_correctness(
    posterior_preds: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    kc: str,
    type: Literal["probs", "preds"],
    point_estimate: Literal["mean", "median", "mode"] = "mean",
    *,
    percentiles: tuple[float, float] = (2.5, 97.5),
    problem_ids: Optional[list[str]] = None,
    trajectory: bool = False,
    frac=1.0,
    column_mapping: Union[
        Mapping[ColumnNames, str],
        Mapping[str, str],
        Mapping[ColumnNames | str, str],
    ] = {},
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
    point_estimate : Literal["mean", "median", "mode"], optional, default "mean"
        The point estimate to display for each problem. Can be "mean", "median", or "mode".
    problem_ids : list[str], optional
        List of problem IDs to include in the plot. If None, all problems for the KC
    trajectory: bool = False,
        Whether to connect the data points with a line to show the trajectory across problems. Default is False.
    percentiles : tuple[float, float], optional, Default is (2.5, 97.5) for a 95% credible interval.
        The lower and upper percentiles to display as error bars. Values should be in range [1, 99].
    frac : float, optional
        Fraction of problems to use for plotting. The problems will be linearly spaced. Useful for large datasets to reduce overplotting.
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
    column_mapping = ColumnNames.apply_default_mapping(column_mapping)
    if (
        kc
        not in data[column_mapping.get(ColumnNames.KC_ID, ColumnNames.KC_ID)].unique()
    ):
        raise ValueError(f"KC '{kc}' not found in input data.")
    if point_estimate not in ["mean", "median", "mode"]:
        raise ValueError("point_estimate must be 'mean', 'median', or 'mode'.")
    if type not in ["probs", "preds"]:
        raise ValueError(f"type must be either 'probs' or 'preds'. Got {type}.")
    if len(percentiles) != 2 or not all(1 <= p <= 99 for p in percentiles):
        raise ValueError(
            "percentiles must be a tuple of two integer values between 1 and 99."
        )
    if percentiles[0] >= percentiles[1]:
        raise ValueError(
            f"First percentile must be less than second percentile. Got {percentiles}."
        )
    y_axis_label = (
        "Prob/Prop of Correctness" if type == "probs" else "Proportion Correct"
    )

    validate_data(data, column_mapping)

    # subset data to the kc
    data_kc: pd.DataFrame = data[data[column_mapping[ColumnNames.KC_ID]] == kc]
    # subset to requested problems if provided
    if problem_ids is not None:
        data_kc = data_kc[
            data_kc[column_mapping[ColumnNames.PROBLEM_ID]].isin(problem_ids)
        ]
    correctness_by_problem, problem_ids = _point_estimate_correctness_per_problem(
        data_kc, column_mapping, point_estimate, frac
    )
    posterior_kc = posterior_preds[kc]

    if type == "preds":
        posterior_kc[_PCORRECT] = np.random.binomial(
            n=1, p=posterior_kc["pCorrectness"].values, size=len(posterior_kc)
        )

    posterior_kc: pd.DataFrame = posterior_kc.groupby(["problem_id", "draw__"])[
        _PCORRECT
    ].agg("mean")

    posterior_kc: pd.DataFrame = posterior_kc.groupby("problem_id").agg(
        pe=point_estimate,
        lower=lambda x: x.quantile(percentiles[0] / 100),
        upper=lambda x: x.quantile(percentiles[1] / 100),
    )
    posterior_kc: pd.DataFrame = posterior_kc.reindex(problem_ids)

    x_positions = np.arange(len(problem_ids))

    fig, ax = plt.subplots()

    if trajectory:
        ax.plot(
            x_positions - offset,
            correctness_by_problem,
            marker="x",
            color="black",
            label="Proportion Correct (Data)",
            zorder=3,  # ensure data are higher than the predictions
            linestyle=":",
        )
        ax.plot(
            x_positions,
            posterior_kc["pe"].values,
            marker="o",
            color="steelblue",
            linestyle="--",
            zorder=2,
        )
    else:
        # true
        ax.scatter(
            x_positions - offset,
            correctness_by_problem,
            marker="x",
            color="black",
            label="Proportion Correct (Data)",
            zorder=3,
        )
        # predictions
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
        label=f"{percentiles[0]}–{percentiles[1]}%"
        f" {('C.I.' if type == 'probs' else 'P.I.')}",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(problem_ids, rotation=90)
    ax.set_xlabel("Problem ID")
    ax.set_ylabel(y_axis_label)
    ax.set_title(f"Posterior Correctness — {kc}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return ax


def _point_estimate_correctness_per_problem(
    data: pd.DataFrame,
    column_mapping: dict[str, str],
    agg_func: Literal["mean", "median", "mode"] = "mean",
    frac: float = 1.0,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Helper function to compute average correctness per problem for a given data."""
    data = data.copy()
    if data[column_mapping[ColumnNames.KC_ID]].nunique() > 1:
        raise ValueError("Data contains multiple KCs. Please subset to a single KC.")
    data[column_mapping[ColumnNames.PROBLEM_ID]] = data[
        column_mapping[ColumnNames.PROBLEM_ID]
    ].astype(str)
    problem_ids: pd.Series = (
        data[column_mapping[ColumnNames.PROBLEM_ID]].unique().tolist()
    )
    problem_ids.sort(key=natsort.natsort_keygen())
    problems_to_plot = problem_ids
    if frac < 1.0:

        num_problems_to_plot = max(2, int(len(problem_ids) * frac))
        problems_to_plot = np.linspace(
            0, len(problem_ids) - 1, num_problems_to_plot, dtype=int
        )
        problems_to_plot = [problem_ids[i] for i in problems_to_plot]
        data = data[data[column_mapping[ColumnNames.PROBLEM_ID]].isin(problems_to_plot)]
    correctness_by_problem: pd.Series = (
        data[column_mapping[ColumnNames.CORRECTNESS]]
        .groupby(data[column_mapping[ColumnNames.PROBLEM_ID]])
        .agg(agg_func)
    )[problems_to_plot]
    return correctness_by_problem.values, problems_to_plot
