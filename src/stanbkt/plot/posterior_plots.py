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
    posterior_pred_kc: pd.DataFrame,
    data: pd.DataFrame,
    kc: str,
    grouped: bool = False,
    type: Literal["probs", "preds"] = "preds",
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
) -> Union[plt.Axes, npt.NDArray[np.object_]]:
    """Plot posterior predictions of correctness for a given KC.

    Parameters
    ----------
    posterior_pred_kc : pandas.DataFrame
        Posterior predictions for the selected KC.
    data : pandas.DataFrame
        Input data containing student interactions.
    kc : str
        The KC for which to plot posterior predictions.
    grouped : bool, optional, default False
        Whether to produce one subplot per group using the group column from
        `column_mapping` (or `group_id` by default). If True, the group column
        is required in `data`.
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
    matplotlib.axes.Axes or numpy.ndarray[matplotlib.axes.Axes]
        A single axis for ungrouped data, or an array of subplot axes (one per group)
        when grouped data is provided.
    """
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
    student_col = column_mapping[ColumnNames.STUDENT_ID]
    problem_col = column_mapping[ColumnNames.PROBLEM_ID]
    group_col = column_mapping.get(ColumnNames.GROUP, ColumnNames.GROUP)

    validate_data(data, column_mapping, check_groups=grouped)

    # subset data to the kc
    data_kc: pd.DataFrame = data[data[column_mapping[ColumnNames.KC_ID]] == kc]
    # subset to requested problems if provided
    if problem_ids is not None:
        data_kc = data_kc[
            data_kc[column_mapping[ColumnNames.PROBLEM_ID]].isin(problem_ids)
        ]
    grouped_plot = grouped
    posterior_kc = posterior_pred_kc.copy()

    if type == "preds":
        posterior_kc[_PCORRECT] = np.random.binomial(
            n=1, p=posterior_kc["pCorrectness"].values, size=len(posterior_kc)
        )

    posterior_problem_col = (
        problem_col if problem_col in posterior_kc.columns else ColumnNames.PROBLEM_ID
    )
    posterior_student_col = (
        student_col if student_col in posterior_kc.columns else ColumnNames.STUDENT_ID
    )

    if grouped_plot:
        correctness_by_problem_by_group, problem_ids = (
            _point_estimate_correctness_per_problem_by_group(
                data_kc, column_mapping, group_col, point_estimate, frac
            )
        )
        student_group_map = data_kc[[student_col, group_col]].drop_duplicates()
        student_group_check = student_group_map.groupby(student_col, observed=True)[
            group_col
        ].nunique()
        if (student_group_check > 1).any():
            raise ValueError(
                f"Each student must map to exactly one group in '{group_col}' for grouped plotting."
            )

        posterior_kc = posterior_kc.merge(
            student_group_map,
            left_on=posterior_student_col,
            right_on=student_col,
            how="left",
        )
        if posterior_kc[group_col].isna().any():
            raise ValueError(
                f"Could not map all posterior rows to '{group_col}'. Ensure posterior prediction rows include '{posterior_student_col}'."
            )

        posterior_grouped = (
            posterior_kc.groupby(
                [group_col, posterior_problem_col, "draw__"], observed=True
            )[_PCORRECT]
            .agg("mean")
            .reset_index()
        )
        posterior_grouped = posterior_grouped.groupby(
            [group_col, posterior_problem_col], observed=True
        )[_PCORRECT].agg(
            pe=point_estimate,
            lower=lambda x: x.quantile(percentiles[0] / 100),
            upper=lambda x: x.quantile(percentiles[1] / 100),
        )

        groups = natsort.natsorted(list(correctness_by_problem_by_group.keys()))
        if len(groups) == 0:
            raise ValueError(f"No groups found in '{group_col}' for KC '{kc}'.")

        x_positions = np.arange(len(problem_ids))
        fig_width = max(12.0, 0.55 * len(problem_ids))
        fig_height = max(6.0, 2.8 * len(groups))
        fig, axes = plt.subplots(
            len(groups),
            1,
            sharex=True,
            sharey=True,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(axes)
        color_cycle = plt.get_cmap("tab10")
        legend_handles = None
        legend_labels = None
        for group_index, group in enumerate(groups):
            ax = axes_arr[group_index]
            color = color_cycle(group_index % 10)
            group_correctness = correctness_by_problem_by_group[group].reindex(
                problem_ids
            )
            group_posterior = posterior_grouped.xs(group, level=group_col).reindex(
                problem_ids
            )

            if trajectory:
                ax.plot(
                    x_positions - offset,
                    group_correctness.values,
                    marker="x",
                    color=color,
                    label="Proportion Correct (Data)",
                    zorder=3,
                    linestyle=":",
                )
                ax.plot(
                    x_positions,
                    group_posterior["pe"].values,
                    marker="o",
                    color=color,
                    linestyle="--",
                    zorder=2,
                )
            else:
                ax.scatter(
                    x_positions - offset,
                    group_correctness.values,
                    marker="x",
                    color=color,
                    label="Proportion Correct (Data)",
                    zorder=3,
                )
                ax.scatter(
                    x_positions,
                    group_posterior["pe"].values,
                    marker="o",
                    color=color,
                    zorder=2,
                )

            yerr = np.array(
                [
                    group_posterior["pe"].values - group_posterior["lower"].values,
                    group_posterior["upper"].values - group_posterior["pe"].values,
                ]
            )
            ax.errorbar(
                x_positions,
                group_posterior["pe"].values,
                yerr=yerr,
                fmt="none",
                color=color,
                capsize=3,
                label=f"{percentiles[0]}–{percentiles[1]}%"
                f" {('C.I.' if type == 'probs' else 'P.I.')}",
            )
            ax.set_xticks(x_positions)
            if group_index == len(groups) - 1:
                ax.set_xticklabels(problem_ids, rotation=90)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.set_title(f"{group_col}={group}")
            if group_index == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        fig.supylabel(y_axis_label)
        axes_arr[-1].set_xlabel("Problem ID")
        fig.suptitle(
            f"Posterior Correctness ({'predictions' if type == 'preds' else 'probabilities'}) - {kc}"
        )
        if legend_handles and legend_labels:
            legend = fig.legend(
                legend_handles,
                legend_labels,
                loc="outside upper right",
            )
            plt.setp(legend.get_texts(), color="black")
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(1.0)
        return axes_arr
    else:
        fig, ax = plt.subplots(constrained_layout=True)
        correctness_by_problem, problem_ids = _point_estimate_correctness_per_problem(
            data_kc, column_mapping, point_estimate, frac
        )
        posterior_kc = posterior_kc.groupby(
            [posterior_problem_col, "draw__"], observed=True
        )[_PCORRECT].agg("mean")
        posterior_kc = posterior_kc.groupby(level=0, observed=True).agg(
            pe=point_estimate,
            lower=lambda x: x.quantile(percentiles[0] / 100),
            upper=lambda x: x.quantile(percentiles[1] / 100),
        )
        posterior_kc = posterior_kc.reindex(problem_ids)

        x_positions = np.arange(len(problem_ids))

        if trajectory:
            ax.plot(
                x_positions - offset,
                correctness_by_problem,
                marker="x",
                color="black",
                label="Proportion Correct (Data)",
                zorder=3,
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
            label=f"{percentiles[0]}–{percentiles[1]}%"
            f" {('C.I.' if type == 'probs' else 'P.I.')}",
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(problem_ids, rotation=90)
        ax.set_xlabel("Problem ID")
        ax.set_ylabel(y_axis_label)
        ax.set_title(
            f"Posterior Correctness ({'predictions' if type == 'preds' else 'probabilities'}) - {kc}"
        )
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        if legend_handles and legend_labels:
            legend = fig.legend(
                legend_handles,
                legend_labels,
                loc="outside upper right",
            )
            plt.setp(legend.get_texts(), color="black")
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(1.0)
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


def _point_estimate_correctness_per_problem_by_group(
    data: pd.DataFrame,
    column_mapping: dict[str, str],
    group_col: str,
    agg_func: Literal["mean", "median", "mode"] = "mean",
    frac: float = 1.0,
) -> tuple[dict[str, pd.Series], list[str]]:
    """Compute correctness per problem separately for each group."""
    data = data.copy()
    if data[column_mapping[ColumnNames.KC_ID]].nunique() > 1:
        raise ValueError("Data contains multiple KCs. Please subset to a single KC.")

    problem_col = column_mapping[ColumnNames.PROBLEM_ID]
    correctness_col = column_mapping[ColumnNames.CORRECTNESS]

    data[problem_col] = data[problem_col].astype(str)
    problem_ids = data[problem_col].unique().tolist()
    problem_ids.sort(key=natsort.natsort_keygen())

    problems_to_plot = problem_ids
    if frac < 1.0:
        num_problems_to_plot = max(2, int(len(problem_ids) * frac))
        problems_to_plot = np.linspace(
            0, len(problem_ids) - 1, num_problems_to_plot, dtype=int
        )
        problems_to_plot = [problem_ids[i] for i in problems_to_plot]
        data = data[data[problem_col].isin(problems_to_plot)]

    grouped = (
        data.groupby([group_col, problem_col], observed=True)[correctness_col]
        .agg(agg_func)
        .unstack(problem_col)
    )
    grouped = grouped.reindex(columns=problems_to_plot)
    return {str(group): grouped.loc[group] for group in grouped.index}, problems_to_plot
