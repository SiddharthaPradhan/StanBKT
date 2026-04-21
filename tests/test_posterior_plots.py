import matplotlib
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import pytest
from typing import cast

from stanbkt.plot.posterior_plots import plot_posterior_correctness
from stanbkt.utils.data_utils import ColumnNames

matplotlib.use("Agg")


def _base_grouped_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 0, 1],
            "timestamp": [1, 2, 1, 2],
            "kc_id": ["kc_a", "kc_a", "kc_a", "kc_a"],
            "group_id": ["g1", "g1", "g2", "g2"],
        }
    )


def _base_posterior_draws() -> dict[str, pd.DataFrame]:
    return {
        "kc_a": pd.DataFrame(
            {
                "draw__": [1, 1, 1, 1, 2, 2, 2, 2],
                "student_id": ["s1", "s1", "s2", "s2"] * 2,
                "problem_id": ["p1", "p2", "p1", "p2"] * 2,
                "pCorrectness": [0.8, 0.7, 0.3, 0.4, 0.9, 0.6, 0.2, 0.5],
            }
        )
    }


def test_plot_posterior_correctness_grouped_uses_single_figure_legend() -> None:
    data = _base_grouped_data()
    posterior_preds = _base_posterior_draws()

    axes = plot_posterior_correctness(
        posterior_preds=posterior_preds,
        data=data,
        kc="kc_a",
        grouped=True,
        type="probs",
    )

    assert isinstance(axes, np.ndarray)
    axes_list = cast(list[Axes], axes.tolist())
    assert len(axes_list) == 2
    assert axes_list[0].get_shared_y_axes().joined(axes_list[0], axes_list[1])
    assert axes_list[0].get_title() == "group_id=g1"
    assert axes_list[1].get_title() == "group_id=g2"
    assert all(ax.get_legend() is None for ax in axes_list)
    fig = axes_list[0].figure
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert all(text.get_color() == "black" for text in legend.get_texts())
    assert legend.get_frame().get_edgecolor() == (0.0, 0.0, 0.0, 1.0)


def test_plot_posterior_correctness_grouped_respects_group_mapping() -> None:
    data = _base_grouped_data().rename(
        columns={
            "student_id": "sid",
            "problem_id": "pid",
            "correct": "is_correct",
            "timestamp": "t",
            "kc_id": "skill",
            "group_id": "cohort",
        }
    )
    posterior_preds = {
        "kc_a": _base_posterior_draws()["kc_a"].rename(
            columns={"student_id": "sid", "problem_id": "pid"}
        )
    }
    col_mapping = {
        ColumnNames.STUDENT_ID: "sid",
        ColumnNames.PROBLEM_ID: "pid",
        ColumnNames.CORRECTNESS: "is_correct",
        ColumnNames.ORDER: "t",
        ColumnNames.KC_ID: "skill",
        ColumnNames.GROUP: "cohort",
    }

    axes = plot_posterior_correctness(
        posterior_preds=posterior_preds,
        data=data,
        kc="kc_a",
        grouped=True,
        type="probs",
        column_mapping=col_mapping,
    )

    assert isinstance(axes, np.ndarray)
    axes_list = cast(list[Axes], axes.tolist())
    assert len(axes_list) == 2
    assert axes_list[0].get_shared_y_axes().joined(axes_list[0], axes_list[1])
    assert axes_list[0].get_title() == "cohort=g1"
    assert axes_list[1].get_title() == "cohort=g2"


def test_plot_posterior_correctness_grouped_true_requires_group_column() -> None:
    data = _base_grouped_data().drop(columns=["group_id"])
    posterior_preds = _base_posterior_draws()

    with pytest.raises(ValueError, match="Missing required columns"):
        plot_posterior_correctness(
            posterior_preds=posterior_preds,
            data=data,
            kc="kc_a",
            grouped=True,
            type="probs",
        )


def test_plot_posterior_correctness_grouped_false_ignores_group_column() -> None:
    data = _base_grouped_data()
    posterior_preds = _base_posterior_draws()

    ax = plot_posterior_correctness(
        posterior_preds=posterior_preds,
        data=data,
        kc="kc_a",
        grouped=False,
        type="probs",
    )

    assert isinstance(ax, Axes)
    assert ax.get_legend() is None
    fig = ax.figure
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert all(text.get_color() == "black" for text in legend.get_texts())
