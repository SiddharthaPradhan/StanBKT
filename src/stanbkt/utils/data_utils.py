import pandas as pd
from typing import Optional, Callable
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator
from stanbkt.utils.verbose import VerbosityLevel
from enum import Enum


class ColumnNames(str, Enum):
    STUDENT_ID = "student_id"
    PROBLEM_ID = "problem_id"
    CORRECTNESS = "correct"
    KC_ID = "kc_id"
    GROUP = "group_id"

    @staticmethod
    def get_default_mapping() -> dict[str, str]:
        return {
            ColumnNames.STUDENT_ID: ColumnNames.STUDENT_ID,
            ColumnNames.PROBLEM_ID: ColumnNames.PROBLEM_ID,
            ColumnNames.CORRECTNESS: ColumnNames.CORRECTNESS,
            ColumnNames.KC_ID: ColumnNames.KC_ID,
            ColumnNames.GROUP: ColumnNames.GROUP,
        }


# the maximum number of dims allowed in generated quantities for variables of interest
# the dims are usually (student, problem)
MAX_GQ_DIMENSION = 3


@dataclass(slots=True)
class KCData:
    # correctness matrix of shape (num_students, num_problems)
    correctness: np.ndarray
    # list of student ids corresponding to rows in correctness matrix
    groups: Optional[np.ndarray] = None
    # optional mapping from group id to index in groups array
    group_2_index: Optional[dict[str, int]] = None


def validate_data(
    data: pd.DataFrame,
    col_mapping: dict[str, str],
    check_groups: bool = False,
) -> None:
    """
    Validate input data for BKT model fitting.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing student interactions.
    col_mapping : dict, optional
        Mapping of expected column names. Keys should be 'student_id', 'problem_id', 'correct', and 'kc_id'.
        If None, default column names are used.
    check_groups : bool, default=False
        Whether to check for group column in the data.

    Raises
    ------
    ValueError
        If required columns are missing or if correctness values are not binary.
    """

    required_cols = {
        col_mapping.get(ColumnNames.STUDENT_ID),
        col_mapping.get(ColumnNames.PROBLEM_ID),
        col_mapping.get(ColumnNames.CORRECTNESS),
    }
    if check_groups:
        required_cols.add(col_mapping.get(ColumnNames.GROUP))

    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")

    correctness_col = col_mapping.get(ColumnNames.CORRECTNESS)
    if not data[correctness_col].isin([0, 1]).all():
        raise ValueError(
            f"Correctness column '{correctness_col}' must contain only 0 and 1 values."
        )


def format_data(
    data: pd.DataFrame,
    col_mapping: Optional[dict[str, str]] = None,
    return_groups: bool = False,
    print_fn: Optional[Callable] = None,
) -> dict[str, KCData]:
    """
    Format input data for BKT model fitting.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing student interactions.
    col_mapping : dict, optional
        Mapping of expected column names. Keys should be 'student_id', 'problem_id', 'correct', and 'kc_id'.
        If None, default column names are used.
    return_groups : bool, default=False
        Whether to add student id to group id mapping in the returned dictionary.
    print_fn : callable, optional
        Optional function for printing messages (e.g., logging).
    Returns
    -------
    dict[str, KCData]
        Formatted data mapping KCs to correctness data.
    """
    return dict(
        iter_kc_data(
            data=data,
            col_mapping=col_mapping,
            return_groups=return_groups,
            print_fn=print_fn,
        )
    )


def iter_kc_data(
    data: pd.DataFrame,
    col_mapping: Optional[dict[str, str]] = None,
    return_groups: bool = False,
    print_fn: Optional[Callable] = None,
) -> Iterator[tuple[str, KCData]]:
    """
    Yield formatted KC data one KC at a time.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing student interactions.
    col_mapping : dict, optional
        Mapping of expected column names.
    return_groups : bool, default=False
        Whether to include per-student group indices.
    print_fn : callable, optional
        Optional function for printing messages (e.g., logging).

    Yields
    ------
    tuple[str, KCData]
        ``(kc_id, KCData)`` pairs for each KC in the input.
    """
    if col_mapping is None:
        col_mapping = ColumnNames.get_default_mapping().copy()
    else:
        col_mapping = col_mapping.copy()
        default_mapping = ColumnNames.get_default_mapping()
        for key in default_mapping.keys():
            col_mapping.setdefault(key, default_mapping[key])
        validate_data(data, col_mapping, return_groups)

    student_col = col_mapping.get(ColumnNames.STUDENT_ID)
    problem_col = col_mapping.get(ColumnNames.PROBLEM_ID)
    correctness_col = col_mapping.get(ColumnNames.CORRECTNESS)
    kc_column = col_mapping.get(ColumnNames.KC_ID)

    working_data = data
    if data.get(kc_column) is None:
        working_data = data.copy()
        working_data[kc_column] = "default_kc"

    working_data[student_col] = working_data[student_col].astype(str)
    working_data[kc_column] = working_data[kc_column].astype(str)

    # If column name for group_id is the same as student_id (i.e., individualized),
    # create a dummy group column
    if return_groups and col_mapping.get(ColumnNames.GROUP) == col_mapping.get(
        ColumnNames.STUDENT_ID
    ):
        group_col = ColumnNames.GROUP
        working_data = working_data.copy()
        working_data[group_col] = working_data[col_mapping.get(ColumnNames.STUDENT_ID)]
        col_mapping[ColumnNames.GROUP] = group_col

    for kc, subset in working_data.groupby(kc_column, sort=False, observed=True):
        correctness_wide = pd.pivot(
            subset,
            index=student_col,
            columns=problem_col,
            values=correctness_col,
        )
        null_count = correctness_wide.isna().sum().sum()
        if null_count > 0 and print_fn:
            print_fn(
                f"Warning: {null_count} null values detected. These problems will be dropped.",
                VerbosityLevel.INFO,
            )

        correctness_wide.dropna(inplace=True)
        kc_data = KCData(
            correctness=correctness_wide.values.astype(np.int8, copy=False)
        )

        if return_groups:
            group_col = col_mapping.get(ColumnNames.GROUP)
            student_group_df = (
                subset[[student_col, group_col]]
                .drop_duplicates()
                .set_index(student_col)
            )
            student_group_df = student_group_df.loc[correctness_wide.index]
            group_indices, unique_groups = pd.factorize(student_group_df[group_col])
            group_indices = group_indices.astype(np.int32, copy=False) + 1
            group_2_index = {
                group: index for index, group in enumerate(unique_groups, start=1)
            }
            kc_data.groups = group_indices
            kc_data.group_2_index = group_2_index

        yield str(kc), kc_data


# TODO: Make this accept user input for control
def summarize_state_predictions(gq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize generated-quantities draws for Hidden State Predictions.

    Parameters
    ----------
    gq_df : pd.DataFrame
        DataFrame of draws, typically ``gq_fit.draws_pd().T``.

    Returns
    -------
    pd.DataFrame
        Summary with columns: mean, std, median, 2.5%, 97.5%.
    """
    mask = ~gq_df.index.to_series().str.startswith(("chain__", "iter__", "draw__"))
    gq_df_clean = gq_df[mask]

    gq_summary = pd.DataFrame(
        {
            "mean": gq_df_clean.mean(axis=1),
            "std": gq_df_clean.std(axis=1),
            "median": gq_df_clean.median(axis=1),
            "2.5%": gq_df_clean.quantile(0.025, axis=1),
            "97.5%": gq_df_clean.quantile(0.975, axis=1),
        }
    )

    idx_nums = gq_summary.index.to_series().str.extract(r"(\d+),(\d+)")
    valid_rows = idx_nums.notna().all(axis=1)
    if valid_rows.any():
        sortable = idx_nums[valid_rows].astype(int)
        order = np.lexsort((sortable[1].values, sortable[0].values))
        sorted_valid = gq_summary.loc[valid_rows].iloc[order]
        gq_summary = pd.concat([sorted_valid, gq_summary.loc[~valid_rows]], axis=0)

    return gq_summary


def rename_summary_var_columns(
    summary_df: pd.DataFrame, expected_var_cols: list
) -> pd.DataFrame:
    """
    Rename columns in the summary DataFrame to match expected variable names.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with default column names.
    expected_var_cols : list
        List of expected variable column names.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with renamed columns.
    """
    if len(expected_var_cols) != len(summary_df.columns):
        raise ValueError(
            "Length of expected_var_cols must match number of columns in summary_df."
        )

    rename_mapping = {
        old: new for old, new in zip(summary_df.columns, expected_var_cols)
    }
    return summary_df.rename(columns=rename_mapping)


def summarize_state_predictions_test(
    gq_df: pd.DataFrame,
    quantiles=(0.025, 0.975),
    array_index_names: list[str] = [],
) -> pd.DataFrame:
    """
    Summarize generated-quantities draws for Hidden State Predictions.

    Parameters
    ----------
    gq_df : pd.DataFrame
        DataFrame of draws, typically ``gq_fit.draws_pd().T``.

    Returns
    -------
    pd.DataFrame
        Summary with columns: mean, std, median, 2.5%, 97.5%.
    """
    if len(gq_df) < 1:
        raise ValueError("Input DataFrame is empty.")
    if array_index_names and len(array_index_names) > 3:
        raise ValueError("array_index_names cannot have more than 3 elements.")

    mask: pd.Series = ~gq_df.index.to_series().str.startswith(
        ("chain__", "iter__", "draw__")
    )
    gq_df_clean: pd.DataFrame = gq_df[mask]

    # check the range of quantiles
    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")

    gq_summary = pd.DataFrame(
        {
            "mean": gq_df_clean.mean(axis=1),
            "std": gq_df_clean.std(axis=1),
            "median": gq_df_clean.median(axis=1),
        }
    )
    for q in quantiles:
        gq_summary[f"{round(q*100, 2)}%"] = gq_df_clean.quantile(q, axis=1)

    variable_dim = len(gq_summary.iloc[0]["variable"].split(","))

    if array_index_names:
        # check if size of array_index_names matches the number of dimensions in the draws 'variable' columns
        if variable_dim != len(array_index_names):
            raise ValueError(
                f"Length of array_index_names must match the number of dimensions in the variable columns. Expected {variable_dim}, got {len(array_index_names)}."
            )
    else:  # we assert variable_dim is at most 3 for our use case
        array_index_names = ["i", "j", "k"][
            :variable_dim
        ]  # default names for up to 3 dimensions

    # seperate each of the dimension indices in the 'variable' columns into separate columns
    variable_split = (
        gq_summary.index.to_series().str.rstrip("]").str.split("[\\[,\\]]", expand=True)
    )
    variable_split.columns = ["variable_name"] + array_index_names
    gq_summary: pd.DataFrame = pd.concat([variable_split, gq_summary], axis=1)
    gq_summary.reset_index(drop=True, inplace=True)
    gq_summary.sort_values(
        by=["variable_name"] + array_index_names, inplace=True, ignore_index=True
    )

    return gq_summary
