import pandas as pd
from typing import Optional, Callable
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator
from stanbkt.utils.verbose import VerbosityLevel
from enum import StrEnum


class ColumnNames(StrEnum):
    STUDENT_ID = "student_id"
    PROBLEM_ID = "problem_id"
    CORRECTNESS = "correct"
    KC_ID = "kc_id"
    GROUP = "group_id"

    @staticmethod
    def get_default_mapping() -> dict[str, str]:
        return {col: col for col in ColumnNames}


# the maximum number of dims allowed in generated quantities for variables of interest
# the dims are usually (student, problem)
MAX_GQ_DIMENSION = 3

# the basic columns required for any BKT model fitting.
BASE_REQUIRED_COLS: set[ColumnNames] = {
    ColumnNames.STUDENT_ID,
    ColumnNames.PROBLEM_ID,
    ColumnNames.CORRECTNESS,
}


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
    additional_required_cols: Optional[set[str]] = None,
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

    required_cols_mapped: set[str] = {
        col_mapping.get(col, col) for col in BASE_REQUIRED_COLS
    }

    # add any additional required columns to the mapped required columns sets
    if additional_required_cols:
        for additional_col in additional_required_cols:
            required_cols_mapped.add(col_mapping.get(additional_col, additional_col))

    if check_groups:
        required_cols_mapped.add(col_mapping.get(ColumnNames.GROUP, ColumnNames.GROUP))

    data_col_set = set(data.columns)
    if not required_cols_mapped.issubset(data_col_set):
        missing = required_cols_mapped - data_col_set
        raise ValueError(f"Missing required columns: {missing}")

    correctness_col = col_mapping.get(ColumnNames.CORRECTNESS, ColumnNames.CORRECTNESS)
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
        # drop along problem axis rather than the student axis.
        correctness_wide.dropna(axis=1, inplace=True)
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


def summarize_state_predictions(
    gq_df: pd.DataFrame,
    quantiles=(0.025, 0.975),
    array_index_names: Optional[list[str]] = None,
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
    if array_index_names and len(array_index_names) > MAX_GQ_DIMENSION:
        raise ValueError("array_index_names cannot have more than 3 elements.")

    mask: pd.Series = ~gq_df.index.to_series().str.startswith(
        ("chain__", "iter__", "draw__")
    )
    gq_df_clean: pd.DataFrame = gq_df[mask]

    # check the range of quantiles
    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")

    if gq_df_clean.empty:
        summary_cols = ["mean", "std", "median"] + [
            f"{round(q*100, 2)}%" for q in quantiles
        ]
        return pd.DataFrame(columns=summary_cols, index=pd.Index([], dtype="object"))

    gq_summary = pd.DataFrame(
        {
            "mean": gq_df_clean.mean(axis=1),
            "std": gq_df_clean.std(axis=1),
            "median": gq_df_clean.median(axis=1),
        }
    )
    for q in quantiles:
        gq_summary[f"{round(q*100, 2)}%"] = gq_df_clean.quantile(q, axis=1)

    extracted = gq_summary.index.to_series().str.extract(
        r"^(?P<variable_name>[^\[]+)\[(?P<indices>\d+(?:,\d+)*)\]$"
    )
    matched_rows = extracted["indices"].notna()

    if matched_rows.any():
        index_lengths = extracted.loc[matched_rows, "indices"].str.count(",") + 1
        variable_dim = int(index_lengths.iloc[0])

        if variable_dim > MAX_GQ_DIMENSION:
            raise ValueError(
                f"Generated quantity indices cannot have more than {MAX_GQ_DIMENSION} elements."
            )

        if array_index_names and variable_dim != len(array_index_names):
            raise ValueError(
                "Length of array_index_names must match the number of dimensions "
                f"in the variable indices. Expected {variable_dim}, got {len(array_index_names)}."
            )

        sort_df = pd.DataFrame(index=gq_summary.index)
        sort_df["_unmatched"] = (~matched_rows).astype(int)
        sort_df["_variable_name"] = extracted["variable_name"].fillna("")
        split_indices = extracted.loc[matched_rows, "indices"].str.split(
            ",", expand=True
        )
        for column_idx in range(variable_dim):
            column_name = f"_dim_{column_idx}"
            sort_df[column_name] = pd.Series(np.inf, index=gq_summary.index)
            sort_df.loc[matched_rows, column_name] = split_indices[column_idx].astype(
                int
            )
        sort_df["_position"] = np.arange(len(sort_df))

        sort_columns = (
            ["_unmatched", "_variable_name"]
            + [f"_dim_{column_idx}" for column_idx in range(variable_dim)]
            + ["_position"]
        )
        ordered_index = sort_df.sort_values(sort_columns).index
        gq_summary = gq_summary.loc[ordered_index]

    return gq_summary
