import pandas as pd
from typing import Optional, Callable, Any
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator
from stanbkt.utils.verbose import VerbosityLevel
from enum import StrEnum
from natsort import natsort_keygen

_NA_FILL_VALUE = -1
_DEFAULT_KC_ID = "default_kc"
_PKNOW = "pKnow"
_PCORRECT = "pCorrectness"


class ColumnNames(StrEnum):
    STUDENT_ID = "student_id"
    PROBLEM_ID = "problem_id"
    CORRECTNESS = "correct"
    KC_ID = "kc_id"
    GROUP = "group_id"

    @staticmethod
    def get_default_mapping() -> dict[str, str]:
        return {col: col for col in ColumnNames}

    @staticmethod
    def apply_default_mapping(col_mapping: Optional[dict[str, str]]) -> dict[str, str]:
        if not col_mapping:
            col_mapping = {}
        col_mapping = col_mapping.copy()
        default_mapping = ColumnNames.get_default_mapping()
        for key in default_mapping.keys():
            col_mapping.setdefault(key, default_mapping[key])
        return col_mapping


# the basic columns required for any BKT model fitting.
BASE_REQUIRED_COLS: set[ColumnNames] = {
    ColumnNames.STUDENT_ID,
    ColumnNames.PROBLEM_ID,
    ColumnNames.CORRECTNESS,
}


@dataclass(slots=True, frozen=True)
class StudentInteraction:
    "DataClass to store the attempted problem ids and the number of non-na interactions"

    problem_ids: list[str]
    length: int


@dataclass(slots=True, frozen=True)
class KCData:
    # correctness matrix of shape (num_students, num_problems)
    correctness: np.ndarray
    # student interaction data containing StudentInteraction
    #    - this is used to keep track of the original problem ids
    #    - and the lengths, to ignore entries in the correctness matrix that correspond to null
    student_inter_dict: dict[str, StudentInteraction]  # TODO: consider trimming this
    lengths: npt.NDArray[np.int32]  # lens of each student's interaction sequence
    student_ids: list[
        str
    ]  # student ids that matches the true interaction sequence in the correctness matrix
    problem_ids: list[str]  # problem ids in sequence
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


def format_kc_data(
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

    def process_student_interactions(
        row: pd.Series,
        student_inter_dict: dict,
        student_ids: list,
    ):
        """Returns interaction data for a student with NA values inserted at the end.
        Also updates the student_inter_dict with the attempted problem ids and length of interactions for the student.
        """
        student_id = row.name
        student_ids.append(student_id)
        # student_dict[student_idx] = row.isna().sum()
        na_mask = row.isna()
        return_row = pd.concat(
            [row[~na_mask], pd.Series([_NA_FILL_VALUE] * na_mask.sum())]
        ).reset_index(drop=True)
        student_entry = StudentInteraction(
            problem_ids=row[~na_mask].index.to_list(),
            length=(~na_mask).sum(),
        )
        # add student interaction the the student_inter_dict
        student_inter_dict[student_id] = student_entry
        # return row with non.nas placed on the left with nas filled with zeros (these will be ignored in the model and outputs)
        return return_row

    col_mapping = ColumnNames.apply_default_mapping(col_mapping)

    validate_data(
        data=data,
        col_mapping=col_mapping,
        check_groups=return_groups,
    )
    student_col = col_mapping.get(ColumnNames.STUDENT_ID)
    problem_col = col_mapping.get(ColumnNames.PROBLEM_ID)
    correctness_col = col_mapping.get(ColumnNames.CORRECTNESS)
    kc_column = col_mapping.get(ColumnNames.KC_ID)

    # if no kc column in data, add a default kc column
    working_data = data
    if data.get(kc_column) is None:
        working_data = data.copy()
        working_data[kc_column] = _DEFAULT_KC_ID

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
        student_inter_dict: dict[str, StudentInteraction] = {}
        student_ids: list[str] = list()
        correctness_wide = pd.pivot(
            subset,
            index=student_col,
            columns=problem_col,
            values=correctness_col,
        )
        # TODO: this may be problematic if the problems are not in natural order
        #       -> But we have to use pivot to get it in the right format.
        #       -> We can ask the users to add a suffix or prefix to the problem ids
        #               to ensure natural sorting.
        # pd.pivot sorts the problem and student id.
        # re-sort problems (columns) and students natural order
        correctness_wide.sort_index(
            axis="columns", key=natsort_keygen(), inplace=True
        )  # ty:ignore[no-matching-overload]
        correctness_wide.sort_index(
            axis="index", key=natsort_keygen(), inplace=True
        )  # ty:ignore[no-matching-overload]
        problem_ids: list[str] = correctness_wide.columns.astype(str).tolist()
        correctness_wide = correctness_wide.apply(
            lambda row: process_student_interactions(
                row, student_inter_dict, student_ids
            ),
            axis=1,
        )
        lengths: npt.NDArray[np.int32] = np.fromiter(
            (s.length for s in student_inter_dict.values()),
            dtype=np.int32,
            count=len(student_inter_dict),
        )
        kc_data_dict: dict[str, Any] = {
            "correctness": correctness_wide.values.astype(np.int8, copy=False),
            "student_inter_dict": student_inter_dict,
            "lengths": lengths,
            "student_ids": student_ids,
            "problem_ids": problem_ids,
        }

        if return_groups:
            group_col = col_mapping.get(ColumnNames.GROUP)
            student_group_df = (
                subset[[student_col, group_col]]
                .drop_duplicates()
                .set_index(student_col)
            )
            student_group_df: pd.DataFrame = student_group_df.loc[
                correctness_wide.index
            ]
            group_indices, unique_groups = pd.factorize(student_group_df[group_col])
            group_indices: npt.NDArray = group_indices.astype(np.int32, copy=False) + 1
            group_2_index: dict[str, int] = {
                group: index for index, group in enumerate(unique_groups, start=1)
            }
            kc_data_dict["groups"] = group_indices
            kc_data_dict["group_2_index"] = group_2_index

        kc_data = KCData(**kc_data_dict)

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


def dict_has_types(
    x: object, key_type: type = str, value_type: type = pd.DataFrame
) -> bool:
    if not isinstance(x, dict):
        return False

    return all(
        isinstance(k, key_type) and isinstance(v, value_type) for k, v in x.items()
    )
