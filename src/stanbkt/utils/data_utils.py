from __future__ import annotations

import pandas as pd
from typing import Optional, Callable, Any, Union
from collections.abc import Mapping
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator
from stanbkt.utils.verbose import VerbosityLevel
from enum import StrEnum
from natsort import natsort_keygen

# fill value for missing interactions (these will be ignored in the model and the outputs)
_NA_FILL_VALUE = -1

# default kc id to use when there is no kc column in the data
_DEFAULT_KC_ID = "default_kc"

# columns names as defined in the stan code
_PKNOW = "pKnow"
_PCORRECT = "pCorrectness"


class ColumnNames(StrEnum):
    """Enumeration of standard column names for BKT data.

    Attributes
    ----------
    STUDENT_ID : str
        Unique student identifier column name.
    PROBLEM_ID : str
        Unique problem identifier column name.
    CORRECTNESS : str
        Binary correctness column (1=correct, 0=incorrect).
    ORDER : str
        The order in which the students attempted the problems.
    KC_ID : str
        Knowledge component identifier column name.
    GROUP : str
        Student or problem group identifier (optional).
    """

    STUDENT_ID = "student_id"
    PROBLEM_ID = "problem_id"
    CORRECTNESS = "correct"
    KC_ID = "kc_id"
    GROUP = "group_id"
    ORDER = "timestamp"

    @staticmethod
    def get_default_mapping() -> dict[str, str]:
        """Get default column name mapping.

        Returns
        -------
        dict[str, str]
            Mapping where all standard column names map to themselves.
        """
        return {col: col for col in ColumnNames}

    @staticmethod
    def apply_default_mapping(
        col_mapping: Optional[
            Union[
                Mapping[ColumnNames, str],
                Mapping[str, str],
                Mapping[ColumnNames | str, str],
            ]
        ] = None,
    ) -> dict[str, str]:
        """Apply default mapping to fill missing column name mappings.

        For any standard column not in the provided mapping, uses the default
        (column name maps to itself).

        Parameters
        ----------
        col_mapping : Optional[Union[dict[str, str], dict[ColumnNames, str]]]
            User-provided column name mapping. If None, treated as empty dict.

        Returns
        -------
        dict[str | ColumnNames, str]
            Complete column mapping with defaults applied.
        """
        result: dict[str, str] = (
            dict(col_mapping) if col_mapping else {}  # ty:ignore[no-matching-overload]
        )
        default_mapping = ColumnNames.get_default_mapping()
        for key in default_mapping.keys():
            result.setdefault(key, default_mapping[key])
        return result


# the basic columns required for any BKT model fitting.
BASE_REQUIRED_COLS: set[ColumnNames] = {
    ColumnNames.STUDENT_ID,
    ColumnNames.PROBLEM_ID,
    ColumnNames.CORRECTNESS,
    ColumnNames.ORDER,
}


@dataclass(slots=True, frozen=True)
class StudentInteraction:
    "DataClass to store the attempted problem ids and the number of non-na interactions"

    problem_ids: list[str]
    length: int


@dataclass(slots=True, frozen=True)
class KCData:
    """Structured data container for a single knowledge component.

    Stores preprocessed interaction data for a KC, including correctness matrix,
    student/problem identifiers, and optional group assignments.

    Attributes
    ----------
    correctness : np.ndarray
        Correctness matrix of shape (num_students, num_problems).
    student_inter_dict : dict[str, StudentInteraction]
        Mapping of student IDs to their interaction sequences, used to preserve
        original problem IDs and interaction counts.
    lengths : npt.NDArray[np.int32]
        Vector of interaction sequence lengths for each student.
    student_ids : list[str]
        Student identifiers matching rows of correctness matrix.
    problem_ids : list[str]
        Problem identifiers matching columns of correctness matrix.
    groups : Optional[np.ndarray], default None
        Optional group array (e.g., student group assignments or problem difficulty groups).
    group_2_index : Optional[dict[str, int]], default None
        Optional mapping from group ID to index in groups array.
    """

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
    # check if dataframe is empty
    if data.empty:
        raise ValueError("Input data is empty.")

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
    col_mapping: Optional[Mapping[str, str]] = None,
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
    col_mapping: Optional[
        Union[
            Mapping[ColumnNames, str],
            Mapping[str, str],
            Mapping[ColumnNames | str, str],
        ]
    ] = None,
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

    col_mapping = ColumnNames.apply_default_mapping(col_mapping)

    validate_data(
        data=data,
        col_mapping=col_mapping,
        check_groups=return_groups,
    )
    student_col = col_mapping.get(ColumnNames.STUDENT_ID)
    problem_col = col_mapping.get(ColumnNames.PROBLEM_ID)
    correctness_col = col_mapping.get(ColumnNames.CORRECTNESS)
    order_col = col_mapping.get(ColumnNames.ORDER)
    kc_column = col_mapping.get(ColumnNames.KC_ID)

    # if no kc column in data, add a default kc column
    working_data = data
    if data.get(kc_column) is None:
        working_data = data.copy()
        working_data[kc_column] = _DEFAULT_KC_ID

    working_data[student_col] = working_data[student_col].astype(str)
    working_data[kc_column] = working_data[kc_column].astype(str)

    # ignore rows without ordering
    working_data = working_data[working_data[order_col].notna()].copy()

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
        duplicate_order_rows = subset.duplicated(
            subset=[student_col, order_col], keep=False
        )
        if duplicate_order_rows.any():
            duplicate_view = subset.loc[duplicate_order_rows, [student_col, order_col]]
            preview = duplicate_view.drop_duplicates().head(5).to_dict("records")
            raise ValueError(
                f"Duplicate ORDER values found within student for KC '{kc}'. "
                f"Examples: {preview}"
            )

        try:
            subset = subset.sort_values([student_col, order_col], kind="mergesort")
        except TypeError as exc:
            raise ValueError(
                f"ORDER column '{order_col}' could not be sorted. "
                "Ensure values are comparable."
            ) from exc

        student_ids: list[str] = sorted(
            subset[student_col].unique().tolist(), key=natsort_keygen()
        )

        observed_subset = subset.loc[subset[correctness_col].notna()]
        observed_by_student: dict[str, pd.DataFrame] = {
            str(student_id): student_rows
            for student_id, student_rows in observed_subset.groupby(
                student_col, sort=False, observed=True
            )
        }

        student_inter_dict: dict[str, StudentInteraction] = {}
        sequences_with_lens: list[tuple[np.ndarray, int]] = []
        max_len = 0
        for student_id in student_ids:
            observed_rows = observed_by_student.get(student_id)
            if observed_rows is None:
                sequence = np.empty(0, dtype=np.int8)
                attempted_problem_ids: list[str] = []
                seq_len = 0
            else:
                sequence = observed_rows[correctness_col].to_numpy(dtype=np.int8)
                attempted_problem_ids = observed_rows[problem_col].astype(str).tolist()
                seq_len = len(attempted_problem_ids)

            sequences_with_lens.append((sequence, seq_len))
            student_inter_dict[student_id] = StudentInteraction(
                problem_ids=attempted_problem_ids,
                length=seq_len,
            )
            if seq_len > max_len:
                max_len = seq_len

        correctness_array = np.full(
            (len(student_ids), max_len), _NA_FILL_VALUE, dtype=np.int8
        )
        for i, (sequence, seq_len) in enumerate(sequences_with_lens):
            if seq_len:
                correctness_array[i, :seq_len] = sequence

        lengths: npt.NDArray[np.int32] = np.array(
            [seq_len for _, seq_len in sequences_with_lens], dtype=np.int32
        )

        problem_ids: list[str] = [str(i) for i in range(1, max_len + 1)]
        kc_data_dict: dict[str, Any] = {
            "correctness": correctness_array,
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
            student_group_df: pd.DataFrame = student_group_df.loc[student_ids]
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
    """Check if object is a dict with specific key and value types.

    Parameters
    ----------
    x : object
        Object to validate.
    key_type : type, default str
        Expected type for all dictionary keys.
    value_type : type, default pd.DataFrame
        Expected type for all dictionary values.

    Returns
    -------
    bool
        True if x is a dict with all keys and values matching the specified types.
    """
    if not isinstance(x, dict):
        return False

    return all(
        isinstance(k, key_type) and isinstance(v, value_type) for k, v in x.items()
    )
