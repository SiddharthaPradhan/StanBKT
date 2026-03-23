import numpy as np
import pandas as pd
import pytest

from stanbkt.utils.data_utils import (
    ColumnNames,
    format_kc_data,
    iter_kc_data,
    rename_summary_var_columns,
    validate_data,
)
from stanbkt.utils.verbose import VerbosityLevel


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 0, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a", "kc_a"],
            "group_id": ["g1", "g1", "g2", "g2"],
        }
    )


def test_validate_data_accepts_valid_input() -> None:
    df = _base_df()
    validate_data(df, ColumnNames.get_default_mapping())


def test_validate_data_raises_for_missing_required_column() -> None:
    df = _base_df().drop(columns=["problem_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df, ColumnNames.get_default_mapping())


def test_validate_data_raises_for_non_binary_correctness() -> None:
    df = _base_df().copy()
    df.loc[0, "correct"] = 2
    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        validate_data(df, ColumnNames.get_default_mapping())


def test_validate_data_with_additional_required_cols() -> None:
    df = _base_df()
    # Should pass with extra column present
    validate_data(
        df, ColumnNames.get_default_mapping(), additional_required_cols={"group_id"}
    )


def test_validate_data_raises_when_additional_col_missing() -> None:
    df = _base_df().drop(columns=["group_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(
            df, ColumnNames.get_default_mapping(), additional_required_cols={"group_id"}
        )


def test_iter_kc_data_adds_default_kc_when_absent() -> None:
    df = _base_df().drop(columns=["kc_id"])
    result = list(iter_kc_data(df))

    assert len(result) == 1
    kc_name, kc_data = result[0]
    assert kc_name == "default_kc"
    assert kc_data.correctness.shape == (2, 2)
    assert kc_data.correctness.dtype == np.int8


def test_iter_kc_data_builds_group_indices_and_mapping() -> None:
    df = _base_df()
    result = dict(iter_kc_data(df, return_groups=True))
    kc_data = result["kc_a"]

    assert kc_data.groups is not None
    assert kc_data.group_2_index is not None
    assert kc_data.groups.dtype == np.int32
    assert kc_data.group_2_index is not None
    assert set(kc_data.group_2_index.keys()) == {"g1", "g2"}
    assert sorted(np.unique(kc_data.groups).tolist()) == [1, 2]


def test_iter_kc_data_group_equals_student() -> None:
    df = _base_df().drop(columns=["group_id"])
    col_mapping = ColumnNames.get_default_mapping()
    col_mapping[ColumnNames.GROUP] = ColumnNames.STUDENT_ID

    assert dict(iter_kc_data(df, col_mapping=col_mapping, return_groups=True))


def test_format_data_returns_same_kc_structure() -> None:
    df = _base_df()
    formatted = format_kc_data(df)

    assert list(formatted.keys()) == ["kc_a"]
    assert formatted["kc_a"].correctness.shape == (2, 2)


def test_rename_summary_var_columns_renames_columns() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = rename_summary_var_columns(df, ["mean", "std"])

    assert list(out.columns) == ["mean", "std"]


def test_rename_summary_var_columns_raises_on_length_mismatch() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Length of expected_var_cols"):
        rename_summary_var_columns(df, ["mean"])


# ---------------------------------------------------------------------------
# Additional validate_data coverage
# ---------------------------------------------------------------------------


def test_validate_data_with_check_groups_passes_when_group_id_present() -> None:
    df = _base_df()
    validate_data(
        df, ColumnNames.get_default_mapping(), check_groups=True
    )  # should not raise


def test_validate_data_with_check_groups_raises_when_group_id_absent() -> None:
    df = _base_df().drop(columns=["group_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df, ColumnNames.get_default_mapping(), check_groups=True)


# ---------------------------------------------------------------------------
# Additional iter_kc_data coverage
# ---------------------------------------------------------------------------


def test_iter_kc_data_with_custom_col_mapping_validates_data() -> None:
    """Providing a col_mapping triggers the validate_data path; missing column raises."""
    df = _base_df().drop(columns=["problem_id"])
    col_mapping = ColumnNames.get_default_mapping()
    with pytest.raises(ValueError, match="Missing required columns"):
        list(iter_kc_data(df, col_mapping=col_mapping))


def test_iter_kc_data_yields_multiple_kcs() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 0, 1],
            "kc_id": ["kc_a", "kc_a", "kc_b", "kc_b"],
        }
    )
    result = dict(iter_kc_data(df))
    assert set(result.keys()) == {"kc_a", "kc_b"}


def test_iter_kc_data_kc_keys_are_strings() -> None:
    df = _base_df()
    # Replace kc_id with an integer to confirm the key is cast to str
    df["kc_id"] = 42
    result = list(iter_kc_data(df))
    assert len(result) == 1
    key, _ = result[0]
    assert isinstance(key, str)
    assert key == "42"


def test_iter_kc_data_compacts_nas_and_tracks_original_problem_ids() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2"],
            "problem_id": ["p1", "p3", "p2"],
            "correct": [1, 0, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a"],
        }
    )

    result = dict(iter_kc_data(df))
    kc_data = result["kc_a"]

    assert kc_data.correctness.dtype == np.int8
    assert kc_data.correctness.tolist() == [[1, 0, -1], [1, -1, -1]]
    assert kc_data.student_inter_dict["s1"].problem_ids == ["p1", "p3"]
    assert kc_data.student_inter_dict["s1"].length == 2
    assert kc_data.student_inter_dict["s2"].problem_ids == ["p2"]
    assert kc_data.student_inter_dict["s2"].length == 1


def test_iter_kc_data_keeps_student_interactions_isolated_per_kc() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 1, 1],
            "kc_id": ["kc_a", "kc_b", "kc_a", "kc_b"],
        }
    )

    result = dict(iter_kc_data(df))
    kc_a = result["kc_a"]
    kc_b = result["kc_b"]

    assert kc_a.student_inter_dict["s1"].problem_ids == ["p1"]
    assert kc_a.student_inter_dict["s2"].problem_ids == ["p1"]
    assert kc_b.student_inter_dict["s1"].problem_ids == ["p2"]
    assert kc_b.student_inter_dict["s2"].problem_ids == ["p2"]


# ---------------------------------------------------------------------------
# Additional format_data coverage
# ---------------------------------------------------------------------------


def test_format_data_with_return_groups_populates_groups() -> None:
    df = _base_df()
    formatted = format_kc_data(df, return_groups=True)
    kc_data = formatted["kc_a"]
    assert kc_data.groups is not None
    assert kc_data.group_2_index is not None


def test_format_data_with_return_groups_correct_mapping() -> None:
    df = _base_df()
    formatted = format_kc_data(df, return_groups=True)
    kc_data = formatted["kc_a"]
    assert kc_data.group_2_index is not None
    assert set(kc_data.group_2_index.keys()) == {"g1", "g2"}
