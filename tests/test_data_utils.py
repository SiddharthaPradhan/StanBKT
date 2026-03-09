import numpy as np
import pandas as pd
import pytest

from stanbkt.utils.data_utils import (
    ColumnNames,
    format_data,
    iter_kc_data,
    rename_summary_var_columns,
    summarize_state_predictions,
    summarize_state_predictions_test,
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
    assert set(kc_data.group_2_index.keys()) == {"g1", "g2"}
    assert sorted(np.unique(kc_data.groups).tolist()) == [1, 2]


def test_iter_kc_data_group_equals_student() -> None:
    df = _base_df().drop(columns=["group_id"])
    col_mapping = ColumnNames.get_default_mapping()
    col_mapping[ColumnNames.GROUP] = ColumnNames.STUDENT_ID

    assert dict(iter_kc_data(df, col_mapping=col_mapping, return_groups=True))


def test_iter_kc_data_warns_and_drops_rows_with_null_pivot_values() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2"],
            "problem_id": ["p1", "p2", "p1"],
            "correct": [1, 0, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a"],
            "group_id": ["g1", "g1", "g2"],
        }
    )
    calls: list[tuple[str, VerbosityLevel]] = []

    def recorder(message: str, level: VerbosityLevel) -> None:
        calls.append((message, level))

    result = dict(iter_kc_data(df, return_groups=False, print_fn=recorder))
    kc_data = result["kc_a"]

    assert calls
    assert "null values detected" in calls[0][0]
    assert calls[0][1] == VerbosityLevel.INFO
    assert kc_data.correctness.shape == (1, 2)


def test_format_data_returns_same_kc_structure() -> None:
    df = _base_df()
    formatted = format_data(df)

    assert list(formatted.keys()) == ["kc_a"]
    assert formatted["kc_a"].correctness.shape == (2, 2)


def test_summarize_state_predictions_filters_metadata_and_sorts() -> None:
    gq_df = pd.DataFrame(
        {
            "draw_1": [0.2, 0.8, 99.0],
            "draw_2": [0.3, 0.7, 99.0],
        },
        index=pd.Index(["pred[2,1]", "pred[1,2]", "chain__"], dtype="object"),
    )

    out = summarize_state_predictions(gq_df)

    assert list(out.columns) == ["mean", "std", "median", "2.5%", "97.5%"]
    assert list(out.index) == ["pred[1,2]", "pred[2,1]"]


def test_rename_summary_var_columns_renames_columns() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = rename_summary_var_columns(df, ["mean", "std"])

    assert list(out.columns) == ["mean", "std"]


def test_rename_summary_var_columns_raises_on_length_mismatch() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Length of expected_var_cols"):
        rename_summary_var_columns(df, ["mean"])


def test_summarize_state_predictions_test_input_validation() -> None:
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        summarize_state_predictions_test(pd.DataFrame())

    gq_df = pd.DataFrame(
        {"d1": [0.5]},
        index=pd.Index(["pred[1,1]"], dtype="object"),
    )
    with pytest.raises(ValueError, match="Quantiles must be between 0 and 1"):
        summarize_state_predictions_test(gq_df, quantiles=(-0.1, 0.5))


# ---------------------------------------------------------------------------
# Additional validate_data coverage
# ---------------------------------------------------------------------------


def test_validate_data_with_check_groups_passes_when_group_id_present() -> None:
    df = _base_df()
    validate_data(df, ColumnNames.get_default_mapping(), check_groups=True)  # should not raise


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


# ---------------------------------------------------------------------------
# Additional format_data coverage
# ---------------------------------------------------------------------------


def test_format_data_with_return_groups_populates_groups() -> None:
    df = _base_df()
    formatted = format_data(df, return_groups=True)
    kc_data = formatted["kc_a"]
    assert kc_data.groups is not None
    assert kc_data.group_2_index is not None


def test_format_data_with_return_groups_correct_mapping() -> None:
    df = _base_df()
    formatted = format_data(df, return_groups=True)
    kc_data = formatted["kc_a"]
    assert set(kc_data.group_2_index.keys()) == {"g1", "g2"}


# ---------------------------------------------------------------------------
# Additional summarize_state_predictions coverage
# ---------------------------------------------------------------------------


def test_summarize_state_predictions_returns_empty_for_only_metadata_rows() -> None:
    """All rows filtered out (only chain__/iter__/draw__ rows) → empty output."""
    gq_df = pd.DataFrame(
        {"draw_1": [99.0, 99.0], "draw_2": [99.0, 99.0]},
        index=pd.Index(["chain__", "iter__"], dtype="object"),
    )
    out = summarize_state_predictions(gq_df)
    assert len(out) == 0


def test_summarize_state_predictions_rows_without_index_pattern_appended_last() -> None:
    """Rows whose index doesn't match pred[i,j] pattern are appended after sorted rows."""
    gq_df = pd.DataFrame(
        {"draw_1": [0.1, 0.2, 0.3], "draw_2": [0.1, 0.2, 0.3]},
        index=pd.Index(["pred[2,1]", "pred[1,1]", "no_pattern"], dtype="object"),
    )
    out = summarize_state_predictions(gq_df)
    # Sorted rows first, then unmatched
    assert list(out.index) == ["pred[1,1]", "pred[2,1]", "no_pattern"]


# ---------------------------------------------------------------------------
# summarize_state_predictions_test — additional validation paths
# ---------------------------------------------------------------------------


def test_summarize_state_predictions_test_raises_for_too_many_index_names() -> None:
    gq_df = pd.DataFrame(
        {"d1": [0.5]},
        index=pd.Index(["pred[1,1]"], dtype="object"),
    )
    with pytest.raises(ValueError, match="array_index_names cannot have more than 3"):
        summarize_state_predictions_test(
            gq_df, array_index_names=["a", "b", "c", "d"]
        )
