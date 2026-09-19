import numpy as np
import pandas as pd
import pytest

from stanbkt.utils.data_utils import (
    ColumnNames,
    format_kc_data,
    iter_kc_data,
    prepare_student_covariates,
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
            "timestamp": [1, 2, 1, 2],
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
            "timestamp": [1, 2, 1, 2],
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
            "timestamp": [1, 2, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a"],
        }
    )

    result = dict(iter_kc_data(df))
    kc_data = result["kc_a"]

    assert kc_data.correctness.dtype == np.int8
    assert kc_data.correctness.tolist() == [[1, 0], [1, -1]]
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
            "timestamp": [1, 1, 2, 2],
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


def test_iter_kc_data_orders_student_sequences_by_order_column() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p10", "p2", "p1", "p3"],
            "correct": [0, 1, 1, 0],
            "timestamp": [2, 1, 2, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a", "kc_a"],
        }
    )

    kc_data = dict(iter_kc_data(df))["kc_a"]

    assert kc_data.correctness.tolist() == [[1, 0], [0, 1]]
    assert kc_data.student_inter_dict["s1"].problem_ids == ["p2", "p10"]
    assert kc_data.student_inter_dict["s2"].problem_ids == ["p3", "p1"]


def test_iter_kc_data_ignores_rows_with_missing_order() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2"],
            "problem_id": ["p1", "p2", "p3"],
            "correct": [1, 0, 1],
            "timestamp": [1, np.nan, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a"],
        }
    )

    kc_data = dict(iter_kc_data(df))["kc_a"]

    assert kc_data.correctness.tolist() == [[1], [1]]
    assert kc_data.student_inter_dict["s1"].problem_ids == ["p1"]
    assert kc_data.student_inter_dict["s2"].problem_ids == ["p3"]


def test_iter_kc_data_raises_on_duplicate_order_within_student() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1"],
            "problem_id": ["p1", "p2"],
            "correct": [1, 0],
            "timestamp": [1, 1],
            "kc_id": ["kc_a", "kc_a"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate ORDER values"):
        list(iter_kc_data(df))


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


# ---------------------------------------------------------------------------
# prepare_student_covariates / iter_kc_data covariate attachment
# ---------------------------------------------------------------------------


def _covariates_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["s1", "s2"],
            "pretest": [0.5, -1.0],
            "age": [10.0, 12.0],
        }
    )


def test_prepare_student_covariates_preserves_column_order() -> None:
    df = _covariates_df()[["student_id", "age", "pretest"]]
    indexed, columns = prepare_student_covariates(df, ["s1", "s2"])
    assert columns == ["age", "pretest"]
    assert list(indexed.index) == ["s1", "s2"]


def test_prepare_student_covariates_warns_and_dedups_on_duplicate_ids() -> None:
    df = pd.concat(
        [_covariates_df(), pd.DataFrame({"student_id": ["s1"], "pretest": [9.0], "age": [99.0]})],
        ignore_index=True,
    )
    with pytest.warns(UserWarning, match="duplicate student IDs"):
        indexed, _ = prepare_student_covariates(df, ["s1", "s2"])
    assert len(indexed) == 2
    assert indexed.loc["s1", "pretest"] == 0.5


def test_prepare_student_covariates_raises_for_missing_student() -> None:
    with pytest.raises(ValueError, match="s3"):
        prepare_student_covariates(_covariates_df(), ["s1", "s2", "s3"])


def test_prepare_student_covariates_raises_for_missing_id_column() -> None:
    df = _covariates_df().rename(columns={"student_id": "sid"})
    with pytest.raises(ValueError, match="missing the student ID column"):
        prepare_student_covariates(df, ["s1"])


def test_iter_kc_data_attaches_covariates_in_student_order() -> None:
    df = _base_df()
    indexed, columns = prepare_student_covariates(_covariates_df(), ["s1", "s2"])
    kc_data = format_kc_data(df, student_covariates=indexed, covariate_columns=columns)[
        "kc_a"
    ]
    assert kc_data.covariate_columns == ["pretest", "age"]
    assert kc_data.covariates is not None
    assert kc_data.covariates.shape == (2, 2)
    np.testing.assert_allclose(kc_data.covariates[0], [0.5, 10.0])
    np.testing.assert_allclose(kc_data.covariates[1], [-1.0, 12.0])


def test_iter_kc_data_attaches_only_kc_students() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 0, 1],
            "timestamp": [1, 2, 1, 2],
            "kc_id": ["kc_a", "kc_a", "kc_b", "kc_b"],
        }
    )
    indexed, columns = prepare_student_covariates(_covariates_df(), ["s1", "s2"])
    formatted = format_kc_data(df, student_covariates=indexed, covariate_columns=columns)
    assert formatted["kc_a"].covariates.shape == (1, 2)
    np.testing.assert_allclose(formatted["kc_b"].covariates[0], [-1.0, 12.0])


def test_iter_kc_data_without_covariates_leaves_fields_none() -> None:
    kc_data = format_kc_data(_base_df())["kc_a"]
    assert kc_data.covariates is None
    assert kc_data.covariate_columns is None


def test_prepare_student_covariates_raises_for_nan() -> None:
    df = _covariates_df()
    df.loc[1, "pretest"] = np.nan
    with pytest.raises(ValueError, match=r"NaN or infinite.*pretest.*s2"):
        prepare_student_covariates(df, ["s1", "s2"])


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_prepare_student_covariates_raises_for_inf(bad) -> None:
    df = _covariates_df()
    df.loc[0, "age"] = bad
    with pytest.raises(ValueError, match=r"NaN or infinite.*age.*s1"):
        prepare_student_covariates(df, ["s1", "s2"])


def test_prepare_student_covariates_names_only_bad_columns_and_caps_students() -> None:
    students = [f"s{i}" for i in range(8)]
    df = pd.DataFrame(
        {"student_id": students, "x": np.arange(8.0), "y": np.nan}
    )
    with pytest.raises(ValueError, match=r"\['y'\].*and more") as err:
        prepare_student_covariates(df, students)
    assert "'x'" not in str(err.value)


def test_prepare_student_covariates_ignores_bad_rows_for_unrequired_students() -> None:
    df = pd.concat(
        [_covariates_df(), pd.DataFrame({"student_id": ["extra"], "pretest": [np.nan], "age": [1.0]})],
        ignore_index=True,
    )
    indexed, columns = prepare_student_covariates(df, ["s1", "s2"])
    assert columns == ["pretest", "age"]
    assert "extra" in indexed.index


def test_prepare_student_covariates_raises_for_non_numeric() -> None:
    df = _covariates_df().assign(group=["a", "b"], flag=[True, False])
    with pytest.raises(ValueError, match=r"non-numeric.*\['group'\]"):
        prepare_student_covariates(df, ["s1", "s2"])


def test_prepare_student_covariates_accepts_bool_and_nullable_ints() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s2"],
            "flag": [True, False],
            "n": pd.array([1, 2], dtype="Int64"),
        }
    )
    _, columns = prepare_student_covariates(df, ["s1", "s2"])
    assert columns == ["flag", "n"]


def test_prepare_student_covariates_raises_for_missing_nullable_value() -> None:
    df = pd.DataFrame(
        {"student_id": ["s1", "s2"], "n": pd.array([1, None], dtype="Int64")}
    )
    with pytest.raises(ValueError, match=r"NaN or infinite.*s2"):
        prepare_student_covariates(df, ["s1", "s2"])


def test_prepare_student_covariates_raises_for_zero_covariate_columns() -> None:
    df = _covariates_df()[["student_id"]]
    with pytest.raises(ValueError, match="at least one covariate column"):
        prepare_student_covariates(df, ["s1", "s2"])


def test_prepare_student_covariates_honors_custom_id_column() -> None:
    df = _covariates_df().rename(columns={"student_id": "sid"})
    indexed, columns = prepare_student_covariates(df, ["s1", "s2"], student_id_col="sid")
    assert columns == ["pretest", "age"]
    assert list(indexed.index) == ["s1", "s2"]


def test_prepare_student_covariates_matches_int_and_str_ids() -> None:
    df = pd.DataFrame({"student_id": [1, 2], "x": [0.1, 0.2]})
    indexed, _ = prepare_student_covariates(df, ["1", "2"])
    assert list(indexed.index) == ["1", "2"]
    indexed, _ = prepare_student_covariates(df.assign(student_id=["1", "2"]), [1, 2])
    assert list(indexed.index) == ["1", "2"]


# iter_kc_data: vectorized implementation against a straightforward reference


def _reference_kc_data(df, return_groups=False, group_col="group_id"):
    """Plain python reference for iter_kc_data, one KC at a time."""
    from natsort import natsorted

    d = df.copy()
    if "kc_id" not in d.columns:
        d["kc_id"] = "default_kc"
    d["student_id"] = d["student_id"].astype(str)
    d["kc_id"] = d["kc_id"].astype(str)
    d = d[d["timestamp"].notna()]
    out = {}
    for kc in d["kc_id"].unique():
        sub = d[d["kc_id"] == kc].sort_values(["student_id", "timestamp"], kind="mergesort")
        students = natsorted(sub["student_id"].unique().tolist())
        rows = {s: sub[sub["student_id"] == s] for s in students}
        lengths = np.array([len(rows[s]) for s in students], dtype=np.int32)
        max_len = int(lengths.max())
        correctness = np.full((len(students), max_len), -1, dtype=np.int8)
        for i, s in enumerate(students):
            correctness[i, : lengths[i]] = rows[s]["correct"].to_numpy(dtype=np.int8)
        entry = {
            "correctness": correctness,
            "lengths": lengths,
            "student_ids": students,
            "problem_ids": [str(i) for i in range(1, max_len + 1)],
            "interactions": {
                s: (rows[s]["problem_id"].astype(str).tolist(), len(rows[s]))
                for s in students
            },
        }
        if return_groups:
            first = {s: rows[s][group_col].iloc[0] for s in students}
            uniques = list(dict.fromkeys(first[s] for s in students))
            entry["groups"] = np.array(
                [uniques.index(first[s]) + 1 for s in students], dtype=np.int32
            )
            entry["group_2_index"] = {g: i for i, g in enumerate(uniques, 1)}
        out[kc] = entry
    return out


def _assert_matches_reference(df, return_groups=False, group_col="group_id", **kwargs):
    got = dict(iter_kc_data(df, return_groups=return_groups, **kwargs))
    ref = _reference_kc_data(df, return_groups=return_groups, group_col=group_col)
    assert list(got) == list(ref)
    for kc, expected in ref.items():
        actual = got[kc]
        assert actual.correctness.dtype == np.int8
        np.testing.assert_array_equal(actual.correctness, expected["correctness"])
        assert actual.lengths.dtype == np.int32
        np.testing.assert_array_equal(actual.lengths, expected["lengths"])
        assert actual.student_ids == expected["student_ids"]
        assert actual.problem_ids == expected["problem_ids"]
        assert list(actual.student_inter_dict) == expected["student_ids"]
        for s, (problem_ids, length) in expected["interactions"].items():
            assert actual.student_inter_dict[s].problem_ids == problem_ids
            assert actual.student_inter_dict[s].length == length
        if return_groups:
            assert actual.groups.dtype == np.int32
            np.testing.assert_array_equal(actual.groups, expected["groups"])
            assert actual.group_2_index == expected["group_2_index"]
    return got


def _random_interactions(seed, n_students=12, kcs=("kc_a", "kc_b", "kc_c")):
    rng = np.random.default_rng(seed)
    student_group = {f"s{i}": f"g{rng.integers(1, 4)}" for i in range(1, n_students + 1)}
    rows = []
    for kc in kcs:
        for student, group in student_group.items():
            if rng.random() < 0.25:
                continue
            n = int(rng.integers(1, 9))
            orders = rng.permutation(np.arange(1, 30))[:n]
            for order in orders:
                rows.append(
                    {
                        "student_id": student,
                        "problem_id": f"p{rng.integers(1, 6)}",
                        "correct": int(rng.integers(0, 2)),
                        "timestamp": int(order),
                        "kc_id": kc,
                        "group_id": group,
                    }
                )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(5))
def test_iter_kc_data_matches_reference_on_random_data(seed) -> None:
    df = _random_interactions(seed)
    _assert_matches_reference(df)
    _assert_matches_reference(df, return_groups=True)


def test_iter_kc_data_row_order_does_not_matter() -> None:
    df = _random_interactions(3)
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = dict(iter_kc_data(df))
    b = dict(iter_kc_data(shuffled))
    assert set(a) == set(b)
    for kc in a:
        np.testing.assert_array_equal(a[kc].correctness, b[kc].correctness)
        assert a[kc].student_ids == b[kc].student_ids
        assert a[kc].student_inter_dict == b[kc].student_inter_dict


def test_iter_kc_data_student_ids_use_natural_sort() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s10", "s2", "s1", "s10", "s2", "s1"],
            "problem_id": ["p1"] * 6,
            "correct": [1, 0, 1, 0, 1, 1],
            "timestamp": [1, 1, 1, 2, 2, 2],
        }
    )
    got = dict(iter_kc_data(df))["default_kc"]
    assert got.student_ids == ["s1", "s2", "s10"]
    np.testing.assert_array_equal(
        got.correctness, np.array([[1, 1], [0, 1], [1, 0]], dtype=np.int8)
    )


def test_iter_kc_data_kcs_yield_in_order_of_appearance() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s1"],
            "problem_id": ["p1", "p1", "p1"],
            "correct": [1, 0, 1],
            "timestamp": [1, 2, 3],
            "kc_id": ["kc_z", "kc_a", "kc_m"],
        }
    )
    assert [kc for kc, _ in iter_kc_data(df)] == ["kc_z", "kc_a", "kc_m"]


def test_iter_kc_data_drops_rows_with_missing_order() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s1", "s2", "s2"],
            "problem_id": ["p1", "p2", "p3", "p1", "p2"],
            "correct": [1, 0, 1, 0, 1],
            "timestamp": [1.0, np.nan, 2.0, 1.0, 2.0],
        }
    )
    got = dict(iter_kc_data(df))["default_kc"]
    np.testing.assert_array_equal(got.lengths, [2, 2])
    assert got.student_inter_dict["s1"].problem_ids == ["p1", "p3"]
    np.testing.assert_array_equal(got.correctness, [[1, 1], [0, 1]])
    _assert_matches_reference(df)


def test_iter_kc_data_student_with_all_order_missing_is_absent() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s2", "s2", "s3"],
            "problem_id": ["p1", "p2", "p1", "p2", "p1"],
            "correct": [1, 0, 0, 1, 1],
            "timestamp": [1.0, 2.0, np.nan, np.nan, 1.0],
        }
    )
    got = dict(iter_kc_data(df))["default_kc"]
    assert got.student_ids == ["s1", "s3"]
    assert "s2" not in got.student_inter_dict
    np.testing.assert_array_equal(got.lengths, [2, 1])
    _assert_matches_reference(df)


def test_iter_kc_data_all_order_missing_yields_nothing() -> None:
    df = _base_df().assign(timestamp=np.nan)
    assert list(iter_kc_data(df)) == []


def test_iter_kc_data_student_missing_order_in_one_kc_only() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s2", "s1", "s2"],
            "problem_id": ["p1", "p1", "p1", "p1"],
            "correct": [1, 0, 1, 1],
            "timestamp": [1.0, np.nan, 1.0, 1.0],
            "kc_id": ["kc_a", "kc_a", "kc_b", "kc_b"],
        }
    )
    got = dict(iter_kc_data(df))
    assert got["kc_a"].student_ids == ["s1"]
    assert got["kc_b"].student_ids == ["s1", "s2"]


def test_iter_kc_data_integer_ids_become_strings() -> None:
    df = pd.DataFrame(
        {
            "student_id": [10, 10, 2, 2],
            "problem_id": [7, 8, 7, 8],
            "correct": [1, 0, 0, 1],
            "timestamp": [1, 2, 1, 2],
            "kc_id": [5, 5, 5, 5],
        }
    )
    got = dict(iter_kc_data(df))
    assert list(got) == ["5"]
    assert got["5"].student_ids == ["2", "10"]
    assert got["5"].student_inter_dict["10"].problem_ids == ["7", "8"]
    _assert_matches_reference(df)


def test_iter_kc_data_without_kc_column_uses_default_kc() -> None:
    df = _base_df().drop(columns=["kc_id"])
    got = dict(iter_kc_data(df))
    assert list(got) == ["default_kc"]
    _assert_matches_reference(df)


def test_iter_kc_data_ragged_sequences_are_padded_with_na_value() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"] * 4 + ["s2"] * 2 + ["s3"],
            "problem_id": ["p1", "p2", "p3", "p4", "p1", "p2", "p1"],
            "correct": [1, 0, 1, 1, 0, 0, 1],
            "timestamp": [1, 2, 3, 4, 1, 2, 1],
        }
    )
    got = dict(iter_kc_data(df))["default_kc"]
    expected = np.array(
        [[1, 0, 1, 1], [0, 0, -1, -1], [1, -1, -1, -1]], dtype=np.int8
    )
    np.testing.assert_array_equal(got.correctness, expected)
    np.testing.assert_array_equal(got.lengths, [4, 2, 1])
    assert got.problem_ids == ["1", "2", "3", "4"]
    assert got.student_inter_dict["s3"].problem_ids == ["p1"]
    assert got.student_inter_dict["s2"].length == 2


def test_iter_kc_data_students_in_only_some_kcs() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s2", "s3", "s1", "s3"],
            "problem_id": ["p1"] * 5,
            "correct": [1, 0, 1, 0, 1],
            "timestamp": [1, 1, 1, 1, 1],
            "kc_id": ["kc_a", "kc_a", "kc_a", "kc_b", "kc_b"],
        }
    )
    got = dict(iter_kc_data(df))
    assert got["kc_a"].student_ids == ["s1", "s2", "s3"]
    assert got["kc_b"].student_ids == ["s1", "s3"]
    np.testing.assert_array_equal(got["kc_b"].correctness, [[0], [1]])


def test_iter_kc_data_groups_are_one_based_by_first_appearance() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s10", "s2", "s1"],
            "problem_id": ["p1"] * 3,
            "correct": [1, 0, 1],
            "timestamp": [1, 1, 1],
            "group_id": ["gB", "gA", "gB"],
        }
    )
    got = dict(iter_kc_data(df, return_groups=True))["default_kc"]
    assert got.student_ids == ["s1", "s2", "s10"]
    assert got.group_2_index == {"gB": 1, "gA": 2}
    np.testing.assert_array_equal(got.groups, [1, 2, 1])
    assert got.groups.dtype == np.int32


def test_iter_kc_data_individualized_groups_when_group_column_is_student() -> None:
    df = _random_interactions(1)
    mapping = {ColumnNames.GROUP: "student_id"}
    got = dict(iter_kc_data(df, col_mapping=mapping, return_groups=True))
    ref = _reference_kc_data(df, return_groups=True, group_col="student_id")
    for kc, expected in ref.items():
        np.testing.assert_array_equal(got[kc].groups, expected["groups"])
        assert got[kc].group_2_index == expected["group_2_index"]
        assert got[kc].groups.tolist() == list(range(1, len(got[kc].student_ids) + 1))


def test_iter_kc_data_integer_groups_keep_raw_values() -> None:
    df = _base_df().assign(group_id=[3, 3, 1, 1])
    got = dict(iter_kc_data(df, return_groups=True))["kc_a"]
    assert got.group_2_index == {3: 1, 1: 2}
    np.testing.assert_array_equal(got.groups, [1, 2])


@pytest.mark.parametrize(
    "student_dtype",
    ["object", "int64", "category"],
)
def test_iter_kc_data_does_not_mutate_input(student_dtype) -> None:
    df = pd.DataFrame(
        {
            "student_id": [3, 3, 1, 1],
            "problem_id": ["p1", "p2", "p1", "p2"],
            "correct": [1, 0, 0, 1],
            "timestamp": [1, 2, 1, 2],
            "kc_id": [9, 9, 9, 9],
            "group_id": ["g1", "g1", "g2", "g2"],
        }
    )
    if student_dtype == "object":
        df["student_id"] = df["student_id"].astype(str).astype(object)
    elif student_dtype == "category":
        df["student_id"] = pd.Categorical(
            df["student_id"].astype(str), categories=["1", "3", "unused"]
        )
    before = df.copy(deep=True)
    dtypes_before = df.dtypes.copy()

    list(iter_kc_data(df, return_groups=True))

    pd.testing.assert_frame_equal(df, before)
    assert df.dtypes.equals(dtypes_before)
    assert list(df.columns) == list(before.columns)


def test_iter_kc_data_does_not_mutate_input_without_kc_column() -> None:
    df = _base_df().drop(columns=["kc_id"])
    before = df.copy(deep=True)
    list(iter_kc_data(df))
    pd.testing.assert_frame_equal(df, before)
    assert "kc_id" not in df.columns


def test_iter_kc_data_repeated_calls_give_same_result() -> None:
    df = _random_interactions(2)
    first = dict(iter_kc_data(df, return_groups=True))
    second = dict(iter_kc_data(df, return_groups=True))
    for kc in first:
        np.testing.assert_array_equal(first[kc].correctness, second[kc].correctness)
        assert first[kc].student_ids == second[kc].student_ids


def test_iter_kc_data_categorical_student_column_with_unused_categories() -> None:
    df = _base_df()
    df["student_id"] = pd.Categorical(
        df["student_id"], categories=["s0", "s1", "s2", "s3"]
    )
    got = dict(iter_kc_data(df))["kc_a"]
    assert got.student_ids == ["s1", "s2"]
    np.testing.assert_array_equal(got.lengths, [2, 2])


def test_iter_kc_data_duplicate_order_within_student_raises() -> None:
    df = _base_df()
    df.loc[1, "timestamp"] = 1
    with pytest.raises(ValueError, match="Duplicate ORDER values"):
        list(iter_kc_data(df))


def test_iter_kc_data_same_order_for_different_students_is_fine() -> None:
    got = dict(iter_kc_data(_base_df()))["kc_a"]
    assert got.student_ids == ["s1", "s2"]


@pytest.mark.parametrize(
    "order_values",
    [
        [0.5, 1.5, 0.5, 1.5],
        pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
        ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
    ],
)
def test_iter_kc_data_supports_other_order_types(order_values) -> None:
    df = _base_df().assign(timestamp=order_values)
    got = dict(iter_kc_data(df))["kc_a"]
    np.testing.assert_array_equal(got.correctness, [[1, 0], [0, 1]])


def test_iter_kc_data_boolean_correctness_becomes_int8() -> None:
    df = _base_df().assign(correct=[True, False, False, True])
    got = dict(iter_kc_data(df))["kc_a"]
    assert got.correctness.dtype == np.int8
    np.testing.assert_array_equal(got.correctness, [[1, 0], [0, 1]])


def test_iter_kc_data_custom_column_mapping() -> None:
    df = _base_df().rename(
        columns={
            "student_id": "user",
            "problem_id": "item",
            "correct": "ok",
            "timestamp": "t",
            "kc_id": "skill",
        }
    )
    mapping = {
        ColumnNames.STUDENT_ID: "user",
        ColumnNames.PROBLEM_ID: "item",
        ColumnNames.CORRECTNESS: "ok",
        ColumnNames.ORDER: "t",
        ColumnNames.KC_ID: "skill",
    }
    got = dict(iter_kc_data(df, col_mapping=mapping))["kc_a"]
    assert got.student_ids == ["s1", "s2"]
    np.testing.assert_array_equal(got.correctness, [[1, 0], [0, 1]])


def test_iter_kc_data_covariates_follow_natural_student_order_per_kc() -> None:
    df = _random_interactions(4)
    students = sorted({f"s{i}" for i in range(1, 13)})
    covariates = pd.DataFrame(
        {
            "pretest": np.arange(len(students), dtype=float),
            "age": np.arange(len(students), dtype=float) * 2,
        },
        index=students,
    )
    got = dict(
        iter_kc_data(
            df, student_covariates=covariates, covariate_columns=["age", "pretest"]
        )
    )
    for kc, kc_data in got.items():
        expected = covariates.loc[kc_data.student_ids, ["age", "pretest"]].to_numpy()
        np.testing.assert_array_equal(kc_data.covariates, expected)
        assert kc_data.covariate_columns == ["age", "pretest"]


def test_iter_kc_data_covariates_missing_student_raises() -> None:
    df = _base_df()
    covariates = pd.DataFrame({"pretest": [0.1]}, index=["s1"])
    with pytest.raises(ValueError, match="no matching row"):
        list(
            iter_kc_data(
                df, student_covariates=covariates, covariate_columns=["pretest"]
            )
        )


def test_iter_kc_data_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        list(iter_kc_data(_base_df().iloc[0:0]))
