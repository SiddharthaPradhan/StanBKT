import pytest

from stanbkt.utils import sim_simple_BKT, sim_grouped_BKT


def test_sim_simple_bkt_accepts_scalar_parameters() -> None:
    data_df = sim_simple_BKT(
        n_students=3,
        n_problems=4,
        n_kcs=2,
        prior=0.2,
        learn=0.3,
        forget=0.1,
        guess=0.2,
        slip=0.1,
        rng_seed=42,
    )

    assert data_df.shape[0] == 12
    assert set(data_df["kc_id"].unique()) <= {"kc_0", "kc_1"}


def test_sim_simple_bkt_uses_per_kc_list_parameters() -> None:
    data_df = sim_simple_BKT(
        n_students=2,
        n_problems=4,
        n_kcs=2,
        prior=[0.0, 1.0],
        learn=[0.0, 0.0],
        forget=[0.0, 0.0],
        guess=[0.0, 0.0],
        slip=[0.0, 0.0],
        kc_sequence=[0, 1, 0, 1],
        rng_seed=7,
    )

    pivot = data_df.pivot(index="student_id", columns="problem_id", values="correct")
    assert pivot["prob_0"].eq(0).all()
    assert pivot["prob_1"].eq(1).all()
    assert pivot["prob_2"].eq(0).all()
    assert pivot["prob_3"].eq(1).all()


def test_sim_simple_bkt_raises_on_invalid_parameter_list_length() -> None:
    with pytest.raises(ValueError, match="prior must be scalar or length n_kcs"):
        sim_simple_BKT(n_kcs=3, prior=[0.1, 0.2], rng_seed=1)


# =============================================================================
# Tests for sim_grouped_BKT
# =============================================================================


def test_sim_grouped_bkt_accepts_scalar_parameters() -> None:
    """Test that grouped BKT simulation works with scalar parameters (same for all groups)."""
    data_df = sim_grouped_BKT(
        n_students=6,
        n_problems=4,
        n_kcs=2,
        n_groups=2,
        prior=0.2,
        learn=0.3,
        forget=0.1,
        guess=0.2,
        slip=0.1,
        rng_seed=42,
    )

    assert data_df.shape[0] == 24  # 6 students * 4 problems
    assert "group_id" in data_df.columns
    assert set(data_df["group_id"].unique()) == {"group_0", "group_1"}
    assert set(data_df["kc_id"].unique()) <= {"kc_0", "kc_1"}


def test_sim_grouped_bkt_uses_per_group_list_parameters() -> None:
    """Test that grouped BKT uses different parameters for different groups."""
    data_df = sim_grouped_BKT(
        n_students=4,
        n_problems=4,
        n_kcs=1,
        n_groups=2,
        prior=[
            0.0,
            1.0,
        ],  # Group 0 starts with no knowledge, Group 1 starts with full knowledge
        learn=[0.0, 0.0],
        forget=[0.0, 0.0],
        guess=[0.0, 0.0],
        slip=[0.0, 0.0],
        rng_seed=42,
    )

    # Group 0 students (student 0, 2) should get all 0s (no knowledge, no guessing)
    group_0_data = data_df[data_df["group_id"] == "group_0"]
    # Group 1 students (student 1, 3) should get all 1s (full knowledge, no slipping)
    group_1_data = data_df[data_df["group_id"] == "group_1"]

    assert group_0_data["correct"].sum() == 0
    assert group_1_data["correct"].sum() == len(group_1_data)


def test_sim_grouped_bkt_raises_on_invalid_parameter_list_length() -> None:
    """Test that invalid parameter list lengths raise appropriate errors."""
    with pytest.raises(ValueError, match=r"prior must be scalar, shape \(n_groups,\),"):
        sim_grouped_BKT(n_groups=3, prior=[0.1, 0.2], rng_seed=1)


def test_sim_grouped_bkt_uses_group_kc_matrix_parameters() -> None:
    """Test that grouped BKT supports parameters that vary by both group and KC."""
    data_df = sim_grouped_BKT(
        n_students=2,
        n_problems=4,
        n_kcs=2,
        n_groups=2,
        prior=[[0.0, 1.0], [1.0, 0.0]],
        learn=0.0,
        forget=0.0,
        guess=0.0,
        slip=0.0,
        kc_sequence=[0, 1, 0, 1],
        group_sequence=[0, 1],
        rng_seed=42,
    )

    pivot = data_df.pivot(index="student_id", columns="problem_id", values="correct")
    assert pivot.loc["stu_0", "prob_0"] == 0
    assert pivot.loc["stu_0", "prob_1"] == 1
    assert pivot.loc["stu_0", "prob_2"] == 0
    assert pivot.loc["stu_0", "prob_3"] == 1
    assert pivot.loc["stu_1", "prob_0"] == 1
    assert pivot.loc["stu_1", "prob_1"] == 0
    assert pivot.loc["stu_1", "prob_2"] == 1
    assert pivot.loc["stu_1", "prob_3"] == 0


def test_sim_grouped_bkt_raises_on_invalid_group_kc_matrix_shape() -> None:
    """Test that invalid 2D parameter shapes raise an informative error."""
    with pytest.raises(ValueError, match="got shape"):
        sim_grouped_BKT(
            n_groups=2,
            n_kcs=3,
            prior=[[0.1, 0.2], [0.3, 0.4]],
            rng_seed=1,
        )


def test_sim_grouped_bkt_requires_group_id_column() -> None:
    """Test that the output always includes a group_id column."""
    data_df = sim_grouped_BKT(
        n_students=10,
        n_problems=5,
        n_groups=3,
        rng_seed=42,
    )

    assert "group_id" in data_df.columns
    assert data_df["group_id"].notna().all()


def test_sim_grouped_bkt_distributes_students_evenly() -> None:
    """Test that students are evenly distributed across groups by default."""
    n_students = 6
    n_groups = 3
    data_df = sim_grouped_BKT(
        n_students=n_students,
        n_problems=4,
        n_groups=n_groups,
        rng_seed=42,
    )

    # Count students per group
    students_per_group = data_df.groupby("group_id")["student_id"].nunique()
    expected_per_group = n_students // n_groups
    assert (students_per_group == expected_per_group).all()


def test_sim_grouped_bkt_custom_group_sequence() -> None:
    """Test that custom group assignments are respected."""
    n_students = 4
    n_problems = 3
    group_sequence = [0, 0, 1, 1]  # First 2 students in group 0, last 2 in group 1

    data_df = sim_grouped_BKT(
        n_students=n_students,
        n_problems=n_problems,
        n_groups=2,
        group_sequence=group_sequence,
        rng_seed=42,
    )

    # Check that students are assigned to correct groups
    for i, expected_group in enumerate(group_sequence):
        student_id = f"stu_{i}"
        student_data = data_df[data_df["student_id"] == student_id]
        actual_group = student_data["group_id"].iloc[0]
        assert actual_group == f"group_{expected_group}"


def test_sim_grouped_bkt_raises_on_invalid_group_sequence() -> None:
    """Test that invalid group sequences raise appropriate errors."""
    with pytest.raises(ValueError, match="group_sequence must have length n_students"):
        sim_grouped_BKT(
            n_students=5,
            n_groups=2,
            group_sequence=[0, 1, 0, 1],  # Wrong length
            rng_seed=1,
        )

    with pytest.raises(ValueError, match="group_sequence entries must be in"):
        sim_grouped_BKT(
            n_students=3,
            n_groups=2,
            group_sequence=[0, 1, 2],  # Index out of range
            rng_seed=1,
        )


def test_sim_grouped_bkt_has_expected_columns() -> None:
    """Test that output has all expected columns."""
    data_df = sim_grouped_BKT(
        n_students=5,
        n_problems=3,
        n_groups=2,
        rng_seed=42,
    )

    expected_columns = {
        "student_id",
        "problem_id",
        "correct",
        "timestamp",
        "kc_id",
        "group_id",
    }
    assert expected_columns.issubset(set(data_df.columns))
