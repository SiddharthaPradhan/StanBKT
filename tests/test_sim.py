import pytest

from stanbkt.utils import sim_simple_BKT


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
