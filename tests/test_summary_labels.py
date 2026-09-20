import pandas as pd

from stanbkt.utils.summary_utils import label_summary_index


def _summary(names):
    df = pd.DataFrame({"mean": range(len(names))})
    df.insert(0, "kc_id", "kc_a")
    df.insert(1, "parameter", names)
    return df.set_index(["kc_id", "parameter"])


def _label(df, **overrides):
    kwargs = dict(
        group2index={"kc_a": {"A": 1, "B": 2}},
        student2index={"kc_a": None},
        covariate_columns={"kc_a": None},
        group_col_name="group_id",
        student_col_name="student_id",
    )
    kwargs.update(overrides)
    return label_summary_index(df, **kwargs)


def test_group_params_get_group_labels():
    out = _label(_summary(["lp__", "learn[1]", "learn[2]", "pi_know[2]"]))
    assert out.index.names == ["kc_id", "parameter", "axis", "label"]
    assert ("kc_a", "learn", "group_id", "B") in out.index
    assert ("kc_a", "pi_know", "group_id", "B") in out.index
    assert ("kc_a", "lp__", "", "") in out.index


def test_individual_pi_know_uses_student_labels():
    out = _label(
        _summary(["pi_know[1]", "pi_know[2]"]),
        student2index={"kc_a": {"s1": 1, "s2": 2}},
    )
    assert ("kc_a", "pi_know", "student_id", "s2") in out.index


def test_covariate_and_missing_map():
    out = _label(
        _summary(["pi_b1_know_param[2]", "learn[3]"]),
        covariate_columns={"kc_a": ("age", "pretest")},
    )
    assert ("kc_a", "pi_b1_know_param", "covariate", "pretest") in out.index
    # out of range index keeps the raw integer
    assert ("kc_a", "learn", "group_id", "3") in out.index


def test_joint_scalar_params_have_no_axis_or_label():
    out = _label(_summary(["pi_b0_know_param[1]", "pi_sigma_param[1]", "learn[1]"]))
    assert ("kc_a", "pi_b0_know_param", "", "") in out.index
    assert ("kc_a", "pi_sigma_param", "", "") in out.index
    assert ("kc_a", "learn", "group_id", "A") in out.index
