"""Tests for stanbkt.models.priors."""

from __future__ import annotations

import pytest

from stanbkt.models.priors import (
    HierarchicalPriors,
    MultiPriors,
    PriorsBase,
    StandardPriors,
    CORRECTNESS_ONLY_STRATEGY_KEYS,
    JOINT_STRATEGY_KEYS,
    _UNSET,
    _UnsetType,
)
from stanbkt.models.model_types import InitKnowledgeStrategy
from stanbkt.models.core.standard import StandardBKT
from stanbkt.models.core.multi import MultiBKT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CO = InitKnowledgeStrategy.CORRECTNESS_ONLY
JOINT = InitKnowledgeStrategy.JOINT


# Helper: StandardPriors with all JOINT keys set
def _joint_standard_priors(**kwargs) -> StandardPriors:
    """Return a StandardPriors with JOINT strategy keys explicitly set."""
    joint_defaults = dict(
        pi_b0_know_mu=0.0,
        pi_b0_know_std=5.0,
        pi_b1_know_mu=0.0,
        pi_b1_know_std=5.0,
        pi_sigma_lambda=0.5,
    )
    joint_defaults.update(kwargs)
    return StandardPriors(**joint_defaults)


# ---------------------------------------------------------------------------
# _UnsetType / _UNSET sentinel
# ---------------------------------------------------------------------------


class TestUnsetSentinel:
    def test_unset_is_instance_of_unset_type(self):
        assert isinstance(_UNSET, _UnsetType)

    def test_unset_is_unique(self):
        assert _UNSET is _UNSET

    def test_none_is_not_unset(self):
        assert _UNSET is not None


# ---------------------------------------------------------------------------
# StandardPriors — construction and defaults
# ---------------------------------------------------------------------------


class TestStandardPriorsConstruction:
    def test_default_construction_fills_defaults(self):
        p = StandardPriors()
        assert p.learn_mu == 0.0
        assert p.learn_std == 5.0
        assert p.forget_mu == -2.0
        assert p.forget_std == 5.0
        assert p.guess_mu == -1.0
        assert p.guess_std == 5.0
        assert p.slip_mu == -1.0
        assert p.slip_std == 5.0
        assert p.pi_know_mu == -2.0
        assert p.pi_know_std == 5.0

    def test_use_defaults_false_sets_none(self):
        p = StandardPriors(use_defaults=False)
        for name in (
            "learn_mu",
            "learn_std",
            "forget_mu",
            "forget_std",
            "guess_mu",
            "guess_std",
            "slip_mu",
            "slip_std",
            "pi_know_mu",
            "pi_know_std",
        ):
            assert getattr(p, name) is None, f"{name} should be None"

    def test_explicit_value_is_preserved_with_defaults(self):
        p = StandardPriors(learn_mu=1.5)
        assert p.learn_mu == 1.5
        # other defaults still filled
        assert p.learn_std == 5.0

    def test_explicit_value_is_preserved_without_defaults(self):
        p = StandardPriors(learn_mu=1.5, use_defaults=False)
        assert p.learn_mu == 1.5
        assert p.learn_std is None

    def test_explicit_none_is_preserved_with_defaults(self):
        # Passing None explicitly should keep it as None even when use_defaults=True
        p = StandardPriors(learn_mu=None)
        assert p.learn_mu is None

    def test_joint_priors_default_constructed(self):
        p = StandardPriors()
        # JOINT strategy priors are NOT filled by __post_init__ (which uses CORRECTNESS_ONLY defaults)
        # they remain _UNSET until explicitly set or until use_defaults=False sets them to None
        assert isinstance(p.pi_b0_know_mu, _UnsetType)
        assert isinstance(p.pi_b0_know_std, _UnsetType)
        assert isinstance(p.pi_b1_know_mu, _UnsetType)
        assert isinstance(p.pi_b1_know_std, _UnsetType)
        assert isinstance(p.pi_sigma_lambda, _UnsetType)


# ---------------------------------------------------------------------------
# StandardPriors — to_dict
# ---------------------------------------------------------------------------


class TestStandardPriorsToDict:
    def test_to_dict_correctness_only_excludes_joint_keys(self):
        p = StandardPriors()
        d = p.to_dict(CO)
        for key in JOINT_STRATEGY_KEYS:
            assert (
                key not in d
            ), f"JOINT key '{key}' should be absent for CORRECTNESS_ONLY"

    def test_to_dict_joint_excludes_correctness_only_keys(self):
        p = StandardPriors()
        d = p.to_dict(JOINT)
        for key in CORRECTNESS_ONLY_STRATEGY_KEYS:
            assert (
                key not in d
            ), f"CORRECTNESS_ONLY key '{key}' should be absent for JOINT"

    def test_to_dict_correctness_only_contains_standard_keys(self):
        p = StandardPriors()
        d = p.to_dict(CO)
        for key in (
            "learn_mu",
            "learn_std",
            "forget_mu",
            "forget_std",
            "guess_mu",
            "guess_std",
            "slip_mu",
            "slip_std",
            "pi_know_mu",
            "pi_know_std",
        ):
            assert key in d

    def test_to_dict_joint_contains_joint_keys(self):
        p = StandardPriors()
        d = p.to_dict(JOINT)
        for key in JOINT_STRATEGY_KEYS:
            assert key in d

    def test_to_dict_does_not_contain_use_defaults(self):
        p = StandardPriors()
        assert "use_defaults" not in p.to_dict(CO)
        assert "use_defaults" not in p.to_dict(JOINT)

    def test_to_dict_values_match_fields(self):
        p = StandardPriors(learn_mu=1.0, learn_std=2.0)
        d = p.to_dict(CO)
        assert d["learn_mu"] == 1.0
        assert d["learn_std"] == 2.0

    def test_to_dict_none_values_included(self):
        p = StandardPriors(use_defaults=False)
        d = p.to_dict(CO)
        assert d["learn_mu"] is None
        assert d["learn_std"] is None


# ---------------------------------------------------------------------------
# StandardPriors — key_names
# ---------------------------------------------------------------------------


class TestStandardPriorsKeyNames:
    def test_key_names_returns_tuple(self):
        assert isinstance(StandardPriors.key_names(), tuple)

    def test_key_names_contains_learn_mu(self):
        assert "learn_mu" in StandardPriors.key_names()

    def test_key_names_contains_use_defaults(self):
        assert "use_defaults" in StandardPriors.key_names()

    def test_key_names_contains_all_base_joint_keys(self):
        names = StandardPriors.key_names()
        for key in JOINT_STRATEGY_KEYS:
            assert key in names


# ---------------------------------------------------------------------------
# StandardPriors — expected_class
# ---------------------------------------------------------------------------


class TestStandardPriorsExpectedClass:
    def test_expected_class_is_standard_bkt(self):
        assert StandardPriors.expected_class() is StandardBKT


# ---------------------------------------------------------------------------
# StandardPriors — get_default_priors
# ---------------------------------------------------------------------------


class TestStandardPriorsGetDefaultPriors:
    def test_correctness_only_returns_dict(self):
        d = StandardPriors.get_default_priors(CO)
        assert isinstance(d, dict)

    def test_correctness_only_has_expected_keys(self):
        d = StandardPriors.get_default_priors(CO)
        for key in ("learn_mu", "learn_std", "pi_know_mu", "pi_know_std"):
            assert key in d

    def test_correctness_only_excludes_joint_keys(self):
        d = StandardPriors.get_default_priors(CO)
        for key in JOINT_STRATEGY_KEYS:
            assert key not in d

    def test_joint_includes_joint_keys(self):
        d = StandardPriors.get_default_priors(JOINT)
        for key in JOINT_STRATEGY_KEYS:
            assert key in d

    def test_joint_excludes_correctness_only_keys(self):
        d = StandardPriors.get_default_priors(JOINT)
        for key in CORRECTNESS_ONLY_STRATEGY_KEYS:
            assert key not in d

    def test_return_none_true_gives_all_none(self):
        d = StandardPriors.get_default_priors(CO, return_none=True)
        assert all(v is None for v in d.values())

    def test_return_none_false_gives_floats(self):
        d = StandardPriors.get_default_priors(CO, return_none=False)
        assert all(isinstance(v, float) for v in d.values())

    def test_n_groups_ignored_for_standard(self):
        d1 = StandardPriors.get_default_priors(CO, n_groups=3)
        d2 = StandardPriors.get_default_priors(CO, n_groups=0)
        assert d1 == d2

    def test_invalid_estimation_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported prior estimation type"):
            StandardPriors.get_default_priors("invalid_type")  # type: ignore[arg-type]

    def test_default_learn_mu_value(self):
        d = StandardPriors.get_default_priors(CO)
        assert d["learn_mu"] == 0.0

    def test_default_forget_mu_value(self):
        d = StandardPriors.get_default_priors(CO)
        assert d["forget_mu"] == -2.0

    def test_default_pi_sigma_lambda_in_joint(self):
        d = StandardPriors.get_default_priors(JOINT)
        assert d["pi_sigma_lambda"] == 0.5


# ---------------------------------------------------------------------------
# StandardPriors — _validate_single
# ---------------------------------------------------------------------------


class TestStandardPriorsValidateSingle:
    def test_valid_priors_no_error(self):
        p = StandardPriors()
        StandardPriors._validate_single(p, StandardBKT, CO)

    def test_valid_priors_joint_no_error(self):
        p = _joint_standard_priors()
        StandardPriors._validate_single(p, StandardBKT, JOINT)

    def test_wrong_model_class_raises_type_error(self):
        p = StandardPriors()
        with pytest.raises(TypeError, match="Invalid model class"):
            StandardPriors._validate_single(p, MultiBKT, CO)

    def test_non_positive_std_raises_value_error(self):
        p = StandardPriors(learn_std=-1.0)
        with pytest.raises(ValueError, match="must be positive and non-zero"):
            StandardPriors._validate_single(p, StandardBKT, CO)

    def test_zero_std_raises_value_error(self):
        p = StandardPriors(learn_std=0.0)
        with pytest.raises(ValueError, match="must be positive and non-zero"):
            StandardPriors._validate_single(p, StandardBKT, CO)

    def test_non_positive_lambda_raises_value_error(self):
        p = _joint_standard_priors(pi_sigma_lambda=-0.5)
        with pytest.raises(ValueError, match="must be positive and non-zero"):
            StandardPriors._validate_single(p, StandardBKT, JOINT)

    def test_none_std_is_valid(self):
        p = StandardPriors(learn_std=None)
        StandardPriors._validate_single(p, StandardBKT, CO)  # should not raise

    def test_kc_id_in_error_message(self):
        p = StandardPriors(learn_std=-1.0)
        with pytest.raises(ValueError, match="kc_alpha"):
            StandardPriors._validate_single(p, StandardBKT, CO, kc_id="kc_alpha")

    def test_wrong_value_type_raises_value_error(self):
        p = StandardPriors()
        # Manually set an invalid type to bypass dataclass type hint
        object.__setattr__(p, "learn_mu", "not_a_float")
        with pytest.raises(ValueError, match="Invalid prior value type"):
            StandardPriors._validate_single(p, StandardBKT, CO)


# ---------------------------------------------------------------------------
# StandardPriors — _validate (dict and single dispatch)
# ---------------------------------------------------------------------------


class TestStandardPriorsValidate:
    def test_validate_single_instance(self):
        p = StandardPriors()
        StandardPriors._validate(p, StandardBKT, CO)  # should not raise

    def test_validate_dict_of_valid_priors(self):
        priors_dict = {
            "kc1": StandardPriors(),
            "kc2": StandardPriors(learn_mu=1.0),
        }
        StandardPriors._validate(priors_dict, StandardBKT, CO)  # should not raise

    def test_validate_dict_with_wrong_type_raises(self):
        priors_dict = {
            "kc1": StandardPriors(),
            "kc2": "not_a_priors_object",
        }
        with pytest.raises(ValueError, match="kc2"):
            StandardPriors._validate(priors_dict, StandardBKT, CO)  # type: ignore[arg-type]

    def test_validate_dict_with_invalid_value_raises(self):
        priors_dict = {
            "kc1": StandardPriors(learn_std=-1.0),
        }
        with pytest.raises(ValueError):
            StandardPriors._validate(priors_dict, StandardBKT, CO)

    def test_validate_dict_kc_id_in_error_message(self):
        priors_dict = {"my_kc": StandardPriors(slip_std=0.0)}
        with pytest.raises(ValueError, match="my_kc"):
            StandardPriors._validate(priors_dict, StandardBKT, CO)


# ---------------------------------------------------------------------------
# MultiPriors — construction
# ---------------------------------------------------------------------------


class TestMultiPriorsConstruction:
    def test_default_construction_fills_defaults(self):
        p = MultiPriors()
        # Without n_groups=0, the defaults should be lists of length 0
        assert isinstance(p, MultiPriors)

    def test_explicit_scalar_value_preserved(self):
        p = MultiPriors(learn_mu=1.5)
        assert p.learn_mu == 1.5

    def test_explicit_list_value_preserved(self):
        p = MultiPriors(learn_mu=[0.0, 1.0, 2.0])
        assert p.learn_mu == [0.0, 1.0, 2.0]

    def test_use_defaults_false_sets_none(self):
        p = MultiPriors(use_defaults=False)
        for name in ("learn_mu", "learn_std", "pi_know_mu", "pi_know_std"):
            assert getattr(p, name) is None


# ---------------------------------------------------------------------------
# MultiPriors — expected_class
# ---------------------------------------------------------------------------


class TestMultiPriorsExpectedClass:
    def test_expected_class_is_multi_bkt(self):
        assert MultiPriors.expected_class() is MultiBKT


# ---------------------------------------------------------------------------
# MultiPriors — get_default_priors
# ---------------------------------------------------------------------------


class TestMultiPriorsGetDefaultPriors:
    def test_correctness_only_n_groups_returns_lists(self):
        d = MultiPriors.get_default_priors(CO, n_groups=3)
        for key in ("learn_mu", "learn_std", "pi_know_mu", "pi_know_std"):
            assert isinstance(d[key], list), f"{key} should be a list"
            assert len(d[key]) == 3, f"{key} list should have length 3"

    def test_joint_strategy_keys_remain_scalar(self):
        d = MultiPriors.get_default_priors(JOINT, n_groups=3)
        for key in JOINT_STRATEGY_KEYS:
            assert not isinstance(
                d[key], list
            ), f"JOINT key '{key}' should not be a list"

    def test_return_none_true_gives_all_none(self):
        d = MultiPriors.get_default_priors(CO, return_none=True, n_groups=2)
        for key, val in d.items():
            if isinstance(val, list):
                assert all(v is None for v in val)
            else:
                assert val is None

    def test_n_groups_zero_returns_scalars(self):
        # n_groups=0 means no group info yet — priors pass through as scalars
        d = MultiPriors.get_default_priors(CO, n_groups=0)
        for key in ("learn_mu", "learn_std"):
            assert isinstance(d[key], float)

    def test_invalid_estimation_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported prior estimation type"):
            MultiPriors.get_default_priors("invalid_type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MultiPriors — _expand_grouped_priors
# ---------------------------------------------------------------------------


class TestMultiPriorsExpandGroupedPriors:
    def test_scalar_is_replicated(self):
        scalar = {"learn_mu": 0.0, "learn_std": 5.0}
        expanded = MultiPriors._expand_grouped_priors(scalar, n_groups=3)
        assert expanded["learn_mu"] == [0.0, 0.0, 0.0]
        assert expanded["learn_std"] == [5.0, 5.0, 5.0]

    def test_none_is_replicated(self):
        scalar = {"learn_mu": None}
        expanded = MultiPriors._expand_grouped_priors(scalar, n_groups=2)
        assert expanded["learn_mu"] == [None, None]

    def test_joint_keys_left_as_scalar(self):
        scalar = {key: 1.0 for key in JOINT_STRATEGY_KEYS}
        expanded = MultiPriors._expand_grouped_priors(scalar, n_groups=3)
        for key in JOINT_STRATEGY_KEYS:
            assert expanded[key] == 1.0, f"JOINT key '{key}' should remain scalar"

    def test_existing_list_unchanged(self):
        scalar = {"learn_mu": [1.0, 2.0]}
        expanded = MultiPriors._expand_grouped_priors(scalar, n_groups=5)
        assert expanded["learn_mu"] == [1.0, 2.0]  # unchanged, not re-expanded

    def test_n_groups_zero_leaves_scalars_unchanged(self):
        # n_groups < 1 means no expansion — scalars pass through as-is
        scalar = {"learn_mu": 0.0}
        expanded = MultiPriors._expand_grouped_priors(scalar, n_groups=0)
        assert expanded["learn_mu"] == 0.0


# ---------------------------------------------------------------------------
# MultiPriors — _validate_single
# ---------------------------------------------------------------------------


class TestMultiPriorsValidateSingle:
    def test_scalar_priors_valid(self):
        p = MultiPriors(learn_mu=0.0, learn_std=5.0)
        MultiPriors._validate_single(p, MultiBKT, CO)

    def test_list_priors_matching_n_groups_valid(self):
        p = MultiPriors.get_default_priors(CO, n_groups=2)
        priors = MultiPriors(**p, use_defaults=False)
        MultiPriors._validate_single(priors, MultiBKT, CO, n_groups=2)

    def test_wrong_model_class_raises_type_error(self):
        p = MultiPriors()
        with pytest.raises(TypeError, match="Invalid model class"):
            MultiPriors._validate_single(p, StandardBKT, CO)

    def test_list_wrong_length_raises_value_error(self):
        p = MultiPriors(learn_mu=[0.0, 1.0, 2.0])  # length 3
        with pytest.raises(ValueError, match="length 2"):
            MultiPriors._validate_single(p, MultiBKT, CO, n_groups=2)

    def test_non_positive_scalar_std_raises(self):
        p = MultiPriors(learn_std=-1.0)
        with pytest.raises(ValueError, match="must be positive"):
            MultiPriors._validate_single(p, MultiBKT, CO)

    def test_non_positive_list_std_raises(self):
        p = MultiPriors.get_default_priors(CO, n_groups=2)
        p["learn_std"] = [5.0, -1.0]
        priors = MultiPriors(**p, use_defaults=False)
        with pytest.raises(ValueError, match="must be positive"):
            MultiPriors._validate_single(priors, MultiBKT, CO, n_groups=2)

    def test_none_in_list_std_is_valid(self):
        p = MultiPriors.get_default_priors(CO, n_groups=2)
        p["learn_std"] = [5.0, None]
        priors = MultiPriors(**p, use_defaults=False)
        MultiPriors._validate_single(
            priors, MultiBKT, CO, n_groups=2
        )  # should not raise

    def test_kc_id_in_error_message_for_list(self):
        p = MultiPriors.get_default_priors(CO, n_groups=2)
        p["learn_std"] = [5.0, -1.0]
        priors = MultiPriors(**p, use_defaults=False)
        with pytest.raises(ValueError, match="kc_beta"):
            MultiPriors._validate_single(
                priors, MultiBKT, CO, kc_id="kc_beta", n_groups=2
            )

    def test_wrong_value_type_raises_value_error(self):
        p = MultiPriors()
        object.__setattr__(p, "learn_mu", 42)  # int, not float/list/None
        with pytest.raises(ValueError, match="Invalid prior value type"):
            MultiPriors._validate_single(p, MultiBKT, CO)


# ---------------------------------------------------------------------------
# PriorsBase — shared to_dict behaviour (via StandardPriors)
# ---------------------------------------------------------------------------


class TestPriorsBaseToDict:
    def test_unsupported_estimation_type_raises(self):
        p = StandardPriors()
        with pytest.raises(ValueError, match="Unsupported estimation type"):
            p.to_dict("unsupported")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HierarchicalPriors — abstract stub
# ---------------------------------------------------------------------------


class TestHierarchicalPriorsStub:
    def test_is_subclass_of_priors_base(self):
        assert issubclass(HierarchicalPriors, PriorsBase)

    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            HierarchicalPriors()  # type: ignore[abstract]
