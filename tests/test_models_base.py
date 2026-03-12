import pytest

from stanbkt.models.base import (
    BayesianPriors,
    BKTModelBase,
    ModelType,
    PriorEstimationType,
)
from stanbkt.fits.fit_types import FitMethod
from stanbkt.utils.verbose import VerbosityLevel


# ---------------------------------------------------------------------------
# Minimal concrete subclass — only satisfies the abstract interface
# ---------------------------------------------------------------------------


class _ConcreteModel(BKTModelBase):
    def fit(
        self, data, column_mapping=None, method=FitMethod.MCMC, stan_fit_kwargs=None
    ):
        return self

    def evaluate(self, **kwargs):
        return {}

    @property
    def _stan_model_filename(self):
        return "/nonexistent.stan"

    @property
    def _stan_hidden_filename(self):
        return "/nonexistent_hidden.stan"

    @property
    def _stan_smoothed_hidden_filename(self):
        return "/nonexistent_smoothed.stan"


def _make_fitted_model(kcs=("kc_a", "kc_b")):
    """Return a model whose _is_fitted and fits_ reflect a completed fit."""
    model = _ConcreteModel()
    model._is_fitted = True
    model.fits_ = {kc: object() for kc in kcs}
    return model


# ---------------------------------------------------------------------------
# BKTModelBase.__init__
# ---------------------------------------------------------------------------


class TestBKTModelBaseInit:
    def test_default_verbose(self):
        m = _ConcreteModel()
        assert m.verbose == VerbosityLevel.INFO

    def test_stan_compile_kwargs_default_empty(self):
        m = _ConcreteModel()
        assert m.stan_compile_kwargs == {}

    def test_cpp_compile_kwargs_default_empty(self):
        m = _ConcreteModel()
        assert m.cpp_compile_kwargs == {}

    def test_stan_model_is_none(self):
        m = _ConcreteModel()
        assert m._stan_model is None

    def test_fits_is_none(self):
        m = _ConcreteModel()
        assert m.fits_ is None

    def test_is_fitted_is_false(self):
        m = _ConcreteModel()
        assert m._is_fitted is False

    def test_previous_fit_method_is_none(self):
        m = _ConcreteModel()
        assert m._previous_fit_method is None

    def test_custom_compile_kwargs_stored(self):
        m = _ConcreteModel(
            stan_compile_kwargs={"foo": "bar"}, cpp_compile_kwargs={"baz": 1}
        )
        assert m.stan_compile_kwargs == {"foo": "bar"}
        assert m.cpp_compile_kwargs == {"baz": 1}


# ---------------------------------------------------------------------------
# fit_check
# ---------------------------------------------------------------------------


class TestFitCheck:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.fit_check()

    def test_raises_when_fits_is_none_even_if_is_fitted_true(self):
        m = _ConcreteModel()
        m._is_fitted = True
        m.fits_ = None
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.fit_check()

    def test_passes_when_fitted(self):
        m = _make_fitted_model()
        m.fit_check()  # should not raise


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.summary()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_raises_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            m.predict()


# ---------------------------------------------------------------------------
# check_data_contains_fitted_kcs
# ---------------------------------------------------------------------------


class TestCheckDataContainsFittedKCs:
    def test_passes_when_all_kcs_are_fitted(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        m.check_data_contains_fitted_kcs({"kc_a"})  # subset → no error

    def test_passes_for_exact_fitted_set(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        m.check_data_contains_fitted_kcs({"kc_a", "kc_b"})

    def test_raises_for_unknown_kc(self):
        m = _make_fitted_model(kcs=("kc_a",))
        with pytest.raises(ValueError, match="kc_z"):
            m.check_data_contains_fitted_kcs({"kc_z"})

    def test_raises_even_for_partially_overlapping_set(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        with pytest.raises(ValueError, match="kc_z"):
            m.check_data_contains_fitted_kcs({"kc_a", "kc_z"})

    def test_raises_fit_check_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError):
            m.check_data_contains_fitted_kcs({"kc_a"})


# ---------------------------------------------------------------------------
# get_kc_in_fitted_kcs
# ---------------------------------------------------------------------------


class TestGetKCInFittedKCs:
    def test_returns_exact_intersection(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        result = m.get_kc_in_fitted_kcs({"kc_a", "kc_c"})
        assert result == {"kc_a"}

    def test_returns_all_when_full_overlap(self):
        m = _make_fitted_model(kcs=("kc_a", "kc_b"))
        result = m.get_kc_in_fitted_kcs({"kc_a", "kc_b"})
        assert result == {"kc_a", "kc_b"}

    def test_returns_empty_when_no_overlap(self):
        m = _make_fitted_model(kcs=("kc_a",))
        result = m.get_kc_in_fitted_kcs({"kc_z"})
        assert result == set()

    def test_raises_fit_check_when_not_fitted(self):
        m = _ConcreteModel()
        with pytest.raises(RuntimeError):
            m.get_kc_in_fitted_kcs({"kc_a"})


# ---------------------------------------------------------------------------
# BayesianPriors.get_default_priors
# ---------------------------------------------------------------------------


class TestBayesianPriorsGetDefaultPriors:
    def test_standard_returns_scalar_priors(self):
        priors = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert isinstance(priors, dict)
        assert all(isinstance(v, float) for v in priors.values())

    def test_standard_returns_all_10_prior_keys(self):
        priors = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert set(priors.keys()) == set(BayesianPriors)

    def test_nested_returns_same_scalar_priors_as_standard(self):
        standard = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        nested = BayesianPriors.get_default_priors(
            ModelType.NESTED, PriorEstimationType.DEFAULT
        )
        assert standard == nested

    def test_grouped_returns_list_priors(self):
        priors = BayesianPriors.get_default_priors(
            ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=3
        )
        assert all(isinstance(v, list) for v in priors.values())

    def test_grouped_list_length_matches_n_groups(self):
        n = 4
        priors = BayesianPriors.get_default_priors(
            ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=n
        )
        assert all(len(v) == n for v in priors.values())

    def test_grouped_values_replicated_from_scalar(self):
        scalar = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        grouped = BayesianPriors.get_default_priors(
            ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=2
        )
        for key in scalar:
            assert grouped[key] == [scalar[key], scalar[key]]

    def test_grouped_raises_for_non_integer_n_groups(self):
        with pytest.raises(ValueError, match="n_groups must be an integer"):
            BayesianPriors.get_default_priors(
                ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=None
            )

    def test_grouped_raises_for_zero_n_groups(self):
        with pytest.raises(ValueError, match="n_groups must be > 0"):
            BayesianPriors.get_default_priors(
                ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=0
            )

    def test_grouped_raises_for_negative_n_groups(self):
        with pytest.raises(ValueError, match="n_groups must be > 0"):
            BayesianPriors.get_default_priors(
                ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=-1
            )

    def test_joint_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BayesianPriors.get_default_priors(
                ModelType.STANDARD, PriorEstimationType.JOINT
            )


# ---------------------------------------------------------------------------
# BayesianPriors.add_missing_priors
# ---------------------------------------------------------------------------


class TestBayesianPriorsAddMissingPriors:
    def test_empty_input_returns_all_defaults(self):
        result = BayesianPriors.add_missing_priors(
            {}, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        defaults = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert result == defaults

    def test_partial_override_merges_correctly(self):
        override = {BayesianPriors.LEARN_MU: 1.5}
        result = BayesianPriors.add_missing_priors(
            override, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert result[BayesianPriors.LEARN_MU] == 1.5
        # Other keys remain as defaults
        defaults = BayesianPriors.get_default_priors(
            ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        for key in defaults:
            if key != BayesianPriors.LEARN_MU:
                assert result[key] == defaults[key]

    def test_string_key_accepted(self):
        result = BayesianPriors.add_missing_priors(
            {"learn_mu": 2.0}, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert result[BayesianPriors.LEARN_MU] == 2.0

    def test_invalid_string_key_raises(self):
        with pytest.raises(ValueError, match="Unsupported prior key"):
            BayesianPriors.add_missing_priors(
                {"nonexistent_param": 1.0},
                ModelType.STANDARD,
                PriorEstimationType.DEFAULT,
            )

    def test_result_contains_all_prior_keys(self):
        result = BayesianPriors.add_missing_priors(
            {BayesianPriors.SLIP_MU: -0.5},
            ModelType.STANDARD,
            PriorEstimationType.DEFAULT,
        )
        assert set(result.keys()) == set(BayesianPriors)

    def test_grouped_partial_override(self):
        override = {BayesianPriors.LEARN_MU: [1.5, 1.5, 1.5]}
        result = BayesianPriors.add_missing_priors(
            override, ModelType.GROUPED, PriorEstimationType.DEFAULT, n_groups=3
        )
        assert result[BayesianPriors.LEARN_MU] == [1.5, 1.5, 1.5]

    def test_full_override_with_enum_keys(self):
        overrides = {prior: 0.0 for prior in BayesianPriors}
        result = BayesianPriors.add_missing_priors(
            overrides, ModelType.STANDARD, PriorEstimationType.DEFAULT
        )
        assert all(v == 0.0 for v in result.values())
