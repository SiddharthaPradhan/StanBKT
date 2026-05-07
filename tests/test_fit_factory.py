"""Tests for FitFactory class and fit options creation."""

import pytest

from stanbkt.fits.fit_factory import FitFactory
from stanbkt.fits.fit_options import (
    MCMCFitOptions,
    VBFitOptions,
    MLEFitOptions,
    PFFitOptions,
    StanFitOptions,
)
from stanbkt.fits.fit_types import FitMethod
from stanbkt.fits.core.mcmc import MCMCFit
from stanbkt.fits.core.mle import MLEFit
from stanbkt.fits.core.vb import VBFit
from stanbkt.fits.core.pf import PathfinderFit

# ---------------------------------------------------------------------------
# get_fit_class_from_method
# ---------------------------------------------------------------------------


class TestGetFitClassFromMethod:
    def test_returns_mcmc_fit_for_mcmc_method(self):
        result = FitFactory.get_fit_class_from_method(FitMethod.MCMC)
        assert result is MCMCFit

    def test_returns_mle_fit_for_mle_method(self):
        result = FitFactory.get_fit_class_from_method(FitMethod.MLE)
        assert result is MLEFit

    def test_returns_vb_fit_for_vb_method(self):
        result = FitFactory.get_fit_class_from_method(FitMethod.VB)
        assert result is VBFit

    def test_returns_pathfinder_fit_for_pathfinder_method(self):
        result = FitFactory.get_fit_class_from_method(FitMethod.PATHFINDER)
        assert result is PathfinderFit

    def test_raises_for_invalid_fit_method(self):
        # Create a mock invalid fit method by using a string that's not a valid FitMethod
        with pytest.raises(ValueError, match="Unsupported fit method"):
            FitFactory.get_fit_class_from_method("invalid_method")  # type: ignore


# ---------------------------------------------------------------------------
# create_default_fit_options
# ---------------------------------------------------------------------------


class TestCreateDefaultFitOptions:
    def test_returns_mcmc_options_for_mcmc_method(self):
        result = FitFactory.create_default_fit_options(FitMethod.MCMC)
        assert isinstance(result, MCMCFitOptions)
        assert result.chains == 4  # default value
        assert result.iter_sampling == 1000  # default value

    def test_returns_vb_options_for_vb_method(self):
        result = FitFactory.create_default_fit_options(FitMethod.VB)
        assert isinstance(result, VBFitOptions)
        assert result.algorithm == "meanfield"  # default value
        assert result.iter is None  # None lets CmdStanPy choose

    def test_returns_mle_options_for_mle_method(self):
        result = FitFactory.create_default_fit_options(FitMethod.MLE)
        assert isinstance(result, MLEFitOptions)
        assert (
            result.algorithm is None
        )  # None lets CmdStanPy choose (defaults to lbfgs)
        assert result.iter == 2000  # default value

    def test_returns_pf_options_for_pathfinder_method(self):
        result = FitFactory.create_default_fit_options(FitMethod.PATHFINDER)
        assert isinstance(result, PFFitOptions)
        assert result.psis_resample is True  # default value
        assert result.calculate_lp is True  # default value

    def test_raises_for_invalid_fit_method(self):
        with pytest.raises(ValueError, match="Unsupported fit method"):
            # Use a string that would cause KeyError
            FitFactory.create_default_fit_options("invalid")  # type: ignore


# ---------------------------------------------------------------------------
# create_fit_options_from_dict
# ---------------------------------------------------------------------------


class TestCreateFitOptionsFromDict:
    def test_creates_mcmc_options_with_custom_values(self):
        options_dict = {"chains": 8, "iter_sampling": 2000, "seed": 42}
        result = FitFactory.create_fit_options_from_dict(options_dict, FitMethod.MCMC)

        assert isinstance(result, MCMCFitOptions)
        assert result.chains == 8
        assert result.iter_sampling == 2000
        assert result.seed == 42

    def test_creates_vb_options_with_custom_values(self):
        options_dict = {"algorithm": "fullrank", "iter": 5000, "seed": 123}
        result = FitFactory.create_fit_options_from_dict(options_dict, FitMethod.VB)

        assert isinstance(result, VBFitOptions)
        assert result.algorithm == "fullrank"
        assert result.iter == 5000
        assert result.seed == 123

    def test_creates_mle_options_with_custom_values(self):
        options_dict = {"algorithm": "newton", "iter": 1000}
        result = FitFactory.create_fit_options_from_dict(options_dict, FitMethod.MLE)

        assert isinstance(result, MLEFitOptions)
        assert result.algorithm == "newton"
        assert result.iter == 1000

    def test_creates_pf_options_with_custom_values(self):
        options_dict = {"num_paths": 8, "draws": 2000, "seed": 999}
        result = FitFactory.create_fit_options_from_dict(
            options_dict, FitMethod.PATHFINDER
        )

        assert isinstance(result, PFFitOptions)
        assert result.num_paths == 8
        assert result.draws == 2000
        assert result.seed == 999

    def test_handles_extra_kwargs(self):
        options_dict = {
            "chains": 4,
            "custom_param": "value",
            "another_param": 123,
        }
        result = FitFactory.create_fit_options_from_dict(options_dict, FitMethod.MCMC)

        assert isinstance(result, MCMCFitOptions)
        assert result.chains == 4
        assert "custom_param" in result.extra_kwargs
        assert result.extra_kwargs["custom_param"] == "value"
        assert result.extra_kwargs["another_param"] == 123

    def test_creates_empty_options_from_empty_dict(self):
        result = FitFactory.create_fit_options_from_dict({}, FitMethod.MCMC)
        assert isinstance(result, MCMCFitOptions)
        # Should have default values
        assert result.chains == 4

    def test_return_type_is_correct_subclass(self):
        """Test that the return type is the specific subclass, not base class."""
        mcmc_result = FitFactory.create_fit_options_from_dict({}, FitMethod.MCMC)
        vb_result = FitFactory.create_fit_options_from_dict({}, FitMethod.VB)
        mle_result = FitFactory.create_fit_options_from_dict({}, FitMethod.MLE)
        pf_result = FitFactory.create_fit_options_from_dict({}, FitMethod.PATHFINDER)

        # These assertions verify the fix we made with Self type
        assert type(mcmc_result) is MCMCFitOptions
        assert type(vb_result) is VBFitOptions
        assert type(mle_result) is MLEFitOptions
        assert type(pf_result) is PFFitOptions

    def test_raises_for_invalid_fit_method(self):
        with pytest.raises(ValueError, match="Unsupported fit method"):
            FitFactory.create_fit_options_from_dict({}, "invalid")  # type: ignore


# ---------------------------------------------------------------------------
# verify_fit_options_compatibility
# ---------------------------------------------------------------------------


class TestVerifyFitOptionsCompatibility:
    def test_accepts_compatible_mcmc_options(self):
        options = MCMCFitOptions()
        # Should not raise
        FitFactory.verify_fit_options_compatibility(options, FitMethod.MCMC)

    def test_accepts_compatible_vb_options(self):
        options = VBFitOptions()
        # Should not raise
        FitFactory.verify_fit_options_compatibility(options, FitMethod.VB)

    def test_accepts_compatible_mle_options(self):
        options = MLEFitOptions()
        # Should not raise
        FitFactory.verify_fit_options_compatibility(options, FitMethod.MLE)

    def test_accepts_compatible_pf_options(self):
        options = PFFitOptions()
        # Should not raise
        FitFactory.verify_fit_options_compatibility(options, FitMethod.PATHFINDER)

    def test_raises_for_incompatible_options(self):
        mcmc_options = MCMCFitOptions()
        with pytest.raises(TypeError, match="Incompatible fit options type"):
            FitFactory.verify_fit_options_compatibility(mcmc_options, FitMethod.VB)

    def test_raises_with_correct_expected_type_in_message(self):
        vb_options = VBFitOptions()
        with pytest.raises(TypeError, match="Expected MCMCFitOptions"):
            FitFactory.verify_fit_options_compatibility(vb_options, FitMethod.MCMC)

    def test_raises_for_invalid_fit_method(self):
        options = MCMCFitOptions()
        with pytest.raises(ValueError, match="Unsupported fit method"):
            FitFactory.verify_fit_options_compatibility(options, "invalid")  # type: ignore


# ---------------------------------------------------------------------------
# FIT_OPTION_CLASS_MAPPING
# ---------------------------------------------------------------------------


class TestFitOptionClassMapping:
    def test_mapping_contains_all_fit_methods(self):
        """Ensure all FitMethod enum values have a corresponding option class."""
        expected_methods = {
            FitMethod.MCMC,
            FitMethod.VB,
            FitMethod.MLE,
            FitMethod.PATHFINDER,
        }
        assert set(FitFactory.FIT_METHOD_TO_OPTION_MAPPING.keys()) == expected_methods

    def test_mapping_values_are_correct_types(self):
        assert FitFactory.FIT_METHOD_TO_OPTION_MAPPING[FitMethod.MCMC] is MCMCFitOptions
        assert FitFactory.FIT_METHOD_TO_OPTION_MAPPING[FitMethod.VB] is VBFitOptions
        assert FitFactory.FIT_METHOD_TO_OPTION_MAPPING[FitMethod.MLE] is MLEFitOptions
        assert (
            FitFactory.FIT_METHOD_TO_OPTION_MAPPING[FitMethod.PATHFINDER]
            is PFFitOptions
        )


# ---------------------------------------------------------------------------
# to_dict method tests
# ---------------------------------------------------------------------------


class TestFitOptionsToDict:
    def test_mcmc_to_dict_removes_none_values(self):
        options = MCMCFitOptions(chains=4, seed=None, adapt_delta=None)
        result = options.to_dict()

        assert "chains" in result
        assert result["chains"] == 4
        assert "seed" not in result
        assert "adapt_delta" not in result

    def test_mcmc_to_dict_with_all_values(self):
        options = MCMCFitOptions(
            chains=8,
            iter_sampling=2000,
            seed=42,
            adapt_delta=0.95,
        )
        result = options.to_dict()

        assert result["chains"] == 8
        assert result["iter_sampling"] == 2000
        assert result["seed"] == 42
        assert result["adapt_delta"] == 0.95

    def test_to_dict_includes_extra_kwargs(self):
        options = MCMCFitOptions(chains=4)
        options.extra_kwargs = {"custom_param": "value", "another": 123}
        result = options.to_dict()

        assert result["chains"] == 4
        assert result["custom_param"] == "value"
        assert result["another"] == 123

    def test_to_dict_extra_kwargs_override_named_params(self):
        options = MCMCFitOptions(chains=4)
        options.extra_kwargs = {"chains": 999}
        result = options.to_dict()

        # extra_kwargs should override
        assert result["chains"] == 999

    def test_vb_to_dict_removes_none_values(self):
        options = VBFitOptions(algorithm="meanfield", seed=None)
        result = options.to_dict()

        assert "algorithm" in result
        assert "seed" not in result

    def test_mle_to_dict_removes_none_values(self):
        options = MLEFitOptions(algorithm="lbfgs", seed=None)
        result = options.to_dict()

        assert "algorithm" in result
        assert "seed" not in result

    def test_pf_to_dict_with_complex_types(self):
        options = PFFitOptions(
            num_paths=8,
            draws=1000,
            inits={"param": 0.5},
        )
        result = options.to_dict()

        assert result["num_paths"] == 8
        assert result["draws"] == 1000
        assert result["inits"] == {"param": 0.5}

    def test_to_dict_returns_dict_type(self):
        options = MCMCFitOptions()
        result = options.to_dict()
        assert isinstance(result, dict)
