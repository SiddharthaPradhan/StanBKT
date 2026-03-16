"""Tests for fit core classes (mcmc, mle, vb, pf)."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from stanbkt.fits.core.mcmc import MCMCFit
from stanbkt.fits.core.mle import MLEFit
from stanbkt.fits.core.vb import VBFit
from stanbkt.fits.core.pf import PathfinderFit
from stanbkt.fits.fit_types import FitMethod
from stanbkt.utils.verbose import VerbosityLevel


class TestMCMCFit:
    def test_fit_method_property(self):
        fit = MCMCFit()
        assert fit._fit_method == FitMethod.MCMC

    def test_initialization_with_defaults(self):
        fit = MCMCFit()
        assert fit.kc_fits == {}
        assert fit.verbose == VerbosityLevel.INFO

    def test_initialization_with_verbose(self):
        fit = MCMCFit(verbose=VerbosityLevel.DEBUG)
        assert fit.verbose == VerbosityLevel.DEBUG

    def test_create_inits_with_none(self):
        fit = MCMCFit()
        result = fit._create_inits(None)
        assert result == {}

    def test_create_inits_with_single_kc(self):
        fit = MCMCFit()
        result = fit._create_inits("kc_a")
        assert result == {}

    def test_create_inits_with_list_of_kcs(self):
        fit = MCMCFit()
        result = fit._create_inits(["kc_a", "kc_b", "kc_c"])
        assert result == {"kc_a": {}, "kc_b": {}, "kc_c": {}}

    def test_summary_method_exists(self):
        fit = MCMCFit()
        # Method should exist even if not implemented
        assert hasattr(fit, "summary")


class TestMLEFit:
    def test_fit_method_property(self):
        fit = MLEFit()
        assert fit._fit_method == FitMethod.MLE

    def test_initialization_with_defaults(self):
        fit = MLEFit()
        assert fit.kc_fits == {}
        assert fit.verbose == VerbosityLevel.INFO

    def test_initialization_with_verbose(self):
        fit = MLEFit(verbose=VerbosityLevel.WARN)
        assert fit.verbose == VerbosityLevel.WARN

    def test_create_inits_with_none(self):
        fit = MLEFit()
        result = fit._create_inits(None)
        assert result == {}

    def test_create_inits_with_single_kc(self):
        fit = MLEFit()
        result = fit._create_inits("kc_a")
        assert result == {}

    def test_create_inits_with_list_of_kcs(self):
        fit = MLEFit()
        result = fit._create_inits(["kc_a", "kc_b"])
        assert result == {"kc_a": {}, "kc_b": {}}

    def test_summary_method_exists(self):
        fit = MLEFit()
        assert hasattr(fit, "summary")


class TestVBFit:
    def test_fit_method_property(self):
        fit = VBFit()
        assert fit._fit_method == FitMethod.VB

    def test_initialization_with_defaults(self):
        fit = VBFit()
        assert fit.kc_fits == {}
        assert fit.verbose == VerbosityLevel.INFO

    def test_initialization_with_verbose(self):
        fit = VBFit(verbose=VerbosityLevel.DEBUG)
        assert fit.verbose == VerbosityLevel.DEBUG

    def test_create_inits_with_none(self):
        fit = VBFit()
        result = fit._create_inits(None)
        assert result == {}

    def test_create_inits_with_single_kc(self):
        fit = VBFit()
        result = fit._create_inits("kc_a")
        assert result == {}

    def test_create_inits_with_list_of_kcs(self):
        fit = VBFit()
        result = fit._create_inits(["kc_a", "kc_b"])
        assert result == {"kc_a": {}, "kc_b": {}}

    def test_summary_method_exists(self):
        fit = VBFit()
        assert hasattr(fit, "summary")


class TestPathfinderFit:
    def test_fit_method_property(self):
        fit = PathfinderFit()
        assert fit._fit_method == FitMethod.PATHFINDER

    def test_initialization_with_defaults(self):
        fit = PathfinderFit()
        assert fit.kc_fits == {}
        assert fit.verbose == VerbosityLevel.INFO

    def test_initialization_with_verbose(self):
        fit = PathfinderFit(verbose=VerbosityLevel.INFO)
        assert fit.verbose == VerbosityLevel.INFO

    def test_create_inits_with_none(self):
        fit = PathfinderFit()
        result = fit._create_inits(None)
        assert result == {}

    def test_create_inits_with_single_kc(self):
        fit = PathfinderFit()
        result = fit._create_inits("kc_a")
        assert result == {}

    def test_create_inits_with_list_of_kcs(self):
        fit = PathfinderFit()
        result = fit._create_inits(["kc_a", "kc_b", "kc_c"])
        assert result == {"kc_a": {}, "kc_b": {}, "kc_c": {}}

    def test_summary_method_exists(self):
        fit = PathfinderFit()
        assert hasattr(fit, "summary")


class TestFitCoreInteraction:
    """Test that fit core classes can be used with BaseFit functionality."""

    def test_mcmc_fit_can_add_fits(self):
        fit = MCMCFit()
        mock_stan_fit = MagicMock()

        # Mock the fit method detection
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                FitMethod,
                "infer_fit_method_from_stan_fit",
                staticmethod(lambda _: FitMethod.MCMC),
            )
            fit.add_fit("kc_a", mock_stan_fit)

        assert "kc_a" in fit.kc_fits

    def test_vb_fit_can_add_fits(self):
        fit = VBFit()
        mock_stan_fit = MagicMock()

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                FitMethod,
                "infer_fit_method_from_stan_fit",
                staticmethod(lambda _: FitMethod.VB),
            )
            fit.add_fit("kc_a", mock_stan_fit)

        assert "kc_a" in fit.kc_fits

    def test_mle_fit_can_add_fits(self):
        fit = MLEFit()
        mock_stan_fit = MagicMock()

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                FitMethod,
                "infer_fit_method_from_stan_fit",
                staticmethod(lambda _: FitMethod.MLE),
            )
            fit.add_fit("kc_a", mock_stan_fit)

        assert "kc_a" in fit.kc_fits

    def test_pf_fit_can_add_fits(self):
        fit = PathfinderFit()
        mock_stan_fit = MagicMock()

        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                FitMethod,
                "infer_fit_method_from_stan_fit",
                staticmethod(lambda _: FitMethod.PATHFINDER),
            )
            fit.add_fit("kc_a", mock_stan_fit)

        assert "kc_a" in fit.kc_fits
