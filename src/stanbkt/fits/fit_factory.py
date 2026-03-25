""" "" Factory for creating fit classes and options based on fit method."""

from stanbkt.fits.fit_options import (
    StanFitOptions,
    MCMCFitOptions,
    VBFitOptions,
    MLEFitOptions,
    PFFitOptions,
)

import stat
from typing import TypeAlias
from stanbkt.fits.core.mcmc import MCMCFit
from stanbkt.fits.core.mle import MLEFit
from stanbkt.fits.core.vb import VBFit
from stanbkt.fits.core.pf import PathfinderFit
from stanbkt.fits.fit_types import FitMethod, CmdStanFit
from stanbkt.fits.core.base import BaseFit
from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanPathfinder, CmdStanVB

FitClassType: TypeAlias = type[BaseFit]
""" Alias for fit class types, i.e. subclasses of BaseFit."""


class FitFactory:
    """Factory for creating fit classes and options based on fit method."""

    FIT_METHOD_TO_OPTION_MAPPING: dict[FitMethod, type[StanFitOptions]] = {
        FitMethod.MCMC: MCMCFitOptions,
        FitMethod.VB: VBFitOptions,
        FitMethod.MLE: MLEFitOptions,
        FitMethod.PATHFINDER: PFFitOptions,
    }

    @staticmethod
    def get_fit_class_from_method(fit_method: FitMethod) -> FitClassType:
        """Get the expected CmdStan fit class for this fit method.

        Returns
        -------
        type[CmdStanFit]
            Expected CmdStan fit class corresponding to this fit method.

        Raises
        ------
        ValueError
            If fit method is unsupported.
        """

        if fit_method == FitMethod.MCMC:
            return MCMCFit
        elif fit_method == FitMethod.MLE:
            return MLEFit
        elif fit_method == FitMethod.VB:
            return VBFit
        elif fit_method == FitMethod.PATHFINDER:
            return PathfinderFit
        else:
            raise ValueError(
                f"Unsupported fit method '{fit_method}'. Cannot determine fit class."
            )

    @staticmethod
    def create_default_fit_options(fit_method: FitMethod) -> StanFitOptions:
        """Get default fit options for a given fit method.

        Parameters
        ----------
        fit_method : FitMethod
            Fit method for which to get default options.

        Returns
        -------
        StanFitOptions
            Default fit options for the specified method.

        Raises
        ------
        ValueError
            If fit method is unsupported.
        """
        try:
            fit_option_class = FitFactory.FIT_METHOD_TO_OPTION_MAPPING[fit_method]
            return fit_option_class()
        except KeyError:
            raise ValueError(
                f"Unsupported fit method '{fit_method}'. Cannot determine default fit options."
            ) from None

    @staticmethod
    def create_fit_options_from_dict(
        fit_option_dict: dict, fit_method: FitMethod
    ) -> StanFitOptions:
        """Create fit options from a dictionary for a given fit method.

        Parameters
        ----------
        fit_option_dict : dict
            Dictionary containing fit option parameters.
        fit_method : FitMethod
            Fit method for which to create options.

        Returns
        -------
        StanFitOptions
            Fit options instance created from the dictionary.

        Raises
        ------
        ValueError
            If fit method is unsupported.
        """
        try:
            fit_option_class = FitFactory.FIT_METHOD_TO_OPTION_MAPPING[fit_method]
            return fit_option_class.from_dict(fit_option_dict)
        except KeyError:
            raise ValueError(
                f"Unsupported fit method '{fit_method}'. Cannot determine fit options from dictionary."
            ) from None

    @staticmethod
    def verify_fit_options_compatibility(
        fit_options: StanFitOptions, fit_method: FitMethod
    ) -> None:
        """Verify that provided fit options are compatible with the specified fit method.

        Parameters
        ----------
        fit_options : StanFitOptions
            Fit options to verify.
        fit_method : FitMethod
            Fit method for which to verify compatibility.

        Raises
        ------
        TypeError
            If fit options are not compatible with the specified fit method.
        ValueError
            If fit method is unsupported.
        """
        try:
            fit_option_class = FitFactory.FIT_METHOD_TO_OPTION_MAPPING[fit_method]
            if not isinstance(fit_options, fit_option_class):
                raise TypeError(
                    f"Incompatible fit options type {type(fit_options).__name__} for fit method '{fit_method}'. Expected {fit_option_class.__name__}."
                )
        except KeyError:
            raise ValueError(
                f"Unsupported fit method '{fit_method}'. Cannot verify fit options compatibility."
            ) from None
