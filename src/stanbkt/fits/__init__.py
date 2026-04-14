"""BKT Fit results and options for different inference methods.

This module defines the core fit result classes for various inference methods
(MCMC, MLE, VB, Pathfinder) and their associated options. It also includes a
factory for creating fit instances based on user-specified options.
"""

from stanbkt.fits.core.base import FitBase
from stanbkt.fits.core.mcmc import MCMCFit
from stanbkt.fits.core.mle import MLEFit
from stanbkt.fits.core.vb import VBFit
from stanbkt.fits.core.pf import PathfinderFit
from stanbkt.fits.fit_types import FitMethod, FitMetadata
from stanbkt.fits.fit_options import (
    BaseFitOptions,
    MCMCFitOptions,
    VBFitOptions,
    MLEFitOptions,
    PFFitOptions,
    StanFitOptions,
)
from stanbkt.fits.fit_factory import FitFactory

__all__ = [
    # Fit classes
    "FitBase",
    "MCMCFit",
    "MLEFit",
    "VBFit",
    "PathfinderFit",
    # Fit method & metadata
    "FitMethod",
    "FitMetadata",
    # Fit options
    "BaseFitOptions",
    "MCMCFitOptions",
    "VBFitOptions",
    "MLEFitOptions",
    "PFFitOptions",
    "StanFitOptions",
    # Factory
    "FitFactory",
]
