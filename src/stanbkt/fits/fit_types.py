"""Type definitions for fit metadata and fit method resolution.

This module contains non-fit classes and aliases used by fit implementations,
including metadata containers and fit method enumeration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias, Union

from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanPathfinder, CmdStanVB

BaseCmdStanFit = Union[CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanPathfinder]
"""Alias for supported CmdStan fit result objects."""


class FitMethod(str, Enum):
    """Enumeration of supported fitting methods.

    Attributes
    ----------
    MCMC : str
        Markov chain Monte Carlo sampling.
    MLE : str
        Maximum likelihood / optimization.
    VB : str
        Variational Bayes.
    PATHFINDER : str
        Pathfinder variational approximation.
    """

    MCMC = "mcmc"
    MLE = "mle"
    VB = "vb"
    PATHFINDER = "pathfinder"

    @staticmethod
    def get_method_from_fit(fit: BaseCmdStanFit) -> "FitMethod":
        """Infer the fit method from a CmdStan fit object.

        Parameters
        ----------
        fit : BaseCmdStanFit
            Fit object created by CmdStanPy.

        Returns
        -------
        FitMethod
            Inferred fit method enum value.

        Raises
        ------
        ValueError
            If ``fit`` type is unsupported.
        """
        if isinstance(fit, CmdStanMCMC):
            return FitMethod.MCMC
        if isinstance(fit, CmdStanMLE):
            return FitMethod.MLE
        if isinstance(fit, CmdStanVB):
            return FitMethod.VB
        if isinstance(fit, CmdStanPathfinder):
            return FitMethod.PATHFINDER
        raise ValueError(
            f"Unsupported fit type '{type(fit).__name__}'. Cannot determine fit method."
        )


@dataclass(frozen=True, slots=True)
class FitSaveFolder:
    """Mapping from a knowledge component to a fit save folder.

    Attributes
    ----------
    kc : str
        Knowledge component identifier.
    save_folder : os.PathLike | str
        Folder name under fit root for this KC's fit files.
    summary_cache_available : bool, default=False
        Whether summary cache CSV exists for this KC.
    """

    kc: str
    save_folder: Union[os.PathLike, str]
    summary_cache_available: bool = False


FitSaves: TypeAlias = set[FitSaveFolder]
"""Set of fit save folder metadata entries."""


@dataclass
class FitMetadata:
    """Root metadata for persisted fits.

    Attributes
    ----------
    fit_method : FitMethod
        Method used to fit all attached KCs.
    fit_saves : FitSaves
        Saved fit folder entries.
    """

    fit_method: FitMethod
    fit_saves: FitSaves = field(default_factory=set)
