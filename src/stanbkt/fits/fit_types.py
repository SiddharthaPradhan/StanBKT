"""Type definitions for fit metadata and fit method resolution.

This module contains non-fit classes and aliases used by fit implementations,
including metadata containers and fit method enumeration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Union, TypeAlias

from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanPathfinder, CmdStanVB

CmdStanFit: TypeAlias = Union[CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanPathfinder]
"""Alias for supported CmdStan fit result objects."""


class FitMethod(StrEnum):
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
    def infer_fit_method_from_stan_fit(fit: CmdStanFit) -> FitMethod:
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
class FitSaveEntry:
    """Mapping from a knowledge component to a fit save folder.

    Attributes
    ----------
    kc : str
        Knowledge component identifier.
    save_folder : os.PathLike | str
        Folder name under fit root for this KC's fit files.
    summary_cache_available : bool, default=False
        Whether summary cache CSV exists for this KC.
    group2index : dict[str, int] | None, default=None
        Optional mapping from group identifiers to integer indices used in the fit.
    groups : set[str] | None, default=None
        Optional set of group identifiers included in the fit.
    """

    kc: str
    save_folder: Union[os.PathLike, str]
    summary_cache_available: bool = False
    group2index: Union[dict[str, int], None] = None
    groups: Union[set[str], None] = None

    def __hash__(self) -> int:
        """Compute a stable hash even when optional mapping/set fields are present."""
        group2index_hashable = (
            None
            if self.group2index is None
            else frozenset((str(k), int(v)) for k, v in self.group2index.items())
        )
        groups_hashable = (
            None
            if self.groups is None
            else frozenset(str(group_name) for group_name in self.groups)
        )
        return hash(
            (
                self.kc,
                str(self.save_folder),
                self.summary_cache_available,
                group2index_hashable,
                groups_hashable,
            )
        )


FitSaves: TypeAlias = dict[str, FitSaveEntry]
"""Dict mapping knowledge component identifiers to fit save entries."""


@dataclass
class FitMetadata:
    """Root metadata for persisted fits.

    Attributes
    ----------
    fit_method : FitMethod
        Method used to fit all attached KCs.
    fit_saves : FitSaves
        Saved fit folder entries, keyed by knowledge component identifier.
    summary_percentiles : tuple[float, float], default (2.5, 97.5)
        Lower and upper percentiles used when computing summary statistics. Values should be in range [1, 99].
        Persisted so that cached summaries remain valid after a save/load round-trip.
    """

    fit_method: FitMethod
    fit_saves: FitSaves = field(default_factory=dict)
    summary_percentiles: tuple[float, float] = (2.5, 97.5)
