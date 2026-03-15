"""Typed fit-option dataclasses for CmdStanPy fit methods.

This module defines a small set of strongly-typed, commonly used fitting options
for Stan workflows in this project. These options are intentionally conservative:
only frequently used arguments are exposed as explicit dataclass fields, while
less-common (or future) CmdStanPy keyword arguments can still be passed via
``extra_kwargs``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, TypeAlias, Union


@dataclass
class BaseFitOptions:
    """Base dataclass for typed Stan fit options.

    Parameters
    ----------
    extra_kwargs : dict[str, Any]
        Additional keyword arguments forwarded directly to CmdStanPy.
        These keys are merged last and therefore override generated defaults
        from dataclass fields.

    Notes
    -----
    This class is intended to be extended.
    Subclasses can add strongly typed fields and use ``extra_kwargs`` for less common or future CmdStanPy options.
    """

    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert options to a CmdStanPy kwargs dictionary.

        ``None`` values are removed so CmdStanPy can apply its own defaults.

        Returns
        -------
        dict[str, Any]
            Flat kwargs dictionary for CmdStanPy APIs.
        """
        return_dict: dict[str, Any] = asdict(self)
        extras: dict[str, Any] = return_dict.pop("extra_kwargs", {}) or {}
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        return_dict.update(extras)
        return return_dict


@dataclass
class MCMCFitOptions(BaseFitOptions):
    """Common options for :meth:`cmdstanpy.CmdStanModel.sample`.

    Parameters
    ----------
    chains : int
        Number of Markov chains.
    parallel_chains : int
        Number of chains to run in parallel.
    threads_per_chain : int
        Number of threads used per chain.
    iter_warmup : int
        Warmup iterations per chain.
    iter_sampling : int
        Sampling iterations per chain.
    thin : int
        Thinning period.
    seed : int | list[int] | None
        RNG seed (single seed or one seed per chain).
    adapt_delta : float | None
        Target acceptance statistic for NUTS adaptation.
    max_treedepth : int | None
        Maximum tree depth for NUTS.
    show_progress : bool
        Whether to show sampling progress.
    """

    chains: int = 4
    parallel_chains: int = 4
    threads_per_chain: int = 1
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    thin: int = 1
    seed: int | list[int] | None = None
    adapt_delta: float | None = None
    max_treedepth: int | None = None
    show_progress: bool = False


@dataclass
class VBFitOptions(BaseFitOptions):
    """Common options for :meth:`cmdstanpy.CmdStanModel.variational`.

    Parameters
    ----------
    algorithm : str
        Variational algorithm (for example, ``"meanfield"`` or ``"fullrank"``).
    iter : int
        Maximum number of iterations.
    grad_samples : int
        Number of Monte Carlo gradient samples.
    elbo_samples : int
        Number of Monte Carlo ELBO samples.
    eta : float
        Stepsize scaling parameter.
    output_samples : int
        Number of approximate posterior draws to save.
    seed : int | None
        RNG seed.
    """

    algorithm: str = "meanfield"
    iter: int = 10000
    grad_samples: int = 1
    elbo_samples: int = 100
    eta: float = 1.0
    output_samples: int = 1000
    seed: int | None = None


@dataclass
class MLEFitOptions(BaseFitOptions):
    """Common options for :meth:`cmdstanpy.CmdStanModel.optimize`.

    Parameters
    ----------
    algorithm : str
        Optimization algorithm (for example, ``"lbfgs"``, ``"bfgs"``,
        or ``"newton"``).
    iter : int
        Maximum optimization iterations.
    seed : int | None
        RNG seed.
    jacobian : bool
        Whether to include Jacobian adjustment.
    """

    algorithm: str = "lbfgs"
    iter: int = 2000
    seed: int | None = None
    jacobian: bool = False


@dataclass
class PFFitOptions(BaseFitOptions):
    """Common options for :meth:`cmdstanpy.CmdStanModel.pathfinder`.

    Parameters
    ----------
    init_alpha : float | None
        Initial step size parameter for Pathfinder.
    tol_obj : float | None
        Absolute tolerance on the objective value.
    tol_rel_obj : float | None
        Relative tolerance on the objective value.
    tol_grad : float | None
        Absolute tolerance on the gradient norm.
    tol_rel_grad : float | None
        Relative tolerance on the gradient norm.
    tol_param : float | None
        Tolerance on parameter changes.
    history_size : int | None
        History size used by the underlying L-BFGS optimizer.
    num_paths : int | None
        Number of Pathfinder optimization paths.
    max_lbfgs_iters : int | None
        Maximum number of L-BFGS iterations per path.
    draws : int | None
        Number of draws returned from the Pathfinder approximation.
    num_elbo_draws : int | None
        Number of draws used for ELBO estimation.
    psis_resample : bool
        Whether to use PSIS resampling for returned draws.
    calculate_lp : bool
        Whether to calculate log probability values for draws.
    seed : int | None
        RNG seed.
    inits : dict[str, float] | float | os.PathLike | str | None
        Initial parameter values or path to an initialization file.
    show_console : bool
        Whether to stream CmdStan console output.
    refresh : int | None
        Frequency of progress messages written by CmdStan.
    num_threads : int | None
        Number of threads available to the CmdStan process.

    Notes
    -----
    This dataclass intentionally exposes only a small set of commonly used
    Pathfinder arguments. Less common options can be supplied through
    ``extra_kwargs``.
    """

    init_alpha: float | None = None
    tol_obj: float | None = None
    tol_rel_obj: float | None = None
    tol_grad: float | None = None
    tol_rel_grad: float | None = None
    tol_param: float | None = None
    history_size: int | None = None
    num_paths: int | None = None
    max_lbfgs_iters: int | None = None
    draws: int | None = None
    num_elbo_draws: int | None = None
    psis_resample: bool = True
    calculate_lp: bool = True
    seed: int | None = None
    inits: dict[str, float] | float | os.PathLike | str | None = None
    show_console: bool = False
    refresh: int | None = None
    num_threads: int | None = None


StanFitOptions: TypeAlias = Union[
    MCMCFitOptions, VBFitOptions, MLEFitOptions, PFFitOptions
]
"""Supported typed fit options for Stan methods."""
