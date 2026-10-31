"""Typed fit-option dataclasses for CmdStanPy fit methods.

This module defines a small set of strongly-typed, commonly used fitting options.
These options are intentionally conservative: only frequently used arguments are exposed as explicit dataclass fields, while
less-common (or future) CmdStanPy keyword arguments can still be passed via
``extra_kwargs``. It also defines the default options for each fit method usually None.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import os
from typing import Any, Self, TypeAlias, Union
from stanbkt.fits.fit_types import FitMethod


@dataclass
class BaseFitOptions:
    """Base dataclass for typed Stan fit options.

    Parameters
    ----------
    seed : int | None
        RNG seed for reproducibility.
    opencl_ids : tuple[int, int] | None
        OpenCL (platform id, device id) to run on. Requires the model to be compiled
        with ``cpp_compile_kwargs={"STAN_OPENCL": True}``.
    grainsize : int
        Number of students per slice of the parallel (``reduce_sum``) likelihood. ``-1`` (default)
        picks it per KC: one slice when running on a single thread, otherwise about four slices
        per thread. Any positive value is used as given. Not forwarded to CmdStanPy.
    extra_kwargs : dict[str, Any]
        Additional keyword arguments forwarded directly to CmdStanPy.
        These keys are merged last and therefore override generated defaults
        from dataclass fields.

    Notes
    -----
    Subclasses can add strongly typed fields and use ``extra_kwargs`` for less common or future CmdStanPy options.
    """

    seed: int | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    opencl_ids: tuple[int, int] | None = None
    grainsize: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.grainsize, bool) or not isinstance(self.grainsize, int) or (
            self.grainsize != -1 and self.grainsize < 1
        ):
            raise ValueError(
                f"grainsize must be -1 (automatic) or a positive integer, got {self.grainsize!r}."
            )

    def _likelihood_threads(self) -> int:
        """Threads available to the likelihood of a single fit."""
        return 1

    def resolve_grainsize(self, n_students: int) -> int:
        """Return the ``reduce_sum`` grainsize for a KC with ``n_students`` students.

        Parameters
        ----------
        n_students : int
            Number of students in the KC.

        Returns
        -------
        int
            The user grainsize, or with ``-1`` one slice per KC on a single thread and
            about four slices per thread otherwise.
        """
        if self.grainsize != -1:
            return self.grainsize
        n_students = max(1, int(n_students))
        n_threads = max(1, self._likelihood_threads())
        if n_threads == 1:
            return n_students
        # several slices per thread let the scheduler rebalance, as the slices with the longest sequences cost the most
        return max(1, -(-n_students // (4 * n_threads)))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Create fit options from a dictionary.

        Known dataclass fields are extracted and used for instantiation.
        Remaining keys are stored in ``extra_kwargs`` for CmdStanPy.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary containing fit options. Keys matching dataclass fields
            are assigned to those fields; remaining keys go to ``extra_kwargs``.

        Returns
        -------
        Self
            New instance of the fit options class with fields populated from dict.
        """
        field_names = {f.name for f in fields(cls)}

        known = {k: v for k, v in d.items() if k in field_names}
        extra = {k: v for k, v in d.items() if k not in field_names}

        obj = cls(**known)
        obj.extra_kwargs.update(extra)
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Convert options to a CmdStanPy kwargs dictionary.

        ``None`` values are removed so CmdStanPy can apply its own defaults.

        Returns
        -------
        dict[str, Any]
            Flat kwargs dictionary for CmdStanPy APIs.
        """
        return_dict: dict[str, Any] = asdict(self)
        # grainsize is passed to Stan as data, not as a CmdStanPy argument
        return_dict.pop("grainsize", None)
        extras: dict[str, Any] = return_dict.pop("extra_kwargs", {}) or {}
        return_dict.update(extras)
        # Remove any keys with None values so CmdStanPy can apply its own defaults
        # also allows for more cache hits.
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        # sort by key for consistent ordering (more cache hits and easier viewing)
        return_dict = dict(sorted(return_dict.items()))
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
    save_warmup : bool
        Whether to save warmup samples.
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
    show_console : bool
        Whether to stream CmdStan console output.
    opencl_ids : tuple[int, int] | None
        OpenCL (platform id, device id), see :class:`BaseFitOptions`.
    grainsize : int
        Students per ``reduce_sum`` slice, ``-1`` (default) picks it per KC. See :class:`BaseFitOptions`.
    extra_kwargs : dict[str, Any]
        Additional keyword arguments forwarded directly to CmdStanPy.
    """

    chains: int = 4
    parallel_chains: int = 4
    threads_per_chain: int = 1
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    save_warmup: bool | None = None
    thin: int | None = None
    # only MCMC takes one seed per chain
    seed: int | list[int] | None = None  # ty:ignore[invalid-attribute-override]
    adapt_delta: float | None = None
    max_treedepth: int | None = None
    show_progress: bool = True
    show_console: bool = False

    def _likelihood_threads(self) -> int:
        return self.threads_per_chain or 1


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
    draws : int
        Number of approximate posterior draws to save.
    seed : int | None
        RNG seed.
    require_converged : bool, default=True
        Whether to raise if the variational algorithm does not converge.
    opencl_ids : tuple[int, int] | None
        OpenCL (platform id, device id), see :class:`BaseFitOptions`.
    grainsize : int
        Students per ``reduce_sum`` slice, ``-1`` (default) picks it per KC. See :class:`BaseFitOptions`.
    extra_kwargs : dict[str, Any]
        Additional keyword arguments forwarded directly to CmdStanPy.
    """

    algorithm: str = "meanfield"
    iter: int | None = None
    grad_samples: int | None = 1
    elbo_samples: int | None = None
    eta: float | None = None
    draws: int | None = None
    require_converged: bool = True


@dataclass
class MLEFitOptions(BaseFitOptions):
    """Common options for :meth:`cmdstanpy.CmdStanModel.optimize`.

    Parameters
    ----------
    algorithm : str
        Optimization algorithm (for example, ``"lbfgs"``, ``"bfgs"``,
        or ``"newton"``).
    iter : int
        Maximum number of optimization iterations.
    jacobian : bool
        Whether to include Jacobian adjustment for constrained parameters.
    tol_obj : float | None
        Convergence tolerance on changes in objective function value.
    tol_rel_obj : float | None
        Convergence tolerance on relative changes in objective function value.
    tol_grad : float | None
        Convergence tolerance on the norm of the gradient.
    tol_rel_grad : float | None
        Convergence tolerance on the relative norm of the gradient.
    tol_param : float | None
        Convergence tolerance on changes in parameter value.
    history_size : int | None
        History size for the L-BFGS Hessian approximation. Values of 5–10
        are usually sufficient; must be less than the parameter-space
        dimensionality.
    seed : int | None
        RNG seed for reproducibility.
    opencl_ids : tuple[int, int] | None
        OpenCL (platform id, device id), see :class:`BaseFitOptions`.
    grainsize : int
        Students per ``reduce_sum`` slice, ``-1`` (default) picks it per KC. See :class:`BaseFitOptions`.
    extra_kwargs : dict[str, Any]
        Additional keyword arguments forwarded directly to CmdStanPy.
    """

    algorithm: str | None = None
    iter: int = 2000
    jacobian: bool = False
    tol_obj: float | None = None
    tol_rel_obj: float | None = None
    tol_grad: float | None = None
    tol_rel_grad: float | None = None
    tol_param: float | None = None
    history_size: int | None = None


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
    opencl_ids : tuple[int, int] | None
        OpenCL (platform id, device id), see :class:`BaseFitOptions`.
    grainsize : int
        Students per ``reduce_sum`` slice, ``-1`` (default) picks it per KC. See :class:`BaseFitOptions`.
    extra_kwargs : dict[str, Any]
        Additional keyword arguments forwarded directly to CmdStanPy.

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
    inits: dict[str, float] | float | os.PathLike | str | None = None
    show_console: bool = False
    refresh: int | None = None
    num_threads: int | None = None

    def _likelihood_threads(self) -> int:
        return self.num_threads or 1


StanFitOptions: TypeAlias = Union[
    MCMCFitOptions, VBFitOptions, MLEFitOptions, PFFitOptions
]
"""Supported typed fit options for Stan methods."""
