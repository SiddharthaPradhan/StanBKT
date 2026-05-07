from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanPathfinder, CmdStanVB

from stanbkt.utils.summary_utils import summary_parameter_names

_INDEXED_NAME_PATTERN = re.compile(r"^(?P<base>.+)\[(?P<index>\d+)\]$")


def _apply_tight_layout(plot_object: Any) -> None:
    """Apply matplotlib tight layout to ArviZ plot outputs when possible."""
    figures: list[Any] = []
    if hasattr(plot_object, "figures") and plot_object.figures is not None:
        figures = list(plot_object.figures)
    elif hasattr(plot_object, "figure") and plot_object.figure is not None:
        figures = [plot_object.figure]

    for figure in figures:
        if hasattr(figure, "tight_layout"):
            figure.tight_layout()

    if not figures:
        # Fallback for plotting backends that only expose the current figure.
        plt.tight_layout()


def _normalized_parameter_names(parameter_names: Sequence[str]) -> list[str]:
    indexed_counts: dict[str, int] = {}
    for name in parameter_names:
        match = _INDEXED_NAME_PATTERN.match(name)
        if match is not None:
            base_name = match.group("base")
            indexed_counts[base_name] = indexed_counts.get(base_name, 0) + 1

    normalized: list[str] = []
    for name in parameter_names:
        match = _INDEXED_NAME_PATTERN.match(name)
        if match is None:
            normalized.append(name)
            continue

        base_name = match.group("base")
        index = match.group("index")
        if index == "1" and indexed_counts.get(base_name, 0) == 1:
            normalized.append(base_name)
        else:
            normalized.append(name)
    return normalized


def _posterior_group_from_draws(
    fit: CmdStanVB | CmdStanPathfinder,
) -> dict[str, np.ndarray]:
    raw_parameter_names = summary_parameter_names(fit.column_names)
    if not raw_parameter_names:
        raise ValueError("No posterior parameters found in fit.")
    parameter_names = _normalized_parameter_names(raw_parameter_names)

    if isinstance(fit, CmdStanVB):
        draws_matrix = fit.variational_sample_pd[raw_parameter_names].to_numpy(
            dtype=float
        )
    else:
        draws_matrix = fit.draws()

    return {
        name: np.asarray(draws_matrix[:, index], dtype=float)[np.newaxis, :]
        for index, name in enumerate(parameter_names)
    }


def _to_inference_data(fit: Any) -> Any:
    """Convert a single-KC CmdStan fit to ArviZ ``InferenceData``."""
    if isinstance(fit, CmdStanMLE):
        raise ValueError(
            "MLE fits do not contain posterior draws, so plot_dist and plot_trace are not supported."
        )

    try:
        if isinstance(fit, CmdStanMCMC):
            return az.from_cmdstanpy(posterior=fit)
        if isinstance(fit, (CmdStanVB, CmdStanPathfinder)):
            return az.from_dict({"posterior": _posterior_group_from_draws(fit)})
        return az.from_cmdstanpy(posterior=fit)
    except Exception as exc:  # pragma: no cover - defensive error path
        raise ValueError(
            "Could not convert fit to ArviZ InferenceData. "
            "Ensure `fit` is a valid CmdStanPy fit object for a single KC."
        ) from exc


def _resolve_var_names(idata: Any, params: Sequence[str] | None) -> list[str]:
    available = sorted(str(name) for name in idata.posterior.data_vars)
    if not available:
        raise ValueError("No posterior parameters found in fit.")

    if params is None:
        return available

    requested = [str(param) for param in params]
    resolved: list[str] = []
    missing: list[str] = []

    for name in requested:
        if name in available:
            resolved.append(name)
            continue

        indexed_matches = [
            candidate
            for candidate in available
            if (match := _INDEXED_NAME_PATTERN.match(candidate))
            and match.group("base") == name
        ]
        if indexed_matches:
            resolved.extend(indexed_matches)
            continue

        missing.append(name)

    if missing:
        raise ValueError(
            "Unknown parameter(s): " f"{missing}. Available parameters: {available}."
        )

    deduped_resolved = list(dict.fromkeys(resolved))
    return deduped_resolved


def plot_dist(
    fit: Any,
    params: Sequence[str] | None = None,
    *,
    ci_prob: float | None = None,
    ci_kind: str | None = None,
    col_wrap: int = 3,
) -> Any:
    """Plot posterior distributions for selected fit parameters.

    Parameters
    ----------
    fit : Any
        CmdStanPy fit object for one KC.
    params : Sequence[str] | None, optional
        Parameter names to plot. If ``None``, all posterior parameters are plotted.
    ci_prob : float | None, optional
        Probability mass to display in the credible interval.
    ci_kind : str | None, optional
        Credible interval type passed to ArviZ, such as ``"hdi"`` or ``"eti"``.
    col_wrap : int, default 3
        Number of columns before wrapping to a new row.

    Returns
    -------
    Any
        ArviZ plot object for the requested posterior parameters.
    """
    idata = _to_inference_data(fit)
    var_names = _resolve_var_names(idata, params)

    plot_object = az.plot_dist(
        idata,
        var_names=var_names,
        ci_prob=ci_prob,
        ci_kind=ci_kind,
        backend="matplotlib",
        col_wrap=col_wrap,
    )
    _apply_tight_layout(plot_object)
    return plot_object


def plot_trace(
    fit: Any,
    params: Sequence[str] | None = None,
    *,
    col_wrap: int = 3,
) -> Any:
    """Plot MCMC trace diagnostics for selected fit parameters.

    Parameters
    ----------
    fit : Any
        CmdStanPy fit object for one KC.
    params : Sequence[str] | None, optional
        Parameter names to trace. If ``None``, all posterior parameters are plotted.
    col_wrap : int, default 3
        Number of parameter panels before wrapping to a new row.

    Returns
    -------
    Any
        ArviZ plot object returned by ``arviz.plot_trace``.
    """
    idata = _to_inference_data(fit)
    var_names = _resolve_var_names(idata, params)
    plot_object = az.plot_trace(
        idata,
        var_names=var_names,
        backend="matplotlib",
        col_wrap=col_wrap,
    )
    _apply_tight_layout(plot_object)
    return plot_object
