from __future__ import annotations
from stanbkt.utils.verbose import VerbosityLevel
from typing import Union
from cmdstanpy import CmdStanPathfinder
import pandas as pd
from stanbkt.fits.core.base import FitBase
from stanbkt.utils.summary_utils import summarize_draws, summary_parameter_names
from stanbkt.fits.fit_types import FitMethod


class PathfinderFit(FitBase):
    """Fit class using Pathfinder variational approximation.

    This class wraps CmdStanPy's Pathfinder algorithm to fit BKT models using
    a fast variational approximation that explores the posterior geometry.

    Inherits all state management from :class:`BaseFit`.
    """

    @property
    def _fit_method(self) -> FitMethod:
        """Return the fit method identifier.

        Returns
        -------
        FitMethod
            FitMethod.PATHFINDER identifier.
        """
        return FitMethod.PATHFINDER

    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        """Create initial values for Pathfinder algorithm.

        For Pathfinder, initial values are optional. This method returns
        an empty dictionary structure.

        Parameters
        ----------
        kc : Union[list[str], str, None], optional
            Knowledge component identifier(s). If a list, creates keys for each KC.

        Returns
        -------
        object
            Dictionary mapping KC names to empty initialization dicts, or empty dict.
        """
        if isinstance(kc, list):
            return {kc_name: {} for kc_name in kc}
        return {}

    def _summary(
        self,
        kcs: Union[list[str], str, None] = None,
        kc_col_name: str = "kc_id",
        percentiles: tuple[float, float] = (2.5, 97.5),
    ) -> pd.DataFrame:
        """Generate summary statistics for the specified KCs.

        Parameters
        ----------
        kcs : Union[list[str], str, None], optional
            The knowledge components (KCs) to include in the summary. If None, includes all KCs.
            Can be a single KC as a string or a list of KCs.
        kc_col_name : str, default "kc_id"
            Column name for knowledge component identifier.
        percentiles : tuple[float, float], default (2.5, 97.5)
            Percentile bounds for credible intervals.

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing the fit results for the specified KCs.
        """
        if self._should_cache_summary:
            self._clear_summary_cache_if_stale(percentiles)
        if isinstance(kcs, str):
            kcs_set = {kcs}
        elif kcs is None:
            kcs_set = self.get_fitted_kcs()
        else:
            kcs_set = set(kcs)
        available_kcs = self.get_fitted_kcs()
        if len(kcs_set - available_kcs) > 0:
            self.log(
                f"Warning: The following KCs were requested for summary but have not been fitted: {kcs_set - available_kcs}. Skipping these KCs in the summary.",
                level=VerbosityLevel.WARN,
            )
        elif kcs_set.isdisjoint(available_kcs):
            raise ValueError(
                f"No valid KCs found for summary generation. Requested KCs: {kcs_set}. Available KCs: {available_kcs}."
            )

        kcs_set = kcs_set.intersection(available_kcs)
        summary_frames: list[pd.DataFrame] = []
        for kc in sorted(kcs_set):
            kc_stan_fit = self.get_fit(kc)
            if kc_stan_fit is None:
                raise ValueError(
                    f"Stan fit for KC '{kc}' is not available. Failed to generate summary."
                )
            if kc in self._summary_cache:
                summary_frames.append(self._summary_cache[kc])
                continue
            if not isinstance(kc_stan_fit, CmdStanPathfinder):
                raise ValueError(
                    f"Stan fit for KC '{kc}' is not a CmdStanPathfinder object. Found type '{type(kc_stan_fit).__name__}'. Cannot generate summary."
                )

            parameter_names = summary_parameter_names(kc_stan_fit.column_names)
            if not parameter_names:
                continue

            draws_df = pd.DataFrame(
                kc_stan_fit.draws(), columns=kc_stan_fit.column_names
            )
            missing = [name for name in parameter_names if name not in draws_df.columns]
            if missing:
                raise ValueError(
                    f"Unable to summarize Pathfinder fit for KC '{kc}'. Missing columns in pathfinder sample output: {missing}."
                )

            summary_df = summarize_draws(
                draws=draws_df[parameter_names].to_numpy(dtype=float),
                parameter_names=parameter_names,
                percentiles=percentiles,
            )
            summary_df.insert(0, kc_col_name, kc)
            summary_frames.append(summary_df)

            if self._should_cache_summary:
                self._update_summary_cache(kc, summary_df)

        if summary_frames:
            summary_concat_df: pd.DataFrame = pd.concat(summary_frames, axis=0)
            return summary_concat_df.set_index([kc_col_name, "parameter"])
        raise RuntimeError("No valid KCs found for summary generation.")
