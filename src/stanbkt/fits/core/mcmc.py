from __future__ import annotations
from stanbkt.utils.verbose import VerbosityLevel
from typing import Union
import pandas as pd
from stanbkt.fits.core.base import BaseFit
from stanbkt.fits.fit_types import FitMethod

from cmdstanpy import CmdStanMCMC


class MCMCFit(BaseFit):
    @property
    def _fit_method(self) -> FitMethod:
        return FitMethod.MCMC

    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        if isinstance(kc, list):
            return {kc_name: {} for kc_name in kc}
        return {}

    def _summary(
        self,
        kcs: Union[list[str], str, None] = None,
        kc_col_name: str = "kc_id",
        percentiles: tuple[float, float] = (2.5, 97.5),
    ) -> pd.DataFrame:
        """Generate a summary DataFrame for the specified KCs.

        Parameters
        ----------

        kcs : Union[list[str], str, None], optional
            The knowledge components (KCs) to include in the summary. If None, includes all KCs.
            Can be a single KC as a string or a list of KCs.
        percentiles : tuple[float, float], Defaults to (2.5, 97.5).
            The lower and upper percentiles for the summary. Values should be in range [1, 99].

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing the fit results for the specified KCs.
        """
        if self._should_cache_summary:
            self._clear_summary_cache_if_stale(percentiles)
        if isinstance(kcs, str):
            kcs_set = set([kcs])
        elif kcs is None:
            kcs_set = set(self.stan_fits.keys())
        else:
            kcs_set = set(kcs)
        available_kcs = set(self.stan_fits.keys())
        if len(kcs_set - available_kcs) > 0:
            self._print(
                f"Warning: The following KCs were requested for summary but have not been fitted: {kcs_set - available_kcs}. Skipping these KCs in the summary.",
                level=VerbosityLevel.WARN,
            )
        elif kcs_set.isdisjoint(available_kcs):
            raise ValueError(
                f"No valid KCs found for summary generation. Requested KCs: {kcs_set}. Available KCs: {available_kcs}."
            )
        # only keep KCs that have been fitted and are available in stan_fits
        kcs_set = kcs_set.intersection(available_kcs)
        summary_frames: list[pd.DataFrame] = []
        for kc, kc_stan_fit in self.stan_fits.items():
            # skip KCs that were not requested
            if kc not in kcs_set:
                continue
            if kc_stan_fit is None:
                raise ValueError(
                    f"Stan fit for KC '{kc}' is not available. Failed to generate summary."
                )
            # check cache before generating summary
            if kc in self._summary_cache:
                summary_frames.append(self._summary_cache[kc])
                continue
            if not isinstance(kc_stan_fit, CmdStanMCMC):
                raise ValueError(
                    f"Stan fit for KC '{kc}' is not a CmdStanMCMC object. Found type '{type(kc_stan_fit).__name__}'. Cannot generate summary."
                )
            # CMDStanpy's type signature for signature is wrong, as CMDStan Summary supports floats,
            # it is safe to ignore the type error here.
            summary_df: pd.DataFrame = kc_stan_fit.summary(
                percentiles=[
                    percentiles[0],
                    50,
                    percentiles[1],
                ]  # ty:ignore[invalid-argument-type]
            ).reset_index(names="parameter")
            summary_df.insert(0, kc_col_name, kc)
            summary_frames.append(summary_df)

            if self._should_cache_summary:
                # cache summary for future use
                self._update_summary_cache(kc, summary_df)
        if summary_frames:
            summary_concat_df: pd.DataFrame = pd.concat(summary_frames, axis=0)
            return summary_concat_df.set_index([kc_col_name, "parameter"])
        else:
            raise RuntimeError("No valid KCs found for summary generation.")

    def diagnose(self):
        pass
