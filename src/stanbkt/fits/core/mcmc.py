from __future__ import annotations
from typing import Union
import pandas as pd
from stanbkt.fits.core.base import BaseFit
from stanbkt.fits.fit_types import FitMethod


class MCMCFit(BaseFit):
    @property
    def _fit_method(self) -> FitMethod:
        return FitMethod.MCMC

    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        if isinstance(kc, list):
            return {kc_name: {} for kc_name in kc}
        return {}

    # TODO: add
    def summary(self, kcs: Union[list[str], str, None] = None) -> pd.DataFrame:
        """Generate a summary DataFrame for the specified KCs.
        Parameters
        ----------
        kcs : Union[list[str], str, None], optional
            The knowledge components (KCs) to include in the summary. If None, includes all KCs.
            Can be a single KC as a string or a list of KCs.
        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing the fit results for the specified KCs.
        """

        pass
