from __future__ import annotations
from typing import Union
import pandas as pd
from stanbkt.fits.core.base import FitBase
from stanbkt.fits.fit_types import FitMethod


class VBFit(FitBase):
    """Fit class using Variational Bayes (VB) approximation.

    This class wraps CmdStanPy's variational Bayes algorithm to fit BKT models
    using fast approximate posterior inference.

    Inherits all state management from :class:`BaseFit`.
    """

    @property
    def _fit_method(self) -> FitMethod:
        """Return the fit method identifier.

        Returns
        -------
        FitMethod
            FitMethod.VB identifier.
        """
        return FitMethod.VB

    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        """Create initial values for variational inference.

        For VB, initial values are optional. This method returns
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
        """Generate summary statistics (not implemented for VB).

        Parameters
        ----------
        kcs : Union[list[str], str, None], optional
            Knowledge components to summarize.
        kc_col_name : str, default "kc_id"
            Column name for knowledge component identifier.
        percentiles : tuple[float, float], default (2.5, 97.5)
            Percentile bounds for confidence intervals.

        Returns
        -------
        pd.DataFrame
            Summary dataframe (implementation pending).
        """
