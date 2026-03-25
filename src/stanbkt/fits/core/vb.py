from __future__ import annotations
from typing import Union
import pandas as pd
from stanbkt.fits.core.base import BaseFit
from stanbkt.fits.fit_types import FitMethod


class VBFit(BaseFit):
    @property
    def _fit_method(self) -> FitMethod:
        return FitMethod.VB

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
        pass
