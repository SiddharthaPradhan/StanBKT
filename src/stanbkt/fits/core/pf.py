from __future__ import annotations
from typing import Union
import pandas as pd
from stanbkt.fits.core.base import BaseFit
from stanbkt.fits.fit_types import FitMethod


class PathfinderFit(BaseFit):
    @property
    def _fit_method(self) -> FitMethod:
        return FitMethod.PATHFINDER

    def _create_inits(self, kc: Union[list[str], str, None] = None) -> object:
        if isinstance(kc, list):
            return {kc_name: {} for kc_name in kc}
        return {}

    def summary(self, kc: Union[list[str], str]) -> pd.DataFrame:
        pass
