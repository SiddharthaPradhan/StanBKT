"""Model classification enums for Stan BKT models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class ModelType(StrEnum):
    STANDARD = "standard"
    GROUPED = "grouped"
    NESTED = "nested"


# TODO need better names for these, ASK Prof. Adam
class PriorEstimationType(StrEnum):
    JOINT = "joint"
    DEFAULT = "default"


# if PosteriorPredictionOutput is changed update `_check_predict_posterior_args`.
PosteriorPredictionOutput = Literal["default", "summary", "stan"]
"""Type alias for the output format of posterior predictions. 
- 'default': Dictionary  mapping each KC to cleaned up DataFrames of posterior predictions (point estimate for MAP else draws)
- 'summary': Single DataFrame summarizing the fit results for each KC. 
- 'stan': Dictionary mapping each KC to the raw CmdStanGQ fit objects
the generated quantities predictions."""
