"""Model classification enums for Stan BKT models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class ModelType(StrEnum):
    """Enumeration of BKT model types.

    Attributes
    ----------
    STANDARD : str
        Standard Bayesian Knowledge Tracing model (single parameter set across all students).
    GROUPED : str
        Grouped BKT model with parameters varying across student groups.
    NESTED : str
        Nested BKT model with hierarchical parameter structure.
    """

    STANDARD = "standard"
    GROUPED = "grouped"
    NESTED = "nested"


# TODO need better names for these, ASK Prof. Adam
class InitKnowledgeStrategy(StrEnum):
    """Enumeration of initial knowledge estimation strategies.

    Attributes
    ----------
    JOINT : str
        Jointly estimate initial knowledge from additional pre-test or prior data.
    CORRECTNESS_ONLY : str
        Estimate initial knowledge based only on correctness data.
    """

    JOINT = "joint"
    CORRECTNESS_ONLY = "correctness_only"


# if PosteriorPredictionOutput is changed update `_check_predict_posterior_args`.
PosteriorPredictionOutput = Literal["default", "summary", "stan"]
"""Type alias for the output format of posterior predictions. 
- 'default': Dictionary  mapping each KC to cleaned up DataFrames of posterior predictions (point estimate for MAP else draws)
- 'summary': Single DataFrame summarizing the fit results for each KC. 
- 'stan': Dictionary mapping each KC to the raw CmdStanGQ fit objects
the generated quantities predictions."""
