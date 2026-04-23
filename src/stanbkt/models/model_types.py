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
