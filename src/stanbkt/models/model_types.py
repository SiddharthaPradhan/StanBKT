"""Model classification enums for Stan BKT models."""

from __future__ import annotations

from enum import Enum


class ModelType(str, Enum):
    STANDARD = "standard"
    GROUPED = "grouped"
    NESTED = "nested"


class PriorEstimationType(str, Enum):
    JOINT = "joint"
    DEFAULT = "default"
