"""Model classification enums for Stan BKT models."""

from __future__ import annotations

from enum import StrEnum


class ModelType(StrEnum):
    STANDARD = "standard"
    GROUPED = "grouped"
    NESTED = "nested"


# TODO need better names for these, ASK Prof. Adam
class PriorEstimationType(StrEnum):
    JOINT = "joint"
    DEFAULT = "default"
