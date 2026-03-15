"""
Module for custom exceptions related to BKT model fitting and validation.
"""


class FitMethodMismatchError(ValueError):
    """Raised when attempting to refit a model with a different fit method."""
