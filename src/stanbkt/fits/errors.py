"""
Module for custom exceptions related to fit methods.
"""


class FitMethodMismatchError(ValueError, TypeError):
    """Raised when a fit, fit options or saved fit artifacts use a different fit method than the model.

    Subclasses both ``ValueError`` and ``TypeError`` so code that caught the
    previous built-in exceptions keeps working.
    """
