"""
Module for custom exceptions related to BKT model fitting and validation.
"""

# defined with the fits so the fits package can raise it without importing the models
from stanbkt.fits.errors import FitMethodMismatchError

__all__ = ["FitMethodMismatchError"]
