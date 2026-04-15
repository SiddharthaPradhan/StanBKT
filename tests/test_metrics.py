"""Tests for stanbkt.utils.metrics (accuracy, rmse, auc)."""

from __future__ import annotations

import numpy as np
import pytest

from stanbkt.utils.metrics import accuracy, rmse, auc


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_all_correct_returns_one(self):
        y = [1, 0, 1, 0]
        p = [0.9, 0.1, 0.8, 0.2]
        assert accuracy(y, p) == pytest.approx(1.0)

    def test_all_wrong_returns_zero(self):
        y = [1, 0, 1, 0]
        p = [0.1, 0.9, 0.1, 0.9]
        assert accuracy(y, p) == pytest.approx(0.0)

    def test_half_correct_returns_half(self):
        y = [1, 1, 0, 0]
        p = [0.9, 0.1, 0.9, 0.1]  # 2nd and 4th wrong
        assert accuracy(y, p) == pytest.approx(0.5)

    def test_threshold_boundary_exactly_half(self):
        # probability == 0.5 is classified as positive (>=0.5)
        y = [1, 0]
        p = [0.5, 0.5]
        # 1st: predicted 1, label 1 → correct; 2nd: predicted 1, label 0 → wrong
        assert accuracy(y, p) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(accuracy([1, 0], [0.9, 0.1]), float)

    def test_accepts_numpy_arrays(self):
        y = np.array([1, 0, 1])
        p = np.array([0.8, 0.2, 0.7])
        result = accuracy(y, p)
        assert result == pytest.approx(1.0)

    def test_single_element(self):
        assert accuracy([1], [0.9]) == pytest.approx(1.0)
        assert accuracy([0], [0.6]) == pytest.approx(0.0)

    def test_all_zeros_correct(self):
        y = [0, 0, 0]
        p = [0.3, 0.2, 0.4]
        assert accuracy(y, p) == pytest.approx(1.0)

    def test_all_ones_correct(self):
        y = [1, 1, 1]
        p = [0.6, 0.7, 0.9]
        assert accuracy(y, p) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_perfect_predictions_return_zero(self):
        y = [1.0, 0.0, 1.0]
        p = [1.0, 0.0, 1.0]
        assert rmse(y, p) == pytest.approx(0.0)

    def test_known_value(self):
        # errors = [0.5, 0.5] → MSE = 0.25 → RMSE = 0.5
        y = [1, 0]
        p = [0.5, 0.5]
        assert rmse(y, p) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(rmse([1, 0], [0.9, 0.1]), float)

    def test_accepts_numpy_arrays(self):
        y = np.array([1.0, 0.0])
        p = np.array([0.8, 0.2])
        expected = float(np.sqrt(((1.0 - 0.8) ** 2 + (0.0 - 0.2) ** 2) / 2))
        assert rmse(y, p) == pytest.approx(expected)

    def test_symmetric(self):
        y = [1, 0, 1, 0]
        p = [0.8, 0.2, 0.7, 0.3]
        assert rmse(y, p) == rmse(p, y)

    def test_result_is_nonnegative(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=100)
        p = rng.random(size=100)
        assert rmse(y, p) >= 0.0

    def test_single_element_zero(self):
        assert rmse([1], [1.0]) == pytest.approx(0.0)

    def test_single_element_nonzero(self):
        assert rmse([0], [0.5]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# auc
# ---------------------------------------------------------------------------


class TestAUC:
    def test_perfect_classifier_returns_one(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        assert auc(y, p) == pytest.approx(1.0)

    def test_worst_classifier_returns_zero(self):
        y = [0, 0, 1, 1]
        p = [0.9, 0.8, 0.2, 0.1]
        assert auc(y, p) == pytest.approx(0.0)

    def test_random_classifier_near_half(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200)
        p = rng.random(size=200)
        result = auc(y, p)
        assert 0.35 <= result <= 0.65

    def test_returns_float(self):
        assert isinstance(auc([0, 1], [0.3, 0.7]), float)

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, size=50)
        p = rng.random(size=50)
        result = auc(y, p)
        assert 0.0 <= result <= 1.0

    def test_accepts_numpy_arrays(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])
        assert auc(y, p) == pytest.approx(1.0)

    def test_all_positive_raises(self):
        with pytest.raises(ValueError, match="AUC is undefined"):
            auc([1, 1, 1], [0.5, 0.6, 0.7])

    def test_all_negative_raises(self):
        with pytest.raises(ValueError, match="AUC is undefined"):
            auc([0, 0, 0], [0.5, 0.6, 0.7])

    def test_ties_handled(self):
        # All predictions equal — random tie-breaking, but AUC should be computable.
        y = [0, 1, 0, 1]
        p = [0.5, 0.5, 0.5, 0.5]
        result = auc(y, p)
        assert 0.0 <= result <= 1.0

    def test_known_two_point_case(self):
        # One positive, one negative, positive ranked first.
        y = [0, 1]
        p = [0.2, 0.8]
        assert auc(y, p) == pytest.approx(1.0)

    def test_known_two_point_reversed(self):
        y = [0, 1]
        p = [0.8, 0.2]
        assert auc(y, p) == pytest.approx(0.0)
