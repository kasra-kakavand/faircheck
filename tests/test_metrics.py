"""
Unit tests for fairness metrics module.

Tests the core fairness metrics calculations to ensure correctness.
"""

import numpy as np
import pandas as pd
import pytest

from faircheck.core.metrics import (
    FairnessMetrics,
    demographic_parity,
    disparate_impact,
    equal_opportunity,
    equalized_odds,
)


class TestFairnessMetrics:
    """Test the FairnessMetrics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = FairnessMetrics()

        # Perfect predictions, balanced groups
        self.perfect_y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        self.perfect_y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        self.perfect_groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        # Biased predictions
        self.biased_y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.biased_y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.biased_groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    def test_perfect_predictions(self):
        """Test that perfect predictions yield zero disparity."""
        results = self.metrics.evaluate(
            self.perfect_y_true, self.perfect_y_pred, self.perfect_groups
        )

        assert results["overall_accuracy"] == 1.0
        assert results["tpr_disparity"]["disparity"] == 0.0
        assert results["fpr_disparity"]["disparity"] == 0.0
        assert results["equalized_odds"] == 0.0

    def test_biased_predictions(self):
        """Test that biased predictions show disparity."""
        results = self.metrics.evaluate(
            self.biased_y_true, self.biased_y_pred, self.biased_groups
        )

        # Group A: all correct (TPR = 1.0)
        # Group B: all wrong (TPR = 0.0)
        assert results["tpr_disparity"]["disparity"] == 1.0

    def test_group_metrics_structure(self):
        """Test that group metrics DataFrame has correct structure."""
        results = self.metrics.evaluate(
            self.perfect_y_true, self.perfect_y_pred, self.perfect_groups
        )

        group_metrics = results["group_metrics"]
        assert isinstance(group_metrics, pd.DataFrame)

        expected_columns = [
            "group",
            "n_samples",
            "accuracy",
            "tpr",
            "fpr",
            "tnr",
            "fnr",
        ]
        for col in expected_columns:
            assert col in group_metrics.columns

    def test_disparity_bounds(self):
        """Test that disparities are between 0 and 1."""
        results = self.metrics.evaluate(
            self.biased_y_true, self.biased_y_pred, self.biased_groups
        )

        for key in ["tpr_disparity", "fpr_disparity", "accuracy_disparity"]:
            assert 0.0 <= results[key]["disparity"] <= 1.0

    def test_disparate_impact_perfect(self):
        """Test disparate impact with equal prediction rates."""
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        results = self.metrics.evaluate(y_pred, y_pred, groups)
        assert results["disparate_impact"] == 1.0


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_demographic_parity(self):
        """Test demographic parity function."""
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        dp = demographic_parity(y_pred, groups)

        # Group A: 50% positive rate
        # Group B: 25% positive rate
        # Difference: 0.25
        assert abs(dp - 0.25) < 0.001

    def test_equal_opportunity(self):
        """Test equal opportunity function."""
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        eo = equal_opportunity(y_true, y_pred, groups)

        # Group A TPR: 1.0
        # Group B TPR: 0.0
        # Disparity: 1.0
        assert eo == 1.0

    def test_disparate_impact(self):
        """Test disparate impact function."""
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        di = disparate_impact(y_pred, groups)

        # Group A: 100% positive rate
        # Group B: 25% positive rate
        # Ratio: 0.25
        assert abs(di - 0.25) < 0.001

    def test_equalized_odds(self):
        """Test equalized odds function."""
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        eo = equalized_odds(y_true, y_pred, groups)
        assert eo >= 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_group(self):
        """Test with only one demographic group."""
        metrics = FairnessMetrics()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        groups = np.array(["A", "A", "A", "A"])

        results = metrics.evaluate(y_true, y_pred, groups)

        # Single group should have zero disparity
        assert results["tpr_disparity"]["disparity"] == 0.0
        assert results["demographic_parity"] == 0.0

    def test_pandas_series_input(self):
        """Test that pandas Series inputs work."""
        metrics = FairnessMetrics()

        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 0, 1, 0, 1])
        groups = pd.Series(["A", "A", "A", "B", "B", "B"])

        results = metrics.evaluate(y_true, y_pred, groups)
        assert results["overall_accuracy"] == 1.0

    def test_invalid_metric(self):
        """Test that invalid metric names raise errors."""
        metrics = FairnessMetrics()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        groups = np.array(["A", "A", "B", "B"])

        results = metrics.evaluate(y_true, y_pred, groups)

        with pytest.raises(ValueError):
            metrics._compute_disparity(results["group_metrics"], "invalid_metric")
