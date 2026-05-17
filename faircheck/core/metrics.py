"""
Fairness metrics for machine learning models.

This module implements a comprehensive suite of fairness metrics including:
- Group-wise performance metrics (TPR, FPR, Accuracy)
- Disparity measures (max - min across groups)
- Equalized Odds violation
- Demographic Parity difference
- Disparate Impact ratio
- Variance-based fairness regularization

All metrics support both numpy arrays and pandas Series as input.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# Type aliases for clarity
ArrayLike = Union[np.ndarray, pd.Series, List]


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator for binary classification.

    Computes per-group performance metrics and various disparity measures
    to evaluate fairness across demographic groups.

    Example:
        >>> metrics = FairnessMetrics()
        >>> results = metrics.evaluate(
        ...     y_true=labels,
        ...     y_pred=predictions,
        ...     sensitive_attr=demographics
        ... )
    """

    def __init__(self):
        """Initialize the fairness metrics calculator."""
        pass

    def evaluate(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sensitive_attr: ArrayLike,
        group_name: Optional[str] = None,
    ) -> Dict:
        """
        Perform comprehensive fairness evaluation.

        Args:
            y_true: Ground truth labels (binary: 0 or 1)
            y_pred: Predicted labels (binary: 0 or 1)
            sensitive_attr: Demographic group identifiers
            group_name: Optional name for the demographic dimension

        Returns:
            Dictionary containing:
                - overall_accuracy: Overall classification accuracy
                - group_metrics: Per-group performance DataFrame
                - tpr_disparity: True Positive Rate disparity
                - fpr_disparity: False Positive Rate disparity
                - accuracy_disparity: Accuracy disparity across groups
                - equalized_odds: Equalized Odds violation
                - demographic_parity: Demographic Parity difference
                - disparate_impact: Disparate Impact ratio
        """
        # Convert inputs to numpy arrays for consistency
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_attr = np.asarray(sensitive_attr)

        # Compute per-group metrics
        group_metrics_df = self._compute_group_metrics(y_true, y_pred, sensitive_attr)

        # Compute disparities
        tpr_disparity = self._compute_disparity(group_metrics_df, "tpr")
        fpr_disparity = self._compute_disparity(group_metrics_df, "fpr")
        accuracy_disparity = self._compute_disparity(group_metrics_df, "accuracy")

        # Compute composite metrics
        eo_violation = (tpr_disparity["disparity"] + fpr_disparity["disparity"]) / 2
        dp_difference = self._demographic_parity_difference(y_pred, sensitive_attr)
        di_ratio = self._disparate_impact_ratio(y_pred, sensitive_attr)

        # Overall accuracy
        overall_accuracy = np.mean(y_true == y_pred)

        return {
            "group_name": group_name,
            "overall_accuracy": float(overall_accuracy),
            "group_metrics": group_metrics_df,
            "tpr_disparity": tpr_disparity,
            "fpr_disparity": fpr_disparity,
            "accuracy_disparity": accuracy_disparity,
            "equalized_odds": float(eo_violation),
            "demographic_parity": float(dp_difference),
            "disparate_impact": float(di_ratio),
        }

    def _compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute performance metrics for each demographic group.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_attr: Demographic groups

        Returns:
            DataFrame with metrics per group
        """
        groups = np.unique(sensitive_attr)
        results = []

        for group in groups:
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # Calculate confusion matrix components
            tp = int(np.sum((y_true_group == 1) & (y_pred_group == 1)))
            fp = int(np.sum((y_true_group == 0) & (y_pred_group == 1)))
            tn = int(np.sum((y_true_group == 0) & (y_pred_group == 0)))
            fn = int(np.sum((y_true_group == 1) & (y_pred_group == 0)))

            # Calculate metrics with safe division
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            results.append({
                "group": group,
                "n_samples": int(np.sum(mask)),
                "accuracy": accuracy,
                "tpr": tpr,
                "fpr": fpr,
                "tnr": tnr,
                "fnr": fnr,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            })

        return pd.DataFrame(results)

    def _compute_disparity(
        self,
        group_metrics: pd.DataFrame,
        metric: str,
    ) -> Dict:
        """
        Calculate disparity (max - min) for a specific metric.

        Args:
            group_metrics: DataFrame from _compute_group_metrics()
            metric: Metric name to analyze

        Returns:
            Dictionary with disparity statistics
        """
        if metric not in group_metrics.columns:
            raise ValueError(f"Metric '{metric}' not found in group metrics")

        values = group_metrics[metric].values
        max_val = float(np.max(values))
        min_val = float(np.min(values))

        return {
            "metric": metric,
            "max": max_val,
            "min": min_val,
            "disparity": max_val - min_val,
            "max_group": str(group_metrics.loc[group_metrics[metric].idxmax(), "group"]),
            "min_group": str(group_metrics.loc[group_metrics[metric].idxmin(), "group"]),
        }

    def _demographic_parity_difference(
        self,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> float:
        """
        Compute Demographic Parity (DP) difference.

        DP measures whether positive predictions are equally distributed
        across demographic groups, regardless of true labels.
        """
        groups = np.unique(sensitive_attr)
        pred_rates = []

        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                pred_rates.append(np.mean(y_pred[mask]))

        if len(pred_rates) < 2:
            return 0.0

        return float(max(pred_rates) - min(pred_rates))

    def _disparate_impact_ratio(
        self,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> float:
        """
        Compute Disparate Impact ratio.

        The 80% rule: ratio should be at least 0.8 to be considered fair.
        Ratio of min positive prediction rate to max positive prediction rate.
        """
        groups = np.unique(sensitive_attr)
        pred_rates = []

        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                pred_rates.append(np.mean(y_pred[mask]))

        if len(pred_rates) < 2 or max(pred_rates) == 0:
            return 1.0

        return float(min(pred_rates) / max(pred_rates))


# Module-level convenience functions
def demographic_parity(y_pred: ArrayLike, sensitive_attr: ArrayLike) -> float:
    """
    Calculate demographic parity difference.

    Args:
        y_pred: Predicted labels
        sensitive_attr: Demographic groups

    Returns:
        Demographic parity difference (0 = perfect parity)
    """
    metrics = FairnessMetrics()
    return metrics._demographic_parity_difference(
        np.asarray(y_pred), np.asarray(sensitive_attr)
    )


def equalized_odds(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> float:
    """
    Calculate equalized odds violation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Demographic groups

    Returns:
        Equalized odds violation (0 = perfect equality)
    """
    metrics = FairnessMetrics()
    results = metrics.evaluate(y_true, y_pred, sensitive_attr)
    return results["equalized_odds"]


def equal_opportunity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> float:
    """
    Calculate equal opportunity difference (TPR disparity).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Demographic groups

    Returns:
        TPR disparity across groups
    """
    metrics = FairnessMetrics()
    results = metrics.evaluate(y_true, y_pred, sensitive_attr)
    return results["tpr_disparity"]["disparity"]


def disparate_impact(
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> float:
    """
    Calculate disparate impact ratio.

    The 80% rule: ratio should be at least 0.8 to be considered fair.

    Args:
        y_pred: Predicted labels
        sensitive_attr: Demographic groups

    Returns:
        Disparate impact ratio
    """
    metrics = FairnessMetrics()
    return metrics._disparate_impact_ratio(
        np.asarray(y_pred), np.asarray(sensitive_attr)
    )
