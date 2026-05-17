"""
Main audit function for FairCheck.

This module provides the high-level `audit()` function that performs
comprehensive fairness analysis with a single function call.

Example:
    >>> from faircheck import audit
    >>> results = audit(model, dataset, sensitive_attrs=['gender'])
    >>> results.report()
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from faircheck.core.metrics import FairnessMetrics


class AuditResults:
    """
    Container for fairness audit results.

    Provides methods to display, export, and analyze fairness metrics
    computed during an audit.

    Attributes:
        metrics_by_attr: Dictionary mapping sensitive attribute names to metrics
        model_info: Information about the audited model
        dataset_info: Information about the dataset
    """

    def __init__(
        self,
        metrics_by_attr: Dict[str, Dict],
        model_info: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None,
    ):
        """
        Initialize audit results.

        Args:
            metrics_by_attr: Per-attribute fairness metrics
            model_info: Optional model metadata
            dataset_info: Optional dataset metadata
        """
        self.metrics_by_attr = metrics_by_attr
        self.model_info = model_info or {}
        self.dataset_info = dataset_info or {}

    def summary(self) -> str:
        """
        Generate a text summary of audit results.

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append("FAIRCHECK AUDIT RESULTS")
        lines.append("=" * 70)

        # Model info
        if self.model_info:
            lines.append("\nModel Information:")
            for key, value in self.model_info.items():
                lines.append(f"  {key}: {value}")

        # Dataset info
        if self.dataset_info:
            lines.append("\nDataset Information:")
            for key, value in self.dataset_info.items():
                lines.append(f"  {key}: {value}")

        # Per-attribute results
        for attr_name, metrics in self.metrics_by_attr.items():
            lines.append(f"\n{'-' * 70}")
            lines.append(f"Sensitive Attribute: {attr_name}")
            lines.append("-" * 70)

            lines.append(f"\n  Overall Accuracy: {metrics['overall_accuracy']:.2%}")

            lines.append("\n  Per-Group Performance:")
            group_df = metrics["group_metrics"]
            lines.append(
                group_df[["group", "n_samples", "accuracy", "tpr", "fpr"]].to_string(
                    index=False
                )
            )

            lines.append("\n  Disparities:")
            lines.append(
                f"    TPR Disparity:      {metrics['tpr_disparity']['disparity']:.3f}"
            )
            lines.append(
                f"    FPR Disparity:      {metrics['fpr_disparity']['disparity']:.3f}"
            )
            lines.append(
                f"    Accuracy Disparity: {metrics['accuracy_disparity']['disparity']:.3f}"
            )

            lines.append("\n  Composite Metrics:")
            lines.append(f"    Equalized Odds:      {metrics['equalized_odds']:.3f}")
            lines.append(f"    Demographic Parity:  {metrics['demographic_parity']:.3f}")
            lines.append(f"    Disparate Impact:    {metrics['disparate_impact']:.3f}")

            # Fairness assessment
            lines.append("\n  Fairness Assessment:")
            lines.append(self._assess_fairness(metrics))

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def _assess_fairness(self, metrics: Dict) -> str:
        """
        Provide a human-readable fairness assessment.

        Args:
            metrics: Metrics dictionary

        Returns:
            Assessment string
        """
        tpr_disp = metrics["tpr_disparity"]["disparity"]
        fpr_disp = metrics["fpr_disparity"]["disparity"]
        di_ratio = metrics["disparate_impact"]

        issues = []

        if tpr_disp > 0.1:
            issues.append(
                f"    [WARNING] High TPR disparity ({tpr_disp:.3f}) - "
                f"model performs unequally across groups for positive class"
            )

        if fpr_disp > 0.1:
            issues.append(
                f"    [WARNING] High FPR disparity ({fpr_disp:.3f}) - "
                f"unequal false positive rates across groups"
            )

        if di_ratio < 0.8:
            issues.append(
                f"    [WARNING] Disparate impact ratio ({di_ratio:.3f}) < 0.8 - "
                f"fails the 80% rule"
            )

        if not issues:
            return "    [OK] Model appears to meet basic fairness criteria"

        return "\n".join(issues)

    def to_dict(self) -> Dict:
        """
        Export results as a dictionary (JSON-serializable).

        Returns:
            Dictionary with all metrics
        """
        result = {
            "model_info": self.model_info,
            "dataset_info": self.dataset_info,
            "results": {},
        }

        for attr_name, metrics in self.metrics_by_attr.items():
            # Convert DataFrame to dict for JSON serialization
            metrics_copy = metrics.copy()
            if isinstance(metrics_copy.get("group_metrics"), pd.DataFrame):
                metrics_copy["group_metrics"] = metrics_copy[
                    "group_metrics"
                ].to_dict(orient="records")
            result["results"][attr_name] = metrics_copy

        return result

    def report(self, output: Optional[str] = None) -> str:
        """
        Generate a fairness report.

        Args:
            output: Optional file path to save report (txt or html)

        Returns:
            Report content as string
        """
        report_content = self.summary()

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"Report saved to: {output}")

        return report_content

    def __repr__(self) -> str:
        """Brief representation."""
        n_attrs = len(self.metrics_by_attr)
        attrs = ", ".join(self.metrics_by_attr.keys())
        return f"AuditResults(n_attributes={n_attrs}, attributes=[{attrs}])"

    def __str__(self) -> str:
        """String representation shows summary."""
        return self.summary()


def audit(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    sensitive_attrs: Union[Dict[str, Any], pd.DataFrame],
    model_info: Optional[Dict] = None,
    dataset_info: Optional[Dict] = None,
) -> AuditResults:
    """
    Perform a comprehensive fairness audit.

    This is the main entry point for FairCheck. Given true labels, predictions,
    and demographic attributes, it computes a complete set of fairness metrics
    and returns an AuditResults object.

    Args:
        y_true: Ground truth labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)
        sensitive_attrs: Either:
            - Dict mapping attribute names to arrays (e.g., {'gender': [...]})
            - DataFrame with sensitive attributes as columns
        model_info: Optional dictionary with model metadata
        dataset_info: Optional dictionary with dataset metadata

    Returns:
        AuditResults object containing all computed metrics

    Example:
        >>> import numpy as np
        >>> from faircheck import audit
        >>>
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> sensitive_attrs = {
        ...     'gender': ['male', 'female', 'male', 'female', 'male']
        ... }
        >>>
        >>> results = audit(y_true, y_pred, sensitive_attrs)
        >>> print(results.summary())
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have same length"
        )

    # Convert sensitive_attrs to dict format
    if isinstance(sensitive_attrs, pd.DataFrame):
        sensitive_attrs_dict = {
            col: sensitive_attrs[col].values for col in sensitive_attrs.columns
        }
    elif isinstance(sensitive_attrs, dict):
        sensitive_attrs_dict = sensitive_attrs
    else:
        raise TypeError(
            "sensitive_attrs must be a dict or DataFrame, "
            f"got {type(sensitive_attrs).__name__}"
        )

    # Validate each sensitive attribute
    for attr_name, attr_values in sensitive_attrs_dict.items():
        attr_values = np.asarray(attr_values)
        if len(attr_values) != len(y_true):
            raise ValueError(
                f"Sensitive attribute '{attr_name}' has {len(attr_values)} "
                f"samples, expected {len(y_true)}"
            )
        sensitive_attrs_dict[attr_name] = attr_values

    # Compute fairness metrics for each sensitive attribute
    metrics_calculator = FairnessMetrics()
    metrics_by_attr = {}

    for attr_name, attr_values in sensitive_attrs_dict.items():
        metrics = metrics_calculator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attr=attr_values,
            group_name=attr_name,
        )
        metrics_by_attr[attr_name] = metrics

    # Build dataset info if not provided
    if dataset_info is None:
        dataset_info = {
            "n_samples": len(y_true),
            "positive_rate": float(np.mean(y_true)),
            "sensitive_attributes": list(sensitive_attrs_dict.keys()),
        }

    return AuditResults(
        metrics_by_attr=metrics_by_attr,
        model_info=model_info,
        dataset_info=dataset_info,
    )
