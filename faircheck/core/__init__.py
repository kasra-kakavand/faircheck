"""
Core functionality for FairCheck.

This module contains the main audit function and fairness metrics
that form the foundation of the library.
"""

from faircheck.core.audit import audit
from faircheck.core.metrics import FairnessMetrics

__all__ = ["audit", "FairnessMetrics"]
