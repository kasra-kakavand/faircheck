"""
FairCheck: Fairness for AI, made simple.

A comprehensive Python library for auditing, measuring, and mitigating
demographic bias in machine learning models.

Example:
    >>> from faircheck import audit
    >>> results = audit(model, dataset, sensitive_attrs=['gender'])
    >>> results.report()
"""

from faircheck.version import __version__

# Core API - what users will import
from faircheck.core.audit import audit
from faircheck.core.metrics import (
    FairnessMetrics,
    demographic_parity,
    equalized_odds,
    equal_opportunity,
    disparate_impact,
)

__all__ = [
    "__version__",
    "audit",
    "FairnessMetrics",
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "disparate_impact",
]

# Library metadata
__author__ = "Kasra Kakavand"
__email__ = "kasrakakavand@gmail.com"
__license__ = "MIT"
__description__ = "Fairness for AI, made simple."
