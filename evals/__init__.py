"""
Evaluation metrics for MuseTalk lip-sync quality assessment.

This module provides standard metrics for evaluating:
- Visual fidelity (FID)
- Identity preservation (CSIM)
- Lip synchronization (LSE-C)
"""

from .fid_metric import FIDMetric
from .csim_metric import CSIMMetric
from .lse_c_metric import LSECMetric

__all__ = ["FIDMetric", "CSIMMetric", "LSECMetric"]

