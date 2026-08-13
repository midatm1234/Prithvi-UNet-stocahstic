"""Backward-compatible import path for the downscaling fine-tune models.

The implementation moved to :mod:`granitewxc.models.cordex_finetune_model`
when the original ECCC workflow was generalized to CORDEX, MERRA–PRISM, and
NARR–PRISM.  Keep this small alias module so configurations and external code
written against the original default branch do not fail after an additive
branch synchronization.
"""

from granitewxc.models.cordex_finetune_model import (
    ClimateDownscaleFinetuneModel,
    ClimateDownscaleFinetuneUNETModel,
    ClimateECCCFinetuneWrapper,
)

__all__ = [
    "ClimateDownscaleFinetuneModel",
    "ClimateDownscaleFinetuneUNETModel",
    "ClimateECCCFinetuneWrapper",
]
