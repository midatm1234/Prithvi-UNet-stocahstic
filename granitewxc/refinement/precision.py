"""Apply the explicitly configured CUDA arithmetic policy."""
from __future__ import annotations

import torch


def configure_refinement_precision(performance) -> dict[str, bool]:
    """Set both CUDA backends; disabling TF32 must override PyTorch defaults.

    PyTorch may enable TF32 for cuDNN convolutions while disabling it for
    matrix multiplication. Autocast being disabled does not disable TF32.
    These backend flags are process-wide, as in the scientific training CLI.
    """
    enabled = bool(performance.precision.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
