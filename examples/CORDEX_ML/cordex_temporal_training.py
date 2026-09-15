#!/usr/bin/env python
"""Temporal (sequence-conditioned) workflow for the CORDEX cases.

Thin wrapper over :mod:`granitewxc.temporal.entrypoints` so the CORDEX workflow
keeps a familiar entry point. All subcommands take a case YAML containing an
enabled ``temporal:`` block.

Examples
--------
Audit a config without loading weights::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py describe \
        --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml

Engineering checks on real data::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py check \
        --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml

Fine-tune::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py train \
        --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml

Inference on the held-out test period, then evaluate::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py infer \
        --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml \
        --split test
    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_training.py evaluate \
        --config examples/CORDEX_ML/SA_downscaling_refinement_T2_ACCESS-CM2_static_temporal_recurrent.yaml \
        --predictions <path printed by infer>
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from granitewxc.temporal.entrypoints import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(prog="cordex_temporal_training.py"))
