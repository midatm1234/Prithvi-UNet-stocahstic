#!/usr/bin/env python
"""Temporal (sequence-conditioned) workflow for the NARR/PRISM cases.

Thin wrapper over :mod:`granitewxc.temporal.entrypoints`, matching the style of
the existing ``narr_prism_training.py`` / ``narr_prism_refinement.py`` entry
points.

Prerequisite: the NARR/PRISM daily products must already exist under
``examples/NARR_PRISM/preprocessed/<case_name>/<mode>/`` (they are generated
locally by ``preproc_narr_prism.py`` and are deliberately not versioned). The
``describe`` subcommand reports clearly when they are absent.

Examples
--------
::

    mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py describe \
        --config examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml

    mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py train \
        --config examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_recurrent.yaml

    mamba run -n Prithvi python examples/NARR_PRISM/narr_prism_temporal.py infer \
        --config examples/NARR_PRISM/NARR_PRISM_subdomain_temporal_mamba.yaml --split test
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from granitewxc.temporal.entrypoints import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(prog="narr_prism_temporal.py"))
