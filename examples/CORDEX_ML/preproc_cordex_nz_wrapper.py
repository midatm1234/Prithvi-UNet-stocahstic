#!/usr/bin/env python
"""NZ-domain wrapper to run preproc_cordex.py for multiple predictor/output sets.

Fill in OUTPUT_DIRS and FILE_NAMES below. Each OUTPUT_DIRS entry is paired
with the corresponding FILE_NAMES entry (same length), yielding one
preproc_cordex.py invocation per pair. Paths are relative to NZ_TEST_ROOT.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ===================== USER PARAMETERS (EDIT ME) =====================
NZ_TEST_ROOT = "/mnt/data2/kyo/granite-wxc/granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/"

# Use paths relative to NZ_TEST_ROOT to keep the lists compact.

OUTPUT_DIRS = [
    "test/historical/predictors/perfect","test/historical/predictors/perfect",
        "test/historical/predictors/imperfect", "test/historical/predictors/imperfect",
        "test/mid_century/predictors/perfect","test/mid_century/predictors/perfect",
        "test/mid_century/predictors/imperfect", "test/mid_century/predictors/imperfect",
        "test/end_century/predictors/perfect","test/end_century/predictors/perfect",
        "test/end_century/predictors/imperfect", "test/end_century/predictors/imperfect"]

# File names (paired 1:1 with OUTPUT_DIRS).
FILE_NAMES = [
        "ACCESS-CM2_1981-2000.nc","EC-Earth3_1981-2000.nc",
        "ACCESS-CM2_1981-2000.nc","EC-Earth3_1981-2000.nc",
        "ACCESS-CM2_2041-2060.nc","EC-Earth3_2041-2060.nc",
        "ACCESS-CM2_2041-2060.nc","EC-Earth3_2041-2060.nc",
        "ACCESS-CM2_2080-2099.nc","EC-Earth3_2080-2099.nc",
        "ACCESS-CM2_2080-2099.nc","EC-Earth3_2080-2099.nc"
]
TARGET_SAMPLE = (
    "/mnt/data2/kyo/granite-wxc/granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/"
    "train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc"
)
OROGRAPHY_FILE = (
    "/mnt/data2/kyo/granite-wxc/granite-geospatial-wxc-downscaling/CORDEX/NZ_domain/"
    "train/ESD_pseudo_reality/predictors/Static_fields.nc"
)
OROGRAPHY_VAR = "orog"
INPUT_VARS = ["u", "v", "q", "t", "z"]
INPUT_LEVELS = ["850", "700", "500"]
REGRID_METHOD = "bilinear"
OVERWRITE = True
# ================================================================


def _validate_lists() -> None:
    if not OUTPUT_DIRS:
        raise ValueError("OUTPUT_DIRS is empty; add at least one entry.")
    if not FILE_NAMES:
        raise ValueError("FILE_NAMES is empty; add at least one entry.")
    if len(OUTPUT_DIRS) != len(FILE_NAMES):
        raise ValueError(
            "OUTPUT_DIRS and FILE_NAMES must have the same length "
            f"(got {len(OUTPUT_DIRS)} vs {len(FILE_NAMES)})"
        )


def _run_preproc(predictor_files: list[str], output_dir: str) -> None:
    if not predictor_files:
        raise ValueError("predictor_files entry is empty")

    predictor_files = [str(Path(NZ_TEST_ROOT) / p) for p in predictor_files]
    output_dir = str(Path(NZ_TEST_ROOT) / output_dir)

    script_path = Path(__file__).with_name("preproc_cordex.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--predictor-files",
        *predictor_files,
        "--target-sample",
        TARGET_SAMPLE,
        "--orography-file",
        OROGRAPHY_FILE,
        "--orography-var",
        OROGRAPHY_VAR,
        "--input-vars",
        *INPUT_VARS,
        "--input-levels",
        *INPUT_LEVELS,
        "--regrid-method",
        REGRID_METHOD,
        "--output-dir",
        output_dir,
    ]
    if OVERWRITE:
        cmd.append("--overwrite")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    _validate_lists()
    for idx, (output_dir, file_name) in enumerate(zip(OUTPUT_DIRS, FILE_NAMES), start=1):
        predictor_files = [str(Path(output_dir) / file_name)]
        print(f"[{idx}/{len(OUTPUT_DIRS)}] Output dir: {output_dir}")
        _run_preproc(predictor_files, str(output_dir))


if __name__ == "__main__":
    main()
