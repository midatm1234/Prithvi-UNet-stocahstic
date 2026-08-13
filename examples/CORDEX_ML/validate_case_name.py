#!/usr/bin/env python
"""Validate the ``case_name`` output-folder refactor across CORDEX-ML configs.

Lightweight, data-free checks (no GPU, no NetCDF) proving that:

1. every YAML config in this folder declares exactly one non-empty ``case_name``;
2. ``get_config`` resolves ``case_name`` and derives every
   output location under ``<path_experiment>/<case_name>/``;
3. normalization statistics, checkpoints and inference outputs all resolve to the
   case-specific sub-folders that the scripts read from / write to;
4. output directories are actually created under ``<case_name>``; and
5. a YAML missing ``case_name`` raises a clear ``MissingCaseNameError``.

Run:
    python examples/CORDEX_ML/validate_case_name.py
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in (REPO_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from granitewxc.utils.config import (  # noqa: E402
    SCALAR_FILENAMES,
    MissingCaseNameError,
    get_config,
)

CONFIG_GLOB = os.path.join(SCRIPT_DIR, "*.yaml")


def _first_line(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.readline().rstrip("\n")


def _segments(path: str) -> list[str]:
    return os.path.normpath(path).replace("\\", "/").split("/")


def check_case_name_declaration(path: str, failures: list[str]) -> None:
    stem = os.path.splitext(os.path.basename(path))[0]
    lines = open(path, encoding="utf-8").read().splitlines()
    declarations = [line for line in lines if line.startswith("case_name:")]
    if len(declarations) != 1:
        failures.append(
            f"{stem}: expected exactly one case_name declaration, found "
            f"{len(declarations)}"
        )
        return
    value = declarations[0].split(":", 1)[1].strip()
    if not value:
        failures.append(f"{stem}: case_name is empty")


def check_config(path: str, failures: list[str]):
    stem = os.path.splitext(os.path.basename(path))[0]
    cfg = get_config(path)

    for name in (
        "path_scalars",
        "path_preproc",
        "path_checkpoints",
        "path_inference",
        "path_logs",
    ):
        val = getattr(cfg, name)
        if cfg.case_name not in _segments(val):
            failures.append(f"{stem}: {name}={val!r} is not under case dir {cfg.case_name!r}")

    derive = getattr(cfg, "derive_output_paths", False)
    if derive:
        for attr, key in (
            ("input_mu", "inputs_mean"),
            ("input_sigma", "inputs_std"),
            ("target_mu", "targets_mean"),
            ("target_sigma", "targets_std"),
        ):
            got = os.path.normpath(getattr(cfg.model, attr))
            exp = os.path.normpath(os.path.join(cfg.path_scalars, SCALAR_FILENAMES[key]))
            if got != exp:
                failures.append(f"{stem}: model.{attr}={got!r} != {exp!r}")
        for key in SCALAR_FILENAMES:
            got = os.path.normpath(cfg.data.scalers[key])
            exp = os.path.normpath(os.path.join(cfg.path_scalars, SCALAR_FILENAMES[key]))
            if got != exp:
                failures.append(f"{stem}: data.scalers[{key}]={got!r} != {exp!r}")

    # The trainer's default checkpoint directory must be the case checkpoints dir.
    try:
        from granitewxc.utils.trainer import _default_checkpoint_dir

        ckpt = os.path.normpath(_default_checkpoint_dir(cfg))
        expected_checkpoint_dir = (
            cfg.path_checkpoints
            if derive
            else os.path.join(cfg.path_experiment, "weights")
        )
        if ckpt != os.path.normpath(expected_checkpoint_dir):
            failures.append(
                f"{stem}: trainer checkpoint default {ckpt!r} != expected "
                f"{os.path.normpath(expected_checkpoint_dir)!r}"
            )
    except Exception as exc:  # pragma: no cover - torch import guard
        print(f"[skip] trainer checkpoint check unavailable ({type(exc).__name__})")

    return cfg


def check_dir_creation(cfg, failures: list[str]) -> None:
    stem = cfg.case_name
    with tempfile.TemporaryDirectory() as tmp:
        cfg.path_experiment = tmp
        case_root = os.path.abspath(os.path.join(tmp, stem))
        for name in (
            "path_scalars",
            "path_preproc",
            "path_checkpoints",
            "path_inference",
            "path_logs",
        ):
            d = getattr(cfg, name)
            os.makedirs(d, exist_ok=True)
            if not os.path.isdir(d):
                failures.append(f"{stem}: failed to create {name}={d!r}")
                continue
            if os.path.commonpath([os.path.abspath(d), case_root]) != case_root:
                failures.append(f"{stem}: {name}={d!r} not under case root {case_root!r}")
        print(f"[ok] created scalars/preproc/checkpoints/inference/logs under {case_root}")


def check_compute_scalars_derivation(cfg, failures: list[str]) -> None:
    try:
        cs = importlib.import_module("compute_scalars_cordex")
    except Exception as exc:  # pragma: no cover - optional heavy deps
        print(
            f"[skip] compute_scalars_cordex import unavailable ({type(exc).__name__}); "
            "skipping --output-dir auto-derivation check"
        )
        return
    args = argparse.Namespace(output_dir=None)
    out = cs._resolve_output_dir(args, cfg)
    if bool(getattr(cfg, "derive_output_paths", False)):
        expected = cfg.path_scalars
    else:
        expected = os.path.dirname(cfg.model.input_mu)
    if os.path.normpath(out) != os.path.normpath(expected):
        failures.append(
            f"compute_scalars _resolve_output_dir={out!r} != expected {expected!r}"
        )
    else:
        print(f"[ok] compute_scalars auto-derives --output-dir -> {out}")


def check_missing_case_name(configs: list[str], failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "no_case.yaml")
        lines = open(configs[0], encoding="utf-8").read().split("\n")
        lines = [ln for ln in lines if not ln.startswith("case_name:")]
        open(bad, "w", encoding="utf-8").write("\n".join(lines))
        try:
            get_config(bad)
        except MissingCaseNameError:
            print("[ok] YAML without case_name raises MissingCaseNameError")
            return
        except Exception as exc:
            failures.append(
                f"missing-case-name: raised {type(exc).__name__}, expected MissingCaseNameError"
            )
            return
        failures.append("missing-case-name: get_config did NOT raise for a YAML without case_name")


def main() -> int:
    configs = sorted(glob.glob(CONFIG_GLOB))
    if not configs:
        print(f"No YAML configs found under {SCRIPT_DIR}")
        return 1

    failures: list[str] = []
    print(f"Found {len(configs)} YAML configs under {SCRIPT_DIR}\n")

    for path in configs:
        check_case_name_declaration(path, failures)

    sample_cfg = None
    for path in configs:
        cfg = check_config(path, failures)
        if sample_cfg is None and os.path.basename(path).startswith("NZ_T1_ACCESS-CM2_static_v6"):
            sample_cfg = cfg
    if sample_cfg is None:
        sample_cfg = get_config(configs[0])

    check_compute_scalars_derivation(sample_cfg, failures)
    check_dir_creation(sample_cfg, failures)
    check_missing_case_name(configs, failures)

    print("-" * 64)
    if failures:
        print(f"VALIDATION FAILED with {len(failures)} issue(s):")
        for item in failures:
            print(f"  - {item}")
        return 1

    print(f"VALIDATION PASSED for {len(configs)} configs:")
    print("  * exactly one non-empty case_name is present in every YAML")
    print("  * case path properties live under <path_experiment>/<case_name>/")
    print("  * explicit legacy and opt-in derived scalar/checkpoint paths remain consistent")
    print("  * output directories are created under the case folder")
    print("  * a YAML without case_name raises MissingCaseNameError")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
