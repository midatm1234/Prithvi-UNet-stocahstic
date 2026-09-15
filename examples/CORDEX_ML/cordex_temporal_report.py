#!/usr/bin/env python
"""Render the bounded-comparison scorecard into a markdown results section.

Transcribing dozens of metrics by hand invites exactly the kind of arithmetic slip
this project has already had to correct once, so the results table in
``docs/temporal_model_results.md`` is generated from
``runs_temporal/experiment/{scorecard,manifest}.json`` and the per-variant
``evaluation.json`` files instead of being written out manually.

Usage::

    mamba run -n Prithvi python examples/CORDEX_ML/cordex_temporal_report.py \
        --experiment examples/CORDEX_ML/runs_temporal/experiment \
        --write docs/temporal_model_results.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# The rendered markdown contains non-Latin-1 glyphs (check marks, warning signs).
# A Windows console defaults to cp1252 and would raise UnicodeEncodeError on them,
# so make stdout/stderr UTF-8 regardless of the host code page.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PLACEHOLDER = "<!-- RESULTS_TABLE_PLACEHOLDER -->"

#: Metrics shown in the headline per-variable table, with display precision.
HEADLINE = [
    ("rmse", "RMSE", 4, "lower"),
    ("mae", "MAE", 4, "lower"),
    ("bias", "bias", 4, "abs-lower"),
    ("spatial_r_daily_mean", "daily spatial r", 4, "higher"),
    ("tendency_rmse", "tendency RMSE", 4, "lower"),
    ("tendency_std_ratio", "tendency std ratio", 4, "toward-1"),
    ("autocorr_lag1_error", "lag-1 ACF error", 5, "abs-lower"),
    ("autocorr_lag2_error", "lag-2 ACF error", 5, "abs-lower"),
    ("acc3d_rmse", "3-day acc. RMSE", 4, "lower"),
    ("acc5d_rmse", "5-day acc. RMSE", 4, "lower"),
    ("q0.99_bias", "q99 bias", 4, "abs-lower"),
    ("std_ratio", "field std ratio", 4, "toward-1"),
]

#: Precipitation distribution deltas (prediction minus truth).
PRECIP_DELTAS = [
    ("wet_day_frequency", "wet-day freq. error", 4),
    ("p_wet_given_wet", "P(wet|wet) error", 4),
    ("p_wet_given_dry", "P(wet|dry) error", 4),
    ("wet_spell_mean", "wet-spell mean error (d)", 3),
    ("dry_spell_mean", "dry-spell mean error (d)", 3),
]


def _paired(report: dict, var: str) -> dict:
    return (report.get("variables", {}).get(var, {}) or {}).get("paired", {}) or {}


def _delta(report: dict, var: str) -> dict:
    return (report.get("variables", {}).get(var, {}) or {}).get("distribution_delta", {}) or {}


def _fmt(value: Any, places: int) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:+.{places}f}" if places else f"{value}"


def _better(name: str, direction: str, variant: float | None, base: float | None) -> str:
    """Mark improvement relative to the baseline, per the metric's own direction."""
    if variant is None or base is None:
        return ""
    if not (np.isfinite(variant) and np.isfinite(base)):
        return ""
    if direction == "lower":
        good = variant < base
    elif direction == "higher":
        good = variant > base
    elif direction == "abs-lower":
        good = abs(variant) < abs(base)
    elif direction == "toward-1":
        good = abs(variant - 1.0) < abs(base - 1.0)
    else:
        return ""
    return " ✓" if good else ""


def render(experiment: Path) -> str:
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    scorecard_path = experiment / "scorecard.json"
    scorecard = (
        json.loads(scorecard_path.read_text(encoding="utf-8"))
        if scorecard_path.is_file()
        else {"variants": {}}
    )

    order = [v for v in ("baseline", "spatial_ft", "time_only", "convgru", "mamba")
             if (experiment / v / "evaluation.json").is_file()]
    reports = {
        name: json.loads((experiment / name / "evaluation.json").read_text(encoding="utf-8"))
        for name in order
    }
    if not reports:
        return "*No completed variants found; the experiment has not produced any evaluation.json yet.*"

    variables = list(next(iter(reports.values())).get("variables", {}))
    lines: list[str] = []

    missing = [v for v in ("baseline", "spatial_ft", "time_only", "convgru", "mamba") if v not in order]
    if missing:
        lines.append(
            f"> ⚠️ **Incomplete run.** These variants did not finish and are absent from "
            f"the tables below: `{'`, `'.join(missing)}`. Any conclusion that would need "
            "them is not drawn.\n"
        )

    run = manifest.get("variants", {})
    lines.append("### Run record\n")
    lines.append("| variant | trained | optimizer steps | train (s) | inference (s) | test dates | final val loss |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in order:
        info = run.get(name, {})
        hist = info.get("history") or []
        val = hist[-1]["validation"]["total"] if hist else None
        lines.append(
            f"| `{name}` | {'yes' if info.get('trained') else 'no'} "
            f"| {info.get('trained_temporal_steps', '—')} "
            f"| {info.get('train_seconds', 0):.0f} "
            f"| {info.get('inference_seconds', 0):.0f} "
            f"| {info.get('n_dates', '—')} "
            f"| {'—' if val is None else f'{val:.5f}'} |"
        )
    lines.append("")
    lines.append(
        f"Test period: {manifest.get('test_period')}. Every variant used seed "
        f"{1234}, the same splits, the same per-frame loss and the same learning rates.\n"
    )

    # ---- headline metrics ----
    for var in variables:
        lines.append(f"### `{var}` — held-out test period (event-paired)\n")
        header = "| metric | " + " | ".join(f"`{n}`" for n in order) + " |"
        lines.append(header)
        lines.append("|---" * (len(order) + 1) + "|")
        base = _paired(reports["baseline"], var) if "baseline" in reports else {}
        for key, label, places, direction in HEADLINE:
            cells = []
            for name in order:
                value = _paired(reports[name], var).get(key)
                mark = "" if name == "baseline" else _better(key, direction, value, base.get(key))
                cells.append(_fmt(value, places) + mark)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        if var.lower() in {"pr", "ppt", "precip", "precipitation", "tp"}:
            for key, label, places in PRECIP_DELTAS:
                cells = []
                for name in order:
                    value = _delta(reports[name], var).get(key)
                    mark = "" if name == "baseline" else _better(
                        key, "abs-lower", value, _delta(reports["baseline"], var).get(key)
                    )
                    cells.append(_fmt(value, places) + mark)
                lines.append(f"| {label} | " + " | ".join(cells) + " |")

        for key, label in (("rmse_boundary", "RMSE boundary"), ("rmse_interior", "RMSE interior")):
            cells = []
            for name in order:
                value = (reports[name].get("variables", {}).get(var, {}) or {}).get(key)
                cells.append(_fmt(value, 4))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        seam = []
        for name in order:
            s = (reports[name].get("variables", {}).get(var, {}) or {}).get("seams", {}) or {}
            seam.append(_fmt(s.get("seam_jump_ratio"), 4))
        lines.append("| chunk-seam jump ratio | " + " | ".join(seam) + " |")
        lines.append("")
        lines.append("✓ marks a value better than `baseline` in that metric's own direction "
                     "(lower for errors, |.| for signed errors, toward 1 for std ratios).\n")

    # ---- pre-registered verdict ----
    lines.append("### Pre-registered acceptance verdict\n")
    if not scorecard.get("variants"):
        lines.append("*Scorecard not yet written.*\n")
    else:
        lines.append("| variant | primary | guardrail | beats controls | verdict |")
        lines.append("|---|---|---|---|---|")
        for name, entry in scorecard["variants"].items():
            verdict = entry.get("scientific_acceptance")
            label = {True: "**ACCEPTED**", False: "**NOT ACCEPTED**", None: "control"}[verdict]
            beats = entry.get("beats_all_controls")
            lines.append(
                f"| `{name}` | {'pass' if entry.get('pass_primary') else 'FAIL'} "
                f"| {'pass' if entry.get('pass_guardrail') else 'FAIL'} "
                f"| {'—' if beats is None else ('yes' if beats else 'no')} "
                f"| {label} |"
            )
        lines.append("")
        for name, entry in scorecard["variants"].items():
            fails: list[str] = []
            for var, detail in entry.get("variables", {}).items():
                for group in ("primary", "guardrail"):
                    for lab, v in (detail.get(group) or {}).items():
                        if isinstance(v, dict) and v.get("status") == "FAIL":
                            fails.append(
                                f"`{var}` / {lab}: baseline {v['baseline']:.5g} → "
                                f"variant {v['variant']:.5g}"
                            )
            if fails:
                lines.append(f"**`{name}` failing checks ({len(fails)}):**\n")
                lines.extend(f"* {f}" for f in fails)
                lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="examples/CORDEX_ML/runs_temporal/experiment")
    ap.add_argument("--write", default=None, help="Markdown file whose placeholder to replace")
    args = ap.parse_args()

    section = render(Path(args.experiment))
    print(section)

    if args.write:
        target = Path(args.write)
        text = target.read_text(encoding="utf-8")
        if PLACEHOLDER in text:
            text = text.replace(PLACEHOLDER, section)
        else:
            # Idempotent re-render: replace everything between the section heading
            # and the next top-level heading.
            marker = "### Run record"
            if marker in text:
                head, _, rest = text.partition(marker)
                _, sep, tail = rest.partition("\n---\n")
                text = head + section + ("\n---\n" + tail if sep else "")
            else:
                raise SystemExit(
                    f"{target} contains neither {PLACEHOLDER!r} nor a '### Run record' "
                    "section to replace."
                )
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"\n[write] {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
