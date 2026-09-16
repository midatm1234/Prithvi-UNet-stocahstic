"""Additional corrected analysis; never overwrite an archived scorecard."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cordex_temporal_experiment import (SCORER_VERSION, ACCEPTANCE, MUST_BEAT,
    EXTRA_MUST_BEAT, CANDIDATES, score_acceptance, beats_controls)


def rescore(root: Path, expected: list[str] | None = None):
    manifest = json.loads((root / "manifest.json").read_text())
    plan_known = bool(expected or manifest.get("planned_variants"))
    expected = expected or manifest.get("planned_variants") or list(manifest["variants"])
    invalidated = [name for name in expected if manifest.get("variants", {}).get(name, {}).get("status") in {"invalidated", "failed", "interrupted", "running", "not started"}]
    reports = {}
    evaluation_paths = {}
    for name in expected:
        recorded = manifest.get("variants", {}).get(name, {}).get("evaluation")
        path = Path(recorded) if recorded else root / name / "evaluation.json"
        if recorded and not path.is_absolute() and not path.is_file():
            path = root / path
        # A recorded recovery path is authoritative. Do not fall back to an
        # older canonical evaluation when that recorded artifact is missing.
        evaluation_paths[name] = str(path)
        if path.is_file():
            reports[name] = json.loads(path.read_text(encoding="utf-8-sig"))
    variables = list(reports.get("baseline", {}).get("variables", {}))
    missing = sorted(set(expected) - set(reports))
    out = {"scorer_version": SCORER_VERSION, "experiment": str(root),
           "corrections": ["Dotted quantile metric keys are resolved correctly",
                           "Missing or nonfinite required metrics fail instead of silently passing",
                           "Missing control metrics cannot count as beaten"],
           "acceptance_thresholds_changed": False, "expected_variants": expected,
           "evaluation_paths": evaluation_paths,
           "missing_variants": missing, "invalid_or_unfinished_variants": invalidated,
           "plan_known": plan_known, "complete": plan_known and not missing and not invalidated and bool(variables),
           "variables": variables, "variants": {}}
    if "baseline" not in reports:
        return out
    for name in reports:
        if name == "baseline":
            continue
        result = score_acceptance(reports["baseline"], reports[name], variables)
        if name in CANDIDATES:
            controls = MUST_BEAT + EXTRA_MUST_BEAT.get(name, ())
            result["versus_controls"] = beats_controls(reports, name, controls, variables)
            result["must_beat"] = list(controls)
            result["must_beat_missing"] = [n for n in controls if n not in reports]
            result["beats_all_controls"] = all(v.get("beats") is True for v in result["versus_controls"].values())
            result["scientific_acceptance"] = bool(out["complete"] and result["accepted"] and result["beats_all_controls"])
        else:
            result["scientific_acceptance"] = None
        out["variants"][name] = result
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", nargs="+")
    args = ap.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(f"Preserving prior analysis: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = rescore(Path(args.experiment), args.expected)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(target), "complete": result["complete"],
                      "verdicts": {k: v["scientific_acceptance"] for k, v in result["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
