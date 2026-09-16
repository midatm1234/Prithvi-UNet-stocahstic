"""Generate final regional figures and status after a bounded controller exits.

This process uses no GPU and never starts or resumes training. Its wait is bounded
and tied to the exact controller creation time, so a recycled PID cannot trigger
reports for an unrelated run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--wait-created-filetime", required=True, type=int)
    parser.add_argument("--max-wait-seconds", type=float, default=43200)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo, experiment, source, output = (p.resolve() for p in
        (args.repo, args.experiment, args.source, args.out))
    output.mkdir(parents=True, exist_ok=False)
    controller = load_module("bounded_report_wait", source / "cordex_temporal_continue_bounded.py")
    summary = load_module("bounded_report_summary", source / "cordex_temporal_takeover_summary.py")
    summary.ROOT = repo
    record_path = output / "execution.json"
    record = {
        "status": "waiting", "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": str(experiment), "source": str(source),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.glob("*.py"))},
        "wait_pid": args.wait_pid,
        "wait_created_filetime": args.wait_created_filetime,
        "training_started": False,
    }
    controller.atomic_json(record_path, record)
    try:
        def heartbeat(elapsed):
            record.update(wait_seconds=round(elapsed, 1),
                          updated_utc=datetime.now(timezone.utc).isoformat())
            controller.atomic_json(record_path, record)

        controller.wait_for_original(args.wait_pid, args.wait_created_filetime,
                                     args.max_wait_seconds, heartbeat)
        record["status"] = "reporting"
        controller.atomic_json(record_path, record)
        manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
        planned = tuple(manifest.get("planned_variants", ()))
        if planned != controller.VARIANTS:
            raise ValueError("Final figures require the recorded six-variant protocol")
        incomplete = [name for name in planned
                      if manifest["variants"].get(name, {}).get("status") != "completed"]
        if incomplete:
            raise RuntimeError("Training/evaluation remains incomplete: " + ", ".join(incomplete))
        card = json.loads(Path(manifest["scorecard_path"]).read_text(encoding="utf-8"))
        if card.get("complete") is not True:
            raise ValueError("Final scorecard is incomplete")
        regional = load_module("bounded_report_regions", source / "cordex_temporal_regional_report.py")
        record["regional_reports"] = {}
        for name in planned:
            destination = output / name
            regional.generate_report(experiment, name, destination, coordinates=args.coordinates)
            record["regional_reports"][name] = str(destination / "regional_metrics.json")
            controller.atomic_json(record_path, record)
        record.update(status="completed", exit_code=0)
        return 0
    except BaseException as exc:
        record.update(status="failed", exit_code=1,
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        controller.atomic_json(record_path, record)
        try:
            summary.main()
        except Exception as exc:
            record["summary_error"] = f"{type(exc).__name__}: {exc}"
        controller.atomic_json(record_path, record)


if __name__ == "__main__":
    raise SystemExit(main())
