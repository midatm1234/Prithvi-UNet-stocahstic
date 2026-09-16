"""Run the bounded comparison from an immutable source snapshot and record exit.

Invocation: python this_script.py --snapshot <source-root> --out <new-run-root>
Source snapshot construction is deliberately separate from execution.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
import warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    snapshot, out = Path(args.snapshot).resolve(), Path(args.out).resolve()
    if (out / "execution.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "source_snapshot": str(snapshot), "output": str(out),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    (out / "execution.json").write_text(json.dumps(status, indent=2))
    sys.path.insert(0, str(snapshot))
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")
    exit_code = 0
    try:
        script = snapshot / "examples/CORDEX_ML/cordex_temporal_experiment.py"
        sys.argv = [str(script), "--out", str(out), "--steps", "600", "--val-steps", "60",
                    "--epochs", "1", "--test-years", "3", "--variants", "baseline", "spatial_ft",
                    "time_only", "native_pair", "native_pair_nohistory", "native_pair_pretext"]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        status["status"] = "completed"
    except BaseException as exc:
        exit_code = 1
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest.get("variants", {}).values():
                if entry.get("status") == "running":
                    entry.update(status=status["status"], error=status["error"])
            manifest_path.write_text(json.dumps(manifest, indent=2))
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status["exit_code"] = exit_code
        (out / "execution.json").write_text(json.dumps(status, indent=2))
    if status["status"] == "completed":
        # Reports and figures derive only from saved artifacts after completion.
        try:
            report_path = snapshot / "examples/CORDEX_ML/cordex_temporal_report.py"
            spec = importlib.util.spec_from_file_location("bounded_report", report_path)
            report = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(report)
            (out / "report.md").write_text(report.render(out), encoding="utf-8")
            plot_path = snapshot / "examples/CORDEX_ML/cordex_temporal_native_pair_plots.py"
            sys.argv = [str(plot_path), "--experiment", str(out), "--out", str(out / "figures")]
            try:
                runpy.run_path(str(plot_path), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
        except BaseException:
            error = traceback.format_exc()
            (out / "reporting_error.txt").write_text(error)
            status.update(status="reporting_failed", exit_code=1, reporting_error=error)
            (out / "execution.json").write_text(json.dumps(status, indent=2))
            raise
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
