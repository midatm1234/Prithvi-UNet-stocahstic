"""Live epoch progress and warning output shared by the CLI and notebook."""
from contextlib import contextmanager
from html import escape
import json
import os
import subprocess
import sys
import warnings

from tqdm import tqdm


EVENT_PREFIX = "__TEMPORAL_EVENT__ "


def emit_progress(event):
    """Send a flushed, single-line event to the notebook's subprocess reader."""
    print(EVENT_PREFIX + json.dumps(event, allow_nan=False), flush=True)


@contextmanager
def warnings_once(*, structured=False):
    """Display each category/message once, even if libraries reset filters."""
    seen = set()
    with warnings.catch_warnings():
        original = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            key = (category.__name__, str(message))
            if key in seen:
                return
            seen.add(key)
            if structured:
                emit_progress({
                    "event": "warning", "category": category.__name__,
                    "message": str(message),
                    "text": warnings.formatwarning(message, category, filename, lineno, line),
                })
            else:
                original(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = showwarning
        yield


class _NotebookStream:
    """Update a single Jupyter output; works without ipywidgets extensions."""

    def __init__(self):
        self.handle = None

    def write(self, text):
        text = text.strip("\r\n")
        if not text.strip():
            return
        from IPython.display import HTML, display

        content = HTML('<pre style="white-space:pre-wrap">' + escape(text) + "</pre>")
        if self.handle is None:
            self.handle = display(content, display_id=True)
        else:
            self.handle.update(content)

    def flush(self):
        pass


class EpochProgress:
    """Render one tqdm bar per epoch, including validation status and losses."""

    def __init__(self, *, notebook=False):
        self.notebook = notebook
        self.bar = None

    def __call__(self, event):
        kind = event["event"]
        if kind == "epoch_start":
            self.close()
            self.bar = tqdm(
                total=event["total"], initial=event["completed"],
                desc=f"Epoch {event['epoch']}/{event['epochs']}", unit="batch",
                file=_NotebookStream() if self.notebook else sys.stderr,
                leave=True, mininterval=0.5, ncols=110,
            )
        if self.bar is None:
            return
        self.bar.total = event["total"]
        postfix = {"updates": event["updates"]}
        if event.get("loss") is not None:
            postfix["loss"] = f"{event['loss']:.4g}"
        if event.get("val_loss") is not None:
            postfix["val_loss"] = f"{event['val_loss']:.4g}"
        postfix["status"] = {
            "epoch_start": "training", "batch": "training",
            "validation": "validating", "epoch_end": "complete",
        }[kind]
        self.bar.set_postfix(postfix, refresh=False)
        self.bar.update(event["completed"] - self.bar.n)
        if kind != "batch":
            self.bar.refresh()
        if kind == "epoch_end":
            self.bar.close()
            self.bar = None

    def close(self, status="interrupted"):
        if self.bar is not None:
            self.bar.set_postfix_str(status, refresh=False)
            self.bar.close()
            self.bar = None


def run_notebook_command(command, *, cwd, seen_warnings=None):
    """Forward child output immediately and render structured epoch events.

    Pass a notebook-scoped set to remember warnings across train/resume/infer.
    Interrupting a cell also terminates the subprocess launched by that cell.
    """
    if seen_warnings is None:
        seen_warnings = set()
    from IPython import get_ipython

    shell = get_ipython()
    progress = EpochProgress(notebook=shell is not None and hasattr(shell, "kernel"))
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
    )
    try:
        for line in process.stdout:
            if line.startswith(EVENT_PREFIX):
                event = json.loads(line[len(EVENT_PREFIX):])
                if event["event"] == "warning":
                    key = (event["category"], event["message"])
                    if key not in seen_warnings:
                        seen_warnings.add(key)
                        print(event["text"], end="", flush=True)
                else:
                    progress(event)
            else:
                print(line, end="", flush=True)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
        progress.close()
    return subprocess.CompletedProcess(command, returncode)
