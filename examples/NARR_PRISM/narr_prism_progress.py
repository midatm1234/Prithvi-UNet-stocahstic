"""One terminal-style inference bar, updated in place in Jupyter or a terminal."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading

from tqdm import tqdm


class _NotebookProgressStream:
    """Send text redraws to one display ID instead of appending stream output."""

    def __init__(self, display):
        self.display = display
        self.handle = None

    def write(self, text):
        line = text.strip("\r\n ")
        if line:
            data = {"text/plain": line}
            if self.handle is None:
                self.handle = self.display(data, raw=True, display_id=True)
            else:
                self.handle.update(data, raw=True)
        return len(text)

    def flush(self):
        pass


def _progress_stream():
    try:
        from IPython import get_ipython
        from IPython.display import display

        if getattr(get_ipython(), "kernel", None) is not None:
            return _NotebookProgressStream(display), True
    except ImportError:
        pass
    return sys.stderr, sys.stderr.isatty()


@contextmanager
def inference_progress(progress_path, *, interval=1.0):
    """Monitor atomic worker counts while the caller runs its subprocess.

    The caller keeps child bars disabled and redirects diagnostics to a log.
    Only this bar is rendered; a failed run keeps its actual completion count.
    """
    progress_path = Path(progress_path)
    stream, enabled = _progress_stream()
    stop = threading.Event()
    with tqdm(
        total=None,
        desc="Refined inference (starting)",
        unit="tile-batch",
        file=stream,
        ascii=True,
        ncols=110,
        disable=not enabled,
    ) as progress:
        def refresh():
            try:
                state = json.loads(progress_path.read_text())
                completed, total = int(state["completed"]), int(state["total"])
            except (OSError, ValueError, KeyError, TypeError):
                return
            if not 0 <= completed <= total:
                return
            progress.total = total
            progress.set_description("Refined inference", refresh=False)
            progress.update(max(0, completed - progress.n))
            progress.refresh()

        def monitor():
            while not stop.wait(interval):
                refresh()

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            refresh()
