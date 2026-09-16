"""Exercise real subprocess output and Jupyter display updates without a GPU."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import pytest

from granitewxc.temporal.progress import run_notebook_command, warnings_once


ROOT = Path(__file__).resolve().parents[1]


def notebook_displays(monkeypatch, on_display=None):
    import IPython
    import IPython.display

    displays = []

    def display(content, *, display_id):
        assert display_id is True
        updates = [content.data]
        displays.append(updates)
        if on_display:
            on_display()
        return SimpleNamespace(update=lambda content: updates.append(content.data))

    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", display)
    return displays


def test_live_subprocess_one_display_per_epoch_and_warnings_once(monkeypatch, tmp_path, capsys):
    # The child waits for a file written by the first display, proving that the
    # progress reaches the notebook before subprocess completion.
    acknowledgement = tmp_path / "bar_displayed"
    displays = notebook_displays(monkeypatch, lambda: acknowledgement.touch())
    script = '''
from pathlib import Path
import sys, time, warnings
from granitewxc.temporal.progress import emit_progress, warnings_once
with warnings_once(structured=True):
    for epoch in (1, 2):
        event = dict(epoch=epoch, epochs=2, total=3, completed=0, updates=0, loss=None)
        emit_progress(dict(event, event='epoch_start'))
        deadline = time.monotonic() + 10
        while not Path(sys.argv[1]).exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Progress was buffered until process exit')
            time.sleep(.01)
        for batch in (1, 2, 3):
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn('repeated warning', FutureWarning)
            event.update(completed=batch, updates=batch // 2, loss=.5)
            emit_progress(dict(event, event='batch'))
        emit_progress(dict(event, event='validation'))
        emit_progress(dict(event, event='epoch_end', val_loss=.25, status='complete'))
    warnings.warn('different warning', RuntimeWarning)
print('ordinary output survives')
'''
    seen = set()
    result = run_notebook_command(
        [sys.executable, "-u", "-c", script, str(acknowledgement)], cwd=ROOT, seen_warnings=seen,
    )
    assert result.returncode == 0
    assert len(displays) == 2
    for index, updates in enumerate(displays, 1):
        assert f"Epoch {index}/2" in updates[0]
        assert any("validating" in update for update in updates)
        assert "100%" in updates[-1] and "3/3" in updates[-1]
        assert "val_loss=0.25" in updates[-1] and "complete" in updates[-1]
    output = capsys.readouterr().out
    assert output.count("FutureWarning: repeated warning") == 1
    assert output.count("RuntimeWarning: different warning") == 1
    assert "ordinary output survives" in output
    assert "__TEMPORAL_EVENT__" not in output

    run_notebook_command(
        [sys.executable, "-u", "-c", "from granitewxc.temporal.progress import warnings_once; "
         "import warnings\nwith warnings_once(structured=True):\n "
         "warnings.warn('repeated warning', FutureWarning)"], cwd=ROOT, seen_warnings=seen,
    )
    assert "repeated warning" not in capsys.readouterr().out


def test_failed_child_preserves_error_and_closes_unfinished_bar(monkeypatch, capsys):
    displays = notebook_displays(monkeypatch)
    script = (
        "from granitewxc.temporal.progress import emit_progress; "
        "emit_progress(dict(event='epoch_start',epoch=1,epochs=1,total=4,completed=0,updates=0)); "
        "raise RuntimeError('child failed visibly')"
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_notebook_command([sys.executable, "-u", "-c", script], cwd=ROOT)
    assert error.value.returncode != 0
    assert "RuntimeError: child failed visibly" in capsys.readouterr().out
    assert len(displays) == 1 and "interrupted" in displays[0][-1]


def test_cell_interrupt_terminates_its_child(monkeypatch):
    import granitewxc.temporal.progress as module

    class InterruptedOutput:
        closed = False

        def __iter__(self):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    class Process:
        stdout = InterruptedOutput()
        terminated = False
        waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 1

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(KeyboardInterrupt):
        run_notebook_command(["unused"], cwd=ROOT)
    assert process.terminated and process.waited and process.stdout.closed


def test_warning_dedup_survives_filter_resets_and_preserves_warning_policy(recwarn):
    original = warnings.showwarning
    with warnings_once():
        for _ in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("same message", FutureWarning)
                warnings.warn("second message", RuntimeWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="keep warning-as-error"):
                warnings.warn("keep warning-as-error")
    assert warnings.showwarning is original
    assert [(item.category, str(item.message)) for item in recwarn] == [
        (FutureWarning, "same message"), (RuntimeWarning, "second message"),
    ]


def test_cli_routes_notebook_progress_to_callback(monkeypatch, tmp_path, capsys):
    from granitewxc.temporal import entrypoints, training
    from granitewxc.temporal.progress import emit_progress

    parser = entrypoints.build_parser()
    config = SimpleNamespace()
    cfg = SimpleNamespace()
    monkeypatch.setattr(entrypoints, "_load", lambda *args: (config, cfg))
    monkeypatch.setattr(entrypoints, "_device", lambda *args: "cpu")
    callbacks = []

    def train(config_arg, cfg_arg, **kwargs):
        assert config_arg is config and cfg_arg is cfg
        callbacks.append(kwargs["progress_callback"])
        return {"trained": True}

    monkeypatch.setattr(training, "train_temporal_model", train)
    entrypoints.cmd_train(parser.parse_args(["train", "--config", "unused", "--notebook-output"]))
    entrypoints.cmd_train(parser.parse_args(["train", "--config", "unused"]))
    assert callbacks == [emit_progress, None]
