"""Progress-bar contract for Phase-2 refinement training."""

from __future__ import annotations

import torch

import granitewxc.refinement.training as refinement_training
from granitewxc.refinement.training import RefinementTrainer
from refinement_fixtures import make_batch
from test_refinement_models import build


class _RecordingProgress:
    instances: list["_RecordingProgress"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.closed = False
        self.postfixes: list[dict[str, str]] = []
        self.instances.append(self)

    def update(self, amount: int) -> None:
        self.n += amount

    def set_postfix(self, *args, **kwargs) -> None:
        del args
        kwargs.pop("refresh", None)
        self.postfixes.append(dict(kwargs))

    def close(self) -> None:
        self.closed = True


def test_fit_creates_exactly_one_combined_progress_bar_per_epoch(
    monkeypatch, tmp_path
):
    _RecordingProgress.instances.clear()
    monkeypatch.setattr(refinement_training, "tqdm", _RecordingProgress)

    batch = make_batch(height=16, width=16)
    model = build("flow_matching_unet", batch)
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=1.0e-3)
    trainer = RefinementTrainer(
        model,
        optimizer,
        checkpoint_dir=str(tmp_path),
        log_every=1,
        logger=lambda _message: None,
    )
    monkeypatch.setattr(trainer, "save", lambda **_kwargs: str(tmp_path))

    trainer.fit(
        [batch, batch],
        [batch, batch, batch],
        num_epochs=2,
        limit_steps_train=1,
        limit_steps_valid=2,
    )

    assert len(_RecordingProgress.instances) == 2
    for epoch_index, progress in enumerate(
        _RecordingProgress.instances, start=1
    ):
        assert progress.kwargs["total"] == 3
        assert progress.kwargs["desc"] == f"Epoch {epoch_index:03d}/002"
        assert progress.kwargs["leave"] is True
        assert progress.kwargs["mininterval"] == 1.0
        assert progress.n == 3
        assert progress.closed is True
        assert any(item.get("stage") == "train" for item in progress.postfixes)
        assert any(
            item.get("stage") == "validation"
            for item in progress.postfixes
        )
        assert progress.postfixes[-1]["stage"] == "complete"
