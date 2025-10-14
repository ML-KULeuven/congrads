import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn, optim

from congrads.checkpoints import CheckpointManager
from congrads.metrics import MetricManager


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def dummy_model():
    return nn.Linear(2, 1)


@pytest.fixture
def dummy_optimizer(dummy_model):
    return optim.SGD(dummy_model.parameters(), lr=0.01)


@pytest.fixture
def dummy_metrics():
    mm = MagicMock(spec=MetricManager)
    mm.aggregate.return_value = {"loss": torch.tensor(0.5)}
    return mm


@pytest.fixture
def always_improving():
    return lambda current, best: True


def test_initialization_creates_dir(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics, always_improving
):
    non_existent = os.path.join(temp_dir, "new_dir")
    assert not os.path.exists(non_existent)

    manager = CheckpointManager(
        always_improving,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        non_existent,
        create_dir=True,
    )

    assert os.path.exists(non_existent)
    assert isinstance(manager, CheckpointManager)


def test_initialization_raises_if_dir_missing(
    dummy_model, dummy_optimizer, dummy_metrics, always_improving
):
    with pytest.raises(FileNotFoundError):
        CheckpointManager(
            always_improving,
            dummy_model,
            dummy_optimizer,
            dummy_metrics,
            "/nonexistent/path",
            create_dir=False,
        )


def test_save_and_load_checkpoint(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics, always_improving
):
    manager = CheckpointManager(
        always_improving,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        temp_dir,
        create_dir=True,
    )

    # Save
    manager.best_metric_values = {"loss": torch.tensor(0.1)}
    manager.save(epoch=5)

    # Zero weights
    for param in dummy_model.parameters():
        param.data.zero_()

    # Load
    checkpoint = manager.load("checkpoint.pth")

    assert checkpoint["epoch"] == 5
    assert "loss" in manager.best_metric_values
    assert isinstance(manager.best_metric_values["loss"], torch.Tensor)


def test_resume_from_checkpoint(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics, always_improving
):
    manager = CheckpointManager(
        always_improving,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        temp_dir,
        create_dir=True,
    )
    manager.save(epoch=7)
    new_manager = CheckpointManager(
        always_improving, dummy_model, dummy_optimizer, dummy_metrics, temp_dir
    )
    epoch = new_manager.resume()
    assert epoch == 7


def test_resume_missing_with_ignore(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics, always_improving
):
    manager = CheckpointManager(
        always_improving,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        temp_dir,
        create_dir=True,
    )
    epoch = manager.resume(filename="missing.pth", ignore_missing=True)
    assert epoch == 0


def test_resume_missing_without_ignore(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics, always_improving
):
    manager = CheckpointManager(
        always_improving,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        temp_dir,
        create_dir=True,
    )
    with pytest.raises(FileNotFoundError):
        manager.resume(filename="missing.pth", ignore_missing=False)


def test_evaluate_criteria_saves_when_improving(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics
):
    def criteria(curr, best):
        return True  # Always save

    manager = CheckpointManager(
        criteria,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        temp_dir,
        create_dir=True,
    )
    manager.evaluate_criteria(epoch=3)

    checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
    assert os.path.exists(checkpoint_path)


def test_evaluate_criteria_does_not_save_when_not_improving(
    temp_dir, dummy_model, dummy_optimizer, dummy_metrics
):
    def criteria(curr, best):
        return False  # Never save

    manager = CheckpointManager(
        criteria,
        dummy_model,
        dummy_optimizer,
        dummy_metrics,
        temp_dir,
        create_dir=True,
    )
    manager.evaluate_criteria(epoch=3)

    checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
    assert not os.path.exists(checkpoint_path)
