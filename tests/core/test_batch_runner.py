from unittest.mock import ANY, MagicMock

import pytest
import torch
from torch import Tensor, nn

from congrads.core.batch_runner import BatchRunner

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_network():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, batch):
            batch["output"] = self.linear(batch["input"])
            return batch

    return Net()


@pytest.fixture
def criterion():
    # simple MSE-like loss
    def loss_fn(output, target, data=None):
        return ((output - target) ** 2).mean()

    return loss_fn


@pytest.fixture
def optimizer(simple_network):
    return torch.optim.SGD(simple_network.parameters(), lr=0.1)


@pytest.fixture
def constraint_engine():
    ce = MagicMock()
    ce.train = MagicMock(side_effect=lambda batch, loss: loss)
    ce.validate = MagicMock()
    ce.test = MagicMock()
    return ce


@pytest.fixture
def metric_manager():
    mm = MagicMock()
    mm.accumulate = MagicMock()
    return mm


@pytest.fixture
def callback_manager():
    cm = MagicMock()
    cm.run = MagicMock(side_effect=lambda hook, data: data)
    return cm


@pytest.fixture
def batch():
    return {
        "input": torch.randn(4, 3),
        "target": torch.randn(4, 1),
    }


@pytest.fixture
def runner(
    simple_network,
    criterion,
    optimizer,
    constraint_engine,
    metric_manager,
    callback_manager,
    device,
):
    return BatchRunner(
        network=simple_network,
        criterion=criterion,
        optimizer=optimizer,
        constraint_engine=constraint_engine,
        metric_manager=metric_manager,
        callback_manager=callback_manager,
        device=device,
    )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def test_to_device_moves_tensors(runner, batch, device):
    moved = runner._to_device(batch)

    for tensor in moved.values():
        assert isinstance(tensor, Tensor)
        assert tensor.device == device


def test_run_callbacks_passthrough_when_none(
    simple_network, criterion, optimizer, constraint_engine, device, batch
):
    runner = BatchRunner(
        network=simple_network,
        criterion=criterion,
        optimizer=optimizer,
        constraint_engine=constraint_engine,
        metric_manager=None,
        callback_manager=None,
        device=device,
    )

    out = runner._run_callbacks("any_hook", batch)
    assert out is batch


# ---------------------------------------------------------------------
# Training batch
# ---------------------------------------------------------------------


def test_train_batch_full_flow(runner, batch, constraint_engine, metric_manager, callback_manager):
    loss = runner.train_batch(batch)

    # loss returned
    assert isinstance(loss, Tensor)
    assert loss.requires_grad is True

    # callbacks order
    expected_hooks = [
        "on_train_batch_start",
        "after_train_forward",
        "on_train_batch_end",
    ]
    actual_hooks = [c.args[0] for c in callback_manager.run.call_args_list]
    for hook in expected_hooks:
        assert hook in actual_hooks

    # constraint engine used
    constraint_engine.train.assert_called_once()
    constraint_engine.validate.assert_not_called()
    constraint_engine.test.assert_not_called()

    # metric accumulation
    metric_manager.accumulate.assert_called_once()
    metric_manager.accumulate.assert_called_with("Loss/train", ANY)


def test_train_batch_updates_parameters(runner, batch):
    before = [p.detach().clone() for p in runner.network.parameters()]

    runner.train_batch(batch)

    after = list(runner.network.parameters())

    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=False)), (
        "Parameters did not update during training"
    )


# ---------------------------------------------------------------------
# Validation batch
# ---------------------------------------------------------------------


def test_valid_batch_flow(runner, batch, constraint_engine, metric_manager, callback_manager):
    loss = runner.valid_batch(batch)

    assert isinstance(loss, Tensor)
    assert loss.requires_grad is True

    # constraint engine validate only
    constraint_engine.validate.assert_called_once()
    constraint_engine.train.assert_not_called()
    constraint_engine.test.assert_not_called()

    # metrics
    metric_manager.accumulate.assert_called_once_with("Loss/valid", ANY)

    # callbacks
    hooks = [c.args[0] for c in callback_manager.run.call_args_list]
    assert "on_valid_batch_start" in hooks
    assert "after_valid_forward" in hooks
    assert "on_valid_batch_end" in hooks


# ---------------------------------------------------------------------
# Test batch
# ---------------------------------------------------------------------


def test_test_batch_flow(runner, batch, constraint_engine, metric_manager, callback_manager):
    loss = runner.test_batch(batch)

    assert isinstance(loss, Tensor)

    # constraint engine test only
    constraint_engine.test.assert_called_once()
    constraint_engine.train.assert_not_called()
    constraint_engine.validate.assert_not_called()

    # metrics
    metric_manager.accumulate.assert_called_once_with("Loss/test", ANY)

    # callbacks
    hooks = [c.args[0] for c in callback_manager.run.call_args_list]
    assert "on_test_batch_start" in hooks
    assert "after_test_forward" in hooks
    assert "on_test_batch_end" in hooks


# ---------------------------------------------------------------------
# Optional components
# ---------------------------------------------------------------------


def test_no_metric_manager_does_not_crash(
    simple_network, criterion, optimizer, constraint_engine, device, batch
):
    runner = BatchRunner(
        network=simple_network,
        criterion=criterion,
        optimizer=optimizer,
        constraint_engine=constraint_engine,
        metric_manager=None,
        callback_manager=None,
        device=device,
    )

    loss = runner.train_batch(batch)
    assert isinstance(loss, Tensor)


def test_no_callback_manager_does_not_crash(
    simple_network, criterion, optimizer, constraint_engine, metric_manager, device, batch
):
    runner = BatchRunner(
        network=simple_network,
        criterion=criterion,
        optimizer=optimizer,
        constraint_engine=constraint_engine,
        metric_manager=metric_manager,
        callback_manager=None,
        device=device,
    )

    loss = runner.valid_batch(batch)
    assert isinstance(loss, Tensor)
