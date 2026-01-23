from unittest.mock import MagicMock, call

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from congrads.core.congradscore import CongradsCore

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_dataset():
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    return TensorDataset(x, y)


@pytest.fixture
def train_loader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=2)


@pytest.fixture
def valid_loader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=2)


@pytest.fixture
def test_loader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=2)


@pytest.fixture
def mock_descriptor():
    return MagicMock()


@pytest.fixture
def mock_constraints():
    c = MagicMock()
    c.name = "ConstraintA"
    return [c]


@pytest.fixture
def network():
    return nn.Linear(3, 1)


@pytest.fixture
def criterion():
    return nn.MSELoss()


@pytest.fixture
def optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=0.01)


@pytest.fixture
def metric_manager():
    mm = MagicMock()
    mm.register = MagicMock()
    return mm


@pytest.fixture
def callback_manager():
    cm = MagicMock()
    cm.run = MagicMock()
    return cm


@pytest.fixture
def checkpoint_manager():
    cm = MagicMock()
    cm.evaluate_criteria = MagicMock()
    cm.save = MagicMock()
    return cm


# ---------------------------------------------------------------------
# Constructor & wiring
# ---------------------------------------------------------------------


def test_congrads_core_initialization(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
):
    core = CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
    )

    assert core.network is network
    assert core.device == device
    assert core.constraints == mock_constraints
    assert core.epoch_runner is not None
    assert core.batch_runner is not None
    assert core.constraint_engine is not None


# ---------------------------------------------------------------------
# Metric initialization
# ---------------------------------------------------------------------


def test_metric_registration(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
    metric_manager,
):
    CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
        metric_manager=metric_manager,
    )

    # Loss metrics
    metric_manager.register.assert_any_call("Loss/train", "during_training")
    metric_manager.register.assert_any_call("Loss/valid", "during_training")
    metric_manager.register.assert_any_call("Loss/test", "after_training")

    # CSR metrics
    metric_manager.register.assert_any_call("CSR/train", "during_training")
    metric_manager.register.assert_any_call("CSR/valid", "during_training")
    metric_manager.register.assert_any_call("CSR/test", "after_training")

    # Per-constraint metrics
    metric_manager.register.assert_any_call("ConstraintA/train", "during_training")
    metric_manager.register.assert_any_call("ConstraintA/valid", "during_training")
    metric_manager.register.assert_any_call("ConstraintA/test", "after_training")


# ---------------------------------------------------------------------
# Fit loop behavior
# ---------------------------------------------------------------------


def test_fit_runs_epochs_and_calls_epoch_runner(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
):
    mock_epoch_runner = MagicMock()
    mock_epoch_runner.train = MagicMock()
    mock_epoch_runner.validate = MagicMock()
    mock_epoch_runner.test = MagicMock()

    core = CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
        epoch_runner_cls=lambda **_: mock_epoch_runner,
    )

    core.fit(max_epochs=3, test_model=False)

    assert mock_epoch_runner.train.call_count == 3
    assert mock_epoch_runner.validate.call_count == 3
    mock_epoch_runner.test.assert_not_called()


# ---------------------------------------------------------------------
# Callback manager integration
# ---------------------------------------------------------------------


def test_callbacks_are_called_in_correct_order(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
    callback_manager,
):
    mock_epoch_runner = MagicMock()

    core = CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
        callback_manager=callback_manager,
        epoch_runner_cls=lambda **_: mock_epoch_runner,
    )

    core.fit(start_epoch=0, max_epochs=2, test_model=False)

    expected_calls = [
        call("on_train_start", {"epoch": 0}),
        call("on_epoch_start", {"epoch": 0}),
        call("on_epoch_end", {"epoch": 0}),
        call("on_epoch_start", {"epoch": 1}),
        call("on_epoch_end", {"epoch": 1}),
        call("on_train_end", {"epoch": 1}),
    ]

    callback_manager.run.assert_has_calls(expected_calls, any_order=False)


# ---------------------------------------------------------------------
# Checkpoint manager integration
# ---------------------------------------------------------------------


def test_checkpoint_manager_called(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
    checkpoint_manager,
):
    mock_epoch_runner = MagicMock()

    core = CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
        checkpoint_manager=checkpoint_manager,
        epoch_runner_cls=lambda **_: mock_epoch_runner,
    )

    core.fit(max_epochs=2, final_checkpoint_name="final.pth")

    assert checkpoint_manager.evaluate_criteria.call_count == 2
    checkpoint_manager.save.assert_called_once_with(1, "final.pth")


# ---------------------------------------------------------------------
# Test phase behavior
# ---------------------------------------------------------------------


def test_test_phase_runs_when_enabled(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
):
    mock_epoch_runner = MagicMock()
    mock_epoch_runner.test = MagicMock()

    core = CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
        epoch_runner_cls=lambda **_: mock_epoch_runner,
    )

    core.fit(max_epochs=1, test_model=True)

    mock_epoch_runner.test.assert_called_once()


def test_test_phase_skipped_when_disabled(
    mock_descriptor,
    mock_constraints,
    network,
    criterion,
    optimizer,
    device,
    train_loader,
):
    mock_epoch_runner = MagicMock()
    mock_epoch_runner.test = MagicMock()

    core = CongradsCore(
        descriptor=mock_descriptor,
        constraints=mock_constraints,
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        dataloader_train=train_loader,
        epoch_runner_cls=lambda **_: mock_epoch_runner,
    )

    core.fit(max_epochs=1, test_model=False)

    mock_epoch_runner.test.assert_not_called()
