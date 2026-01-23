from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from congrads.core.epoch_runner import EpochRunner

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def dummy_dataset():
    x = torch.randn(10, 3)
    y = torch.randn(10, 1)
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
def mock_network():
    net = MagicMock()
    net.train = MagicMock()
    net.eval = MagicMock()
    return net


@pytest.fixture
def mock_batch_runner():
    br = MagicMock()
    br.train_batch = MagicMock()
    br.valid_batch = MagicMock()
    br.test_batch = MagicMock()
    return br


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def test_train_runs_all_batches(mock_network, mock_batch_runner, train_loader):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
        disable_progress_bar=True,
    )

    runner.train()

    # network switched to train mode
    mock_network.train.assert_called_once()
    mock_network.eval.assert_not_called()

    # train_batch called once per batch
    assert mock_batch_runner.train_batch.call_count == len(train_loader)
    mock_batch_runner.valid_batch.assert_not_called()
    mock_batch_runner.test_batch.assert_not_called()


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_validate_runs_all_batches(mock_network, mock_batch_runner, train_loader, valid_loader):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
        valid_loader=valid_loader,
        disable_progress_bar=True,
    )

    runner.validate()

    # network switched to eval mode
    mock_network.eval.assert_called_once()
    mock_network.train.assert_not_called()

    # valid_batch called once per batch
    assert mock_batch_runner.valid_batch.call_count == len(valid_loader)
    mock_batch_runner.train_batch.assert_not_called()
    mock_batch_runner.test_batch.assert_not_called()


def test_validate_without_loader_warns(mock_network, mock_batch_runner, train_loader):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
    )

    with pytest.warns(UserWarning, match="Validation skipped"):
        runner.validate()

    mock_network.eval.assert_not_called()
    mock_batch_runner.valid_batch.assert_not_called()


def test_validate_respects_gradient_flag(
    mock_network, mock_batch_runner, train_loader, valid_loader
):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
        valid_loader=valid_loader,
        network_uses_grad=False,
        disable_progress_bar=True,
    )

    # Ensure gradients are disabled inside validation
    torch.set_grad_enabled(True)
    runner.validate()
    assert torch.is_grad_enabled() is True


# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------


def test_test_runs_all_batches(mock_network, mock_batch_runner, train_loader, test_loader):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
        test_loader=test_loader,
        disable_progress_bar=True,
    )

    runner.test()

    mock_network.eval.assert_called_once()
    assert mock_batch_runner.test_batch.call_count == len(test_loader)

    mock_batch_runner.train_batch.assert_not_called()
    mock_batch_runner.valid_batch.assert_not_called()


def test_test_without_loader_warns(mock_network, mock_batch_runner, train_loader):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
    )

    with pytest.warns(UserWarning, match="Testing skipped"):
        runner.test()

    mock_network.eval.assert_not_called()
    mock_batch_runner.test_batch.assert_not_called()


def test_test_respects_gradient_flag(mock_network, mock_batch_runner, train_loader, test_loader):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
        test_loader=test_loader,
        network_uses_grad=False,
        disable_progress_bar=True,
    )

    torch.set_grad_enabled(True)
    runner.test()
    assert torch.is_grad_enabled() is True


# ---------------------------------------------------------------------
# Combined behavior
# ---------------------------------------------------------------------


def test_train_validate_test_sequence(
    mock_network, mock_batch_runner, train_loader, valid_loader, test_loader
):
    runner = EpochRunner(
        network=mock_network,
        batch_runner=mock_batch_runner,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        disable_progress_bar=True,
    )

    runner.train()
    runner.validate()
    runner.test()

    assert mock_batch_runner.train_batch.call_count == len(train_loader)
    assert mock_batch_runner.valid_batch.call_count == len(valid_loader)
    assert mock_batch_runner.test_batch.call_count == len(test_loader)

    # eval should be called for validate + test
    assert mock_network.eval.call_count == 2
    mock_network.train.assert_called_once()
