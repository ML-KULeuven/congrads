"""Defines the EpochRunner class for running full training, validation, and test epochs.

This module handles:
- Switching the network between training and evaluation modes
- Iterating over DataLoaders with optional progress bars
- Delegating per-batch processing to a BatchRunner instance
- Optional gradient tracking control for evaluation phases
- Warnings when validation or test loaders are not provided
"""

import warnings

from torch import set_grad_enabled
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.batch_runner import BatchRunner


class EpochRunner:
    """Runs full epochs over DataLoaders.

    Responsibilities:
    - Model mode switching
    - Iteration over DataLoader
    - Delegation to BatchRunner
    - Progress bars
    """

    def __init__(
        self,
        network: Module,
        batch_runner: BatchRunner,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        *,
        network_uses_grad: bool = False,
        disable_progress_bar: bool = False,
    ):
        """Initialize the EpochRunner.

        Args:
            network: The neural network module to train/validate/test.
            batch_runner: The BatchRunner instance for processing batches.
            train_loader: DataLoader for training data.
            valid_loader: DataLoader for validation data. Defaults to None.
            test_loader: DataLoader for test data. Defaults to None.
            network_uses_grad: Whether the network uses gradient computation. Defaults to False.
            disable_progress_bar: Whether to disable progress bar display. Defaults to False.
        """
        self.network = network
        self.batch_runner = batch_runner
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.network_uses_grad = network_uses_grad
        self.disable_progress_bar = disable_progress_bar

    def train(self) -> None:
        """Run a training epoch over the training DataLoader.

        Sets the network to training mode and iterates over batches,
        delegating each batch to the BatchRunner for processing.
        """
        self.network.train()

        for batch in tqdm(
            self.train_loader,
            desc="Training batches",
            leave=False,
            disable=self.disable_progress_bar,
        ):
            self.batch_runner.train_batch(batch)

    def validate(self) -> None:
        """Run a validation epoch over the validation DataLoader.

        Sets the network to evaluation mode and iterates over batches,
        delegating each batch to the BatchRunner for processing.
        Skips validation if no valid_loader is provided.
        """
        if self.valid_loader is None:
            warnings.warn("Validation skipped: no valid_loader provided.", stacklevel=2)
            return

        with set_grad_enabled(self.network_uses_grad):
            self.network.eval()

            for batch in tqdm(
                self.valid_loader,
                desc="Validation batches",
                leave=False,
                disable=self.disable_progress_bar,
            ):
                self.batch_runner.valid_batch(batch)

    def test(self) -> None:
        """Run a test epoch over the test DataLoader.

        Sets the network to evaluation mode and iterates over batches,
        delegating each batch to the BatchRunner for processing.
        Skips testing if no test_loader is provided.
        """
        if self.test_loader is None:
            warnings.warn("Testing skipped: no test_loader provided.", stacklevel=2)
            return

        with set_grad_enabled(self.network_uses_grad):
            self.network.eval()

            for batch in tqdm(
                self.test_loader,
                desc="Test batches",
                leave=False,
                disable=self.disable_progress_bar,
            ):
                self.batch_runner.test_batch(batch)
