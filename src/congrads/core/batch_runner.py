"""Defines the BatchRunner, which executes individual batches for training, validation, and testing.

Responsibilities:
- Move batch data to the appropriate device
- Run forward passes through the network
- Compute base and constraint-adjusted losses
- Perform backpropagation during training
- Accumulate metrics for loss and other monitored quantities
- Trigger callbacks at key points in the batch lifecycle
"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from ..callbacks.base import CallbackManager
from ..core.constraint_engine import ConstraintEngine
from ..metrics import MetricManager


class BatchRunner:
    """Executes a single batch for training, validation, or testing.

    The BatchRunner handles moving data to the correct device, running the network
    forward, computing base and constraint-adjusted losses, performing backpropagation
    during training, accumulating metrics, and dispatching callbacks at key points
    in the batch lifecycle.
    """

    def __init__(
        self,
        network: Module,
        criterion,
        optimizer: Optimizer,
        constraint_engine: ConstraintEngine,
        metric_manager: MetricManager | None,
        callback_manager: CallbackManager | None,
        device: torch.device,
    ):
        """Initialize the BatchRunner.

        Args:
            network: The neural network module to execute.
            criterion: Loss function callable accepting (output, target, data=batch).
            optimizer: Optimizer for updating network parameters.
            constraint_engine: ConstraintEngine instance for evaluating and enforcing constraints.
            metric_manager: Optional MetricManager for logging batch metrics.
            callback_manager: Optional CallbackManager for triggering hooks during batch processing.
            device: Torch device on which to place data and network.
        """
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.constraint_engine = constraint_engine
        self.metric_manager = metric_manager
        self.callback_manager = callback_manager
        self.device = device

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Move all tensors in the batch to the BatchRunner's device.

        Args:
            batch: Dictionary of tensors for a single batch.

        Returns:
            Dictionary of tensors moved to the target device.
        """
        return {k: v.to(self.device) for k, v in batch.items()}

    def _run_callbacks(self, hook: str, data: dict) -> dict:
        """Run the specified callback hook on the batch data.

        Args:
            hook: Name of the callback hook to run.
            data: Dictionary containing batch data.

        Returns:
            Potentially modified batch data after callback execution.
        """
        if self.callback_manager is None:
            return data
        return self.callback_manager.run(hook, data)

    def train_batch(self, batch: dict[str, Tensor]) -> Tensor:
        """Run a single training batch.

        Steps performed:
        1. Move batch to device and run "on_train_batch_start" callback.
        2. Forward pass through the network.
        3. Compute base loss using the criterion and accumulate metric.
        4. Apply constraint-based adjustments to the loss.
        5. Perform backward pass and optimizer step.
        6. Run "on_train_batch_end" callback.

        Args:
            batch: Dictionary of input and target tensors for the batch.

        Returns:
            Tensor: The base loss computed before constraint adjustments.
        """
        batch = self._to_device(batch)
        batch = self._run_callbacks("on_train_batch_start", batch)

        # Forward
        batch = self.network(batch)
        batch = self._run_callbacks("after_train_forward", batch)

        # Base loss
        loss: Tensor = self.criterion(
            batch["output"],
            batch["target"],
            data=batch,
        )

        if self.metric_manager is not None:
            self.metric_manager.accumulate("Loss/train", loss.unsqueeze(0))

        # Constraint-adjusted loss
        combined_loss = self.constraint_engine.train(batch, loss)

        # Backward
        self.optimizer.zero_grad()
        combined_loss.backward()
        self.optimizer.step()

        batch = self._run_callbacks("on_train_batch_end", batch)
        return loss

    def valid_batch(self, batch: dict[str, Tensor]) -> Tensor:
        """Run a single validation batch.

        Steps performed:
        1. Move batch to device and run "on_valid_batch_start" callback.
        2. Forward pass through the network.
        3. Compute base loss using the criterion and accumulate metric.
        4. Evaluate constraints via the ConstraintEngine (does not modify loss).
        5. Run "on_valid_batch_end" callback.

        Args:
            batch: Dictionary of input and target tensors for the batch.

        Returns:
            Tensor: The base loss computed for the batch.
        """
        batch = self._to_device(batch)
        batch = self._run_callbacks("on_valid_batch_start", batch)

        batch = self.network(batch)
        batch = self._run_callbacks("after_valid_forward", batch)

        loss: Tensor = self.criterion(
            batch["output"],
            batch["target"],
            data=batch,
        )

        if self.metric_manager is not None:
            self.metric_manager.accumulate("Loss/valid", loss.unsqueeze(0))

        self.constraint_engine.validate(batch, loss)

        batch = self._run_callbacks("on_valid_batch_end", batch)
        return loss

    def test_batch(self, batch: dict[str, Tensor]) -> Tensor:
        """Run a single test batch.

        Steps performed:
        1. Move batch to device and run "on_test_batch_start" callback.
        2. Forward pass through the network.
        3. Compute base loss using the criterion and accumulate metric.
        4. Evaluate constraints via the ConstraintEngine (does not modify loss).
        5. Run "on_test_batch_end" callback.

        Args:
            batch: Dictionary of input and target tensors for the batch.

        Returns:
            Tensor: The base loss computed for the batch.
        """
        batch = self._to_device(batch)
        batch = self._run_callbacks("on_test_batch_start", batch)

        batch = self.network(batch)
        batch = self._run_callbacks("after_test_forward", batch)

        loss: Tensor = self.criterion(
            batch["output"],
            batch["target"],
            data=batch,
        )

        if self.metric_manager is not None:
            self.metric_manager.accumulate("Loss/test", loss.unsqueeze(0))

        self.constraint_engine.test(batch, loss)

        batch = self._run_callbacks("on_test_batch_end", batch)
        return loss
