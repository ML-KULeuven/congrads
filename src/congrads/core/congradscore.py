"""This module provides the core CongradsCore class for the main training functionality.

It is designed to integrate constraint-guided optimization into neural network training.
It extends traditional training processes by enforcing specific constraints
on the model's outputs, ensuring that the network satisfies domain-specific
requirements during both training and evaluation.

The `CongradsCore` class serves as the central engine for managing the
training, validation, and testing phases of a neural network model,
incorporating constraints that influence the loss function and model updates.
The model is trained with standard loss functions while also incorporating
constraint-based adjustments, which are tracked and logged
throughout the process.

Key features:
- Support for various constraints that can influence the training process.
- Integration with PyTorch's `DataLoader` for efficient batch processing.
- Metric management for tracking loss and constraint satisfaction.
- Checkpoint management for saving and evaluating model states.

"""

from collections.abc import Callable

import torch
from torch import Tensor, sum
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from congrads.utils.utility import LossWrapper

from ..callbacks.base import CallbackManager
from ..checkpoints import CheckpointManager
from ..constraints.base import Constraint
from ..core.batch_runner import BatchRunner
from ..core.constraint_engine import ConstraintEngine
from ..core.epoch_runner import EpochRunner
from ..descriptor import Descriptor
from ..metrics import MetricManager


class CongradsCore:
    """The CongradsCore class is the central training engine for constraint-guided optimization.

    It integrates standard neural network training
    with additional constraint-driven adjustments to the loss function, ensuring
    that the network satisfies domain-specific constraints during training.
    """

    def __init__(
        self,
        descriptor: Descriptor,
        constraints: list[Constraint],
        network: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        device: torch.device,
        dataloader_train: DataLoader,
        dataloader_valid: DataLoader | None = None,
        dataloader_test: DataLoader | None = None,
        metric_manager: MetricManager | None = None,
        callback_manager: CallbackManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        network_uses_grad: bool = False,
        epsilon: float = 1e-6,
        constraint_aggregator: Callable[..., Tensor] = sum,
        enforce_all: bool = True,
        disable_progress_bar_epoch: bool = False,
        disable_progress_bar_batch: bool = False,
        epoch_runner_cls: type["EpochRunner"] | None = None,
        batch_runner_cls: type["BatchRunner"] | None = None,
        constraint_engine_cls: type["ConstraintEngine"] | None = None,
    ):
        """Initialize the CongradsCore object.

        Args:
            descriptor (Descriptor): Describes variable layers in the network.
            constraints (list[Constraint]): List of constraints to guide training.
            network (Module): The neural network model to train.
            criterion (callable): The loss function used for
                training and validation.
            optimizer (Optimizer): The optimizer used for updating model parameters.
            device (torch.device): The device (e.g., CPU or GPU) for computations.
            dataloader_train (DataLoader): DataLoader for training data.
            dataloader_valid (DataLoader, optional): DataLoader for validation data.
                If not provided, validation is skipped.
            dataloader_test (DataLoader, optional): DataLoader for test data.
                If not provided, testing is skipped.
            metric_manager (MetricManager, optional): Manages metric tracking and recording.
            callback_manager (CallbackManager, optional): Manages training callbacks.
            checkpoint_manager (CheckpointManager, optional): Manages
                    checkpointing. If not set, no checkpointing is done.
            network_uses_grad (bool, optional): A flag indicating if the network
                contains gradient calculation computations. Default is False.
            epsilon (float, optional): A small value to avoid division by zero
                in gradient calculations. Default is 1e-10.
            constraint_aggregator (Callable[..., Tensor], optional): A function
                to aggregate the constraint rescale loss. Default is `sum`.
            enforce_all (bool, optional): If set to False, constraints will only be monitored and
                not influence the training process. Overrides constraint-specific `enforce` parameters.
                Defaults to True.
            disable_progress_bar_epoch (bool, optional): If set to True, the epoch
                progress bar will not show. Defaults to False.
            disable_progress_bar_batch (bool, optional): If set to True, the batch
                progress bar will not show. Defaults to False.
            epoch_runner_cls (type[EpochRunner], optional): Custom EpochRunner class.
                If not provided, the default EpochRunner is used.
            batch_runner_cls (type[BatchRunner], optional): Custom BatchRunner class.
                If not provided, the default BatchRunner is used.
            constraint_engine_cls (type[ConstraintEngine], optional): Custom ConstraintEngine class.
                If not provided, the default ConstraintEngine is used.

        Note:
            A warning is logged if the descriptor has no variable layers,
            as at least one variable layer is required for the constraint logic
            to influence the training process.
        """
        # Init object variables
        self.device = device
        self.network = network.to(device)
        self.criterion = LossWrapper(criterion)
        self.optimizer = optimizer
        self.descriptor = descriptor

        self.constraints = constraints or []
        self.epsilon = epsilon
        self.aggregator = constraint_aggregator
        self.enforce_all = enforce_all

        self.metric_manager = metric_manager
        self.callback_manager = callback_manager
        self.checkpoint_manager = checkpoint_manager

        self.disable_progress_bar_epoch = disable_progress_bar_epoch
        self.disable_progress_bar_batch = disable_progress_bar_batch

        # Initialize constraint engine
        self.constraint_engine = (constraint_engine_cls or ConstraintEngine)(
            constraints=self.constraints,
            descriptor=self.descriptor,
            device=self.device,
            epsilon=self.epsilon,
            aggregator=self.aggregator,
            enforce_all=self.enforce_all,
            metric_manager=self.metric_manager,
        )

        # Initialize runners
        self.batch_runner = (batch_runner_cls or BatchRunner)(
            network=self.network,
            criterion=self.criterion,
            optimizer=self.optimizer,
            constraint_engine=self.constraint_engine,
            metric_manager=self.metric_manager,
            callback_manager=self.callback_manager,
            device=self.device,
        )

        self.epoch_runner = (epoch_runner_cls or EpochRunner)(
            batch_runner=self.batch_runner,
            network=self.network,
            train_loader=dataloader_train,
            valid_loader=dataloader_valid,
            test_loader=dataloader_test,
            network_uses_grad=network_uses_grad,
            disable_progress_bar=self.disable_progress_bar_batch,
        )

        # Initialize constraint metrics
        if self.metric_manager is not None:
            self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Register metrics for loss, constraint satisfaction ratio (CSR), and constraints.

        This method registers the following metrics:

        - Loss/train: Training loss.
        - Loss/valid: Validation loss.
        - Loss/test: Test loss after training.
        - CSR/train: Constraint satisfaction ratio during training.
        - CSR/valid: Constraint satisfaction ratio during validation.
        - CSR/test: Constraint satisfaction ratio after training.
        - One metric per constraint, for both training and validation.

        """
        self.metric_manager.register("Loss/train", "during_training")
        self.metric_manager.register("Loss/valid", "during_training")
        self.metric_manager.register("Loss/test", "after_training")

        if len(self.constraints) > 0:
            self.metric_manager.register("CSR/train", "during_training")
            self.metric_manager.register("CSR/valid", "during_training")
            self.metric_manager.register("CSR/test", "after_training")

        for constraint in self.constraints:
            self.metric_manager.register(f"{constraint.name}/train", "during_training")
            self.metric_manager.register(f"{constraint.name}/valid", "during_training")
            self.metric_manager.register(f"{constraint.name}/test", "after_training")

    def fit(
        self,
        *,
        start_epoch: int = 0,
        max_epochs: int = 100,
        test_model: bool = True,
        final_checkpoint_name: str = "checkpoint_final.pth",
    ) -> None:
        """Run the full training loop, including optional validation, testing, and checkpointing.

        This method performs training over multiple epochs with the following steps:
        1. Trigger "on_train_start" callbacks if a callback manager is present.
        2. For each epoch:
        - Trigger "on_epoch_start" callbacks.
        - Run a training epoch via the EpochRunner.
        - Run a validation epoch via the EpochRunner.
        - Evaluate checkpoint criteria if a checkpoint manager is present.
        - Trigger "on_epoch_end" callbacks.
        3. Trigger "on_train_end" callbacks after all epochs.
        4. Optionally run a test epoch via the EpochRunner if `test_model` is True,
        with corresponding "on_test_start" and "on_test_end" callbacks.
        5. Save a final checkpoint using the checkpoint manager.

        Args:
            start_epoch: Index of the starting epoch (default 0). Useful for resuming training.
            max_epochs: Maximum number of epochs to run (default 100).
            test_model: Whether to run a test epoch after training (default True).
            final_checkpoint_name: Filename for the final checkpoint saved at the end of training
                                (default "checkpoint_final.pth").

        Returns:
            None
        """
        if self.callback_manager:
            self.callback_manager.run("on_train_start", {"epoch": start_epoch})

        for epoch in tqdm(
            range(start_epoch, max_epochs),
            initial=start_epoch,
            desc="Epoch",
            disable=self.disable_progress_bar_epoch,
        ):
            if self.callback_manager:
                self.callback_manager.run("on_epoch_start", {"epoch": epoch})

            self.epoch_runner.train()
            self.epoch_runner.validate()

            if self.checkpoint_manager:
                self.checkpoint_manager.evaluate_criteria(epoch)

            if self.callback_manager:
                self.callback_manager.run("on_epoch_end", {"epoch": epoch})

        if self.callback_manager:
            self.callback_manager.run("on_train_end", {"epoch": epoch})

        if test_model:
            if self.callback_manager:
                self.callback_manager.run("on_test_start", {"epoch": epoch})

            self.epoch_runner.test()

            if self.callback_manager:
                self.callback_manager.run("on_test_end", {"epoch": epoch})

        if self.checkpoint_manager:
            self.checkpoint_manager.save(epoch, final_checkpoint_name)
