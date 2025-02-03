"""
This module provides the CongradsCore class, which is designed to integrate 
constraint-guided optimization into neural network training. 
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

Modules in this package provide the following:

- `Descriptor`: Describes variable layers in the network that are 
  subject to constraints.
- `Constraint`: Defines various constraints, which are used to guide 
  the training process.
- `MetricManager`: Manages and tracks performance metrics such as loss 
  and constraint satisfaction.
- `CheckpointManager`: Manages saving and loading model checkpoints 
  during training.
- Utility functions to validate inputs and configurations.

Dependencies:
    - PyTorch (`torch`)
    - tqdm (for progress tracking)

The `CongradsCore` class allows for the use of additional callback functions 
at different stages of the training process to customize behavior for 
specific needs. These include callbacks for the start and end of epochs, as 
well as the start and end of the entire training process.

"""

import warnings
from numbers import Number
from typing import Callable

import torch

# pylint: disable-next=redefined-builtin
from torch import Tensor, float32, maximum, no_grad, norm, numel, sum, tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoints import CheckpointManager
from .constraints import Constraint
from .descriptor import Descriptor
from .metrics import MetricManager
from .utils import (
    validate_callable,
    validate_iterable,
    validate_loaders,
    validate_type,
)


class CongradsCore:
    """
    The CongradsCore class is the central training engine for constraint-guided
    neural network optimization. It integrates standard neural network training
    with additional constraint-driven adjustments to the loss function, ensuring
    that the network satisfies domain-specific constraints during training.

    Args:
        descriptor (Descriptor): Describes variable layers in the network.
        constraints (list[Constraint]): List of constraints to guide training.
        loaders (tuple[DataLoader, DataLoader, DataLoader]): DataLoaders for
            training, validation, and testing.
        network (Module): The neural network model to train.
        criterion (callable): The loss function used for
            training and validation.
        optimizer (Optimizer): The optimizer used for updating model parameters.
        metric_manager (MetricManager): Manages metric tracking and recording.
        device (torch.device): The device (e.g., CPU or GPU) for computations.
        checkpoint_manager (CheckpointManager, optional): Manages
                checkpointing. If not set, no checkpointing is done.
        epsilon (Number, optional): A small value to avoid division by zero
            in gradient calculations. Default is 1e-10.

    Note:
        A warning is logged if the descriptor has no variable layers,
        as at least one variable layer is required for the constraint logic
        to influence the training process.
    """

    def __init__(
        self,
        descriptor: Descriptor,
        constraints: list[Constraint],
        loaders: tuple[DataLoader, DataLoader, DataLoader],
        network: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        metric_manager: MetricManager,
        device: torch.device,
        checkpoint_manager: CheckpointManager = None,
        epsilon: Number = 1e-6,
    ):
        """
        Initialize the CongradsCore object.
        """

        # Type checking
        validate_type("descriptor", descriptor, Descriptor)
        validate_iterable("constraints", constraints, Constraint)
        validate_loaders()
        validate_type("network", network, Module)
        validate_type("criterion", criterion, _Loss)
        validate_type("optimizer", optimizer, Optimizer)
        validate_type("metric_manager", metric_manager, MetricManager)
        validate_type("device", device, torch.device)
        validate_type(
            "checkpoint_manager",
            checkpoint_manager,
            CheckpointManager,
            allow_none=True,
        )
        validate_type("epsilon", epsilon, Number)

        # Init object variables
        self.descriptor = descriptor
        self.constraints = constraints
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        self.test_loader = loaders[2]
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_manager = metric_manager
        self.device = device
        self.checkpoint_manager = checkpoint_manager

        # Init epsilon tensor
        self.epsilon = tensor(epsilon, device=self.device)

        # Perform checks
        if len(self.descriptor.variable_layers) == 0:
            warnings.warn(
                "The descriptor object has no variable layers. The constraint \
                    guided loss adjustment is therefore not used. \
                    Is this the intended behavior?"
            )

        # Initialize constraint metrics
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """
        Register metrics for loss, constraint satisfaction ratio (CSR),
        and individual constraints.

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
            self.metric_manager.register(
                f"{constraint.name}/train", "during_training"
            )
            self.metric_manager.register(
                f"{constraint.name}/valid", "during_training"
            )
            self.metric_manager.register(
                f"{constraint.name}/test", "after_training"
            )

    def fit(
        self,
        start_epoch: int = 0,
        max_epochs: int = 100,
        on_epoch_start: Callable[[int], None] = None,
        on_epoch_end: Callable[[int], None] = None,
        on_train_start: Callable[[int], None] = None,
        on_train_end: Callable[[int], None] = None,
    ) -> None:
        """
        Train the model for a given number of epochs.

        Args:
            start_epoch (int, optional): The epoch number to start the training
                with. Default is 0.
            max_epochs (int, optional): The number of epochs to train the
                model. Default is 100.
            on_epoch_start (Callable[[int], None], optional): A callback
                function that will be executed at the start of each epoch.
            on_epoch_end (Callable[[int], None], optional): A callback
                function that will be executed at the end of each epoch.
            on_train_start (Callable[[int], None], optional): A callback
                function that will be executed before the training starts.
            on_train_end (Callable[[int], None], optional): A callback
                function that will be executed after training ends.
        """

        # Type checking
        validate_type("start_epoch", start_epoch, int)
        validate_callable("on_epoch_start", on_epoch_start, True)
        validate_callable("on_epoch_end", on_epoch_end, True)
        validate_callable("on_train_start", on_train_start, True)
        validate_callable("on_train_end", on_train_end, True)

        # Keep track of epoch
        epoch = start_epoch

        # Execute training start hook if set
        if on_train_start:
            on_train_start(epoch)

        for i in tqdm(range(epoch, max_epochs), initial=epoch, desc="Epoch"):
            epoch = i

            # Execute epoch start hook if set
            if on_epoch_start:
                on_epoch_start(epoch)

            # Execute training and validation epoch
            self._train_epoch()
            self._validate_epoch()

            # Checkpointing
            if self.checkpoint_manager:
                self.checkpoint_manager.evaluate_criteria(epoch)

            # Execute epoch end hook if set
            if on_epoch_end:
                on_epoch_end(epoch)

        # Evaluate model performance on unseen test set
        self._test_model()

        # Save final model
        if self.checkpoint_manager:
            self.checkpoint_manager.save(epoch, "checkpoint_final.pth")

        # Execute training end hook if set
        if on_train_end:
            on_train_end(epoch)

    def _train_epoch(self) -> None:
        """
        Perform training for a single epoch.

        This method:
            - Sets the model to training mode.
            - Processes batches from the training DataLoader.
            - Computes predictions and losses.
            - Adjusts losses based on constraints.
            - Updates model parameters using backpropagation.

        Args:
            epoch (int): The current epoch number.
        """

        # Set model in training mode
        self.network.train()

        for batch in tqdm(
            self.train_loader, desc="Training batches", leave=False
        ):

            # Get input-output pairs from batch
            inputs, outputs = batch

            # Transfer to GPU
            inputs, outputs = inputs.to(self.device), outputs.to(self.device)

            # Model computations
            prediction = self.network(inputs)

            # Calculate loss
            loss = self.criterion(prediction["output"], outputs)
            self.metric_manager.accumulate("Loss/train", loss.unsqueeze(0))

            # Adjust loss based on constraints
            combined_loss = self.train_step(prediction, loss)

            # Backprop
            self.optimizer.zero_grad()
            combined_loss.backward(
                retain_graph=False, inputs=list(self.network.parameters())
            )
            self.optimizer.step()

    def _validate_epoch(self) -> None:
        """
        Perform validation for a single epoch.

        This method:
            - Sets the model to evaluation mode.
            - Processes batches from the validation DataLoader.
            - Computes predictions and losses.
            - Logs constraint satisfaction ratios.

        Args:
            epoch (int): The current epoch number.
        """

        # Set model in evaluation mode
        self.network.eval()

        with no_grad():
            for batch in tqdm(
                self.valid_loader, desc="Validation batches", leave=False
            ):

                # Get input-output pairs from batch
                inputs, outputs = batch

                # Transfer to GPU
                inputs, outputs = inputs.to(self.device), outputs.to(
                    self.device
                )

                # Model computations
                prediction = self.network(inputs)

                # Calculate loss
                loss = self.criterion(prediction["output"], outputs)
                self.metric_manager.accumulate("Loss/valid", loss.unsqueeze(0))

                # Validate constraints
                self.valid_step(prediction, loss)

    def _test_model(self) -> None:
        """
        Evaluate model performance on the test set.

        This method:
            - Sets the model to evaluation mode.
            - Processes batches from the test DataLoader.
            - Computes predictions and losses.
            - Logs constraint satisfaction ratios.

        """

        # Set model in evaluation mode
        self.network.eval()

        with no_grad():
            for batch in tqdm(
                self.test_loader, desc="Test batches", leave=False
            ):

                # Get input-output pairs from batch
                inputs, outputs = batch

                # Transfer to GPU
                inputs, outputs = inputs.to(self.device), outputs.to(
                    self.device
                )

                # Model computations
                prediction = self.network(inputs)

                # Calculate loss
                loss = self.criterion(prediction["output"], outputs)
                self.metric_manager.accumulate("Loss/test", loss.unsqueeze(0))

                # Validate constraints
                self.test_step(prediction, loss)

    def train_step(
        self,
        prediction: dict[str, Tensor],
        loss: Tensor,
    ) -> Tensor:
        """
        Adjust the training loss based on constraints
        and compute the combined loss.

        Args:
            prediction (dict[str, Tensor]): Model predictions
                for variable layers.
            loss (Tensor): The base loss computed by the criterion.

        Returns:
            Tensor: The combined loss (base loss + constraint adjustments).
        """

        # Init scalar tensor for loss
        total_rescale_loss = tensor(0, dtype=float32, device=self.device)
        loss_grads = {}

        # Precalculate loss gradients for each variable layer
        with no_grad():
            for layer in self.descriptor.variable_layers:
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True, inputs=prediction[layer])
                loss_grads[layer] = prediction[layer].grad

        for constraint in self.constraints:

            # Check if constraints are satisfied and calculate directions
            with no_grad():
                constraint_checks, relevant_constraint_count = (
                    constraint.check_constraint(prediction)
                )

            # Only do adjusting calculation if constraint is not observant
            if not constraint.monitor_only:
                with no_grad():
                    constraint_directions = constraint.calculate_direction(
                        prediction
                    )

                # Only do direction calculations for variable
                # layers affecting constraint
                for layer in (
                    constraint.layers & self.descriptor.variable_layers
                ):

                    with no_grad():
                        # Multiply direction modifiers with constraint result
                        constraint_result = (
                            1 - constraint_checks.unsqueeze(1)
                        ) * constraint_directions[layer]

                        # Multiply result with rescale factor of constraint
                        constraint_result *= constraint.rescale_factor

                        # Calculate loss gradient norm
                        norm_loss_grad = norm(
                            loss_grads[layer], dim=1, p=2, keepdim=True
                        )

                        # Apply minimum epsilon
                        norm_loss_grad = maximum(norm_loss_grad, self.epsilon)

                    # Calculate rescale loss
                    rescale_loss = (
                        prediction[layer]
                        * constraint_result
                        * norm_loss_grad.detach().clone()
                    ).mean()

                    # Store rescale loss for this reference space
                    total_rescale_loss += rescale_loss

            # Log constraint satisfaction ratio
            self.metric_manager.accumulate(
                f"{constraint.name}/train",
                (
                    (
                        sum(constraint_checks)
                        - numel(constraint_checks)
                        + relevant_constraint_count
                    )
                    / relevant_constraint_count
                ).unsqueeze(0),
            )
            self.metric_manager.accumulate(
                "CSR/train",
                (
                    (
                        sum(constraint_checks)
                        - numel(constraint_checks)
                        + relevant_constraint_count
                    )
                    / relevant_constraint_count
                ).unsqueeze(0),
            )

        # Return combined loss
        return loss + total_rescale_loss

    def valid_step(
        self,
        prediction: dict[str, Tensor],
        loss: Tensor,
    ) -> Tensor:
        """
        Evaluate constraints during validation and log satisfaction metrics.

        Args:
            prediction (dict[str, Tensor]): Model predictions for
                variable layers.
            loss (Tensor): The base loss computed by the criterion.

        Returns:
            Tensor: The unchanged base loss.
        """

        # For each constraint in this reference space, calculate directions
        for constraint in self.constraints:

            # Check if constraints are satisfied for
            constraint_checks, relevant_constraint_count = (
                constraint.check_constraint(prediction)
            )

            # Log constraint satisfaction ratio
            self.metric_manager.accumulate(
                f"{constraint.name}/valid",
                (
                    (
                        sum(constraint_checks)
                        - numel(constraint_checks)
                        + relevant_constraint_count
                    )
                    / relevant_constraint_count
                ).unsqueeze(0),
            )
            self.metric_manager.accumulate(
                "CSR/valid",
                (
                    (
                        sum(constraint_checks)
                        - numel(constraint_checks)
                        + relevant_constraint_count
                    )
                    / relevant_constraint_count
                ).unsqueeze(0),
            )

        # Return loss
        return loss

    def test_step(
        self,
        prediction: dict[str, Tensor],
        loss: Tensor,
    ) -> Tensor:
        """
        Evaluate constraints during test and log satisfaction metrics.

        Args:
            prediction (dict[str, Tensor]): Model predictions
                for variable layers.
            loss (Tensor): The base loss computed by the criterion.

        Returns:
            Tensor: The unchanged base loss.
        """

        # For each constraint in this reference space, calculate directions
        for constraint in self.constraints:

            # Check if constraints are satisfied for
            constraint_checks, relevant_constraint_count = (
                constraint.check_constraint(prediction)
            )

            # Log constraint satisfaction ratio
            self.metric_manager.accumulate(
                f"{constraint.name}/test",
                (
                    (
                        sum(constraint_checks)
                        - numel(constraint_checks)
                        + relevant_constraint_count
                    )
                    / relevant_constraint_count
                ).unsqueeze(0),
            )
            self.metric_manager.accumulate(
                "CSR/test",
                (
                    (
                        sum(constraint_checks)
                        - numel(constraint_checks)
                        + relevant_constraint_count
                    )
                    / relevant_constraint_count
                ).unsqueeze(0),
            )

        # Return loss
        return loss
