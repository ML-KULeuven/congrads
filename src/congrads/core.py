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

The `CongradsCore` class allows for the use of additional callback functions
at different stages of the training process to customize behavior for
specific needs. These include callbacks for the start and end of epochs, as
well as the start and end of the entire training process.

"""

import warnings
from collections.abc import Callable

import torch
from torch import Tensor, float32, no_grad, sum, tensor
from torch.linalg import vector_norm
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
    is_torch_loss,
    torch_loss_wrapper,
    validate_callable,
    validate_callable_iterable,
    validate_iterable,
    validate_loaders,
    validate_type,
)


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
        loaders: tuple[DataLoader, DataLoader, DataLoader],
        network: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        metric_manager: MetricManager,
        device: torch.device,
        network_uses_grad: bool = False,
        checkpoint_manager: CheckpointManager = None,
        epsilon: float = 1e-6,
        constraint_aggregator: Callable[..., Tensor] = sum,
        disable_progress_bar_epoch: bool = False,
        disable_progress_bar_batch: bool = False,
        enforce_all: bool = True,
    ):
        """Initialize the CongradsCore object.

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
            network_uses_grad (bool, optional): A flag indicating if the network
                contains gradient calculation computations. Default is False.
            checkpoint_manager (CheckpointManager, optional): Manages
                    checkpointing. If not set, no checkpointing is done.
            epsilon (float, optional): A small value to avoid division by zero
                in gradient calculations. Default is 1e-10.
            constraint_aggregator (Callable[..., Tensor], optional): A function
                to aggregate the constraint rescale loss. Default is `sum`.
            disable_progress_bar_epoch (bool, optional): If set to True, the epoch
                progress bar will not show. Defaults to False.
            disable_progress_bar_batch (bool, optional): If set to True, the batch
                progress bar will not show. Defaults to False.
            enforce_all (bool, optional): If set to False, constraints will only be monitored and
                not influence the training process. Overrides constraint-specific `enforce` parameters.
                Defaults to True.

        Note:
            A warning is logged if the descriptor has no variable layers,
            as at least one variable layer is required for the constraint logic
            to influence the training process.
        """
        # Type checking
        validate_type("descriptor", descriptor, Descriptor)
        validate_iterable("constraints", constraints, Constraint, allow_empty=True)
        validate_loaders("loaders", loaders)
        validate_type("network", network, Module)
        validate_type("criterion", criterion, _Loss)
        validate_type("optimizer", optimizer, Optimizer)
        validate_type("metric_manager", metric_manager, MetricManager)
        validate_type("device", device, torch.device)
        validate_type("network_uses_grad", network_uses_grad, bool)
        validate_type(
            "checkpoint_manager",
            checkpoint_manager,
            CheckpointManager,
            allow_none=True,
        )
        validate_type("epsilon", epsilon, float)
        validate_callable("constraint_aggregator", constraint_aggregator, allow_none=True)
        validate_type("disable_progress_bar_epoch", disable_progress_bar_epoch, bool)
        validate_type("disable_progress_bar_batch", disable_progress_bar_batch, bool)
        validate_type("enforce_all", enforce_all, bool)

        # Init object variables
        self.descriptor = descriptor
        self.constraints = constraints
        self.train_loader = loaders[0]
        self.valid_loader = loaders[1]
        self.test_loader = loaders[2]
        self.network = network
        self.optimizer = optimizer
        self.metric_manager = metric_manager
        self.device = device
        self.network_uses_grad = network_uses_grad
        self.checkpoint_manager = checkpoint_manager
        self.epsilon = epsilon
        self.constraint_aggregator = constraint_aggregator
        self.disable_progress_bar_epoch = disable_progress_bar_epoch
        self.disable_progress_bar_batch = disable_progress_bar_batch
        self.enforce_all = enforce_all

        # Check if criterion is a torch loss function
        if is_torch_loss(criterion):
            # If so, wrap it in a custom loss function
            self.criterion = torch_loss_wrapper(criterion)
        else:
            self.criterion = criterion

        # Perform checks
        if len(self.descriptor.variable_keys) == 0:
            warnings.warn(
                "The descriptor object has no variable layers. The constraint \
                    guided loss adjustment is therefore not used. \
                    Is this the intended behavior?",
                stacklevel=2,
            )

        # Initialize constraint metrics
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
        start_epoch: int = 0,
        max_epochs: int = 100,
        test_model: bool = True,
        on_batch_start: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_batch_end: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_train_batch_start: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_train_batch_end: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_valid_batch_start: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_valid_batch_end: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_test_batch_start: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_test_batch_end: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]] | None = None,
        on_epoch_start: list[Callable[[int], None]] | None = None,
        on_epoch_end: list[Callable[[int], None]] | None = None,
        on_train_start: list[Callable[[int], None]] | None = None,
        on_train_end: list[Callable[[int], None]] | None = None,
        on_train_completion_forward_pass: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]]
        | None = None,
        on_val_completion_forward_pass: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]]
        | None = None,
        on_test_completion_forward_pass: list[Callable[[dict[str, Tensor]], dict[str, Tensor]]]
        | None = None,
        on_test_start: list[Callable[[int], None]] | None = None,
        on_test_end: list[Callable[[int], None]] | None = None,
    ) -> None:
        """Train the model over multiple epochs with optional validation and testing.

        This method manages the full training loop, including:

        - Executing epoch-level and batch-level callbacks.
        - Training and validating the model each epoch.
        - Adjusting losses according to constraints.
        - Logging metrics via the metric manager.
        - Optional evaluation on the test set.
        - Checkpointing the model during and after training.

        Args:
            start_epoch (int, optional): Epoch number to start training from. Defaults to 0.
            max_epochs (int, optional): Total number of epochs to train. Defaults to 100.
            test_model (bool, optional): If True, evaluate the model on the test set after training. Defaults to True.
            on_batch_start (list[Callable], optional): Callbacks executed at the start of every batch. Defaults to None.
            on_batch_end (list[Callable], optional): Callbacks executed at the end of every batch. Defaults to None.
            on_train_batch_start (list[Callable], optional): Callbacks executed at the start of each training batch. Defaults to `on_batch_start` if not provided.
            on_train_batch_end (list[Callable], optional): Callbacks executed at the end of each training batch. Defaults to `on_batch_end` if not provided.
            on_valid_batch_start (list[Callable], optional): Callbacks executed at the start of each validation batch. Defaults to `on_batch_start` if not provided.
            on_valid_batch_end (list[Callable], optional): Callbacks executed at the end of each validation batch. Defaults to `on_batch_end` if not provided.
            on_test_batch_start (list[Callable], optional): Callbacks executed at the start of each test batch. Defaults to `on_batch_start` if not provided.
            on_test_batch_end (list[Callable], optional): Callbacks executed at the end of each test batch. Defaults to `on_batch_end` if not provided.
            on_epoch_start (list[Callable], optional): Callbacks executed at the start of each epoch. Defaults to None.
            on_epoch_end (list[Callable], optional): Callbacks executed at the end of each epoch. Defaults to None.
            on_train_start (list[Callable], optional): Callbacks executed before training starts. Defaults to None.
            on_train_end (list[Callable], optional): Callbacks executed after training ends. Defaults to None.
            on_train_completion_forward_pass (list[Callable], optional): Callbacks executed after the forward pass during training. Defaults to None.
            on_val_completion_forward_pass (list[Callable], optional): Callbacks executed after the forward pass during validation. Defaults to None.
            on_test_completion_forward_pass (list[Callable], optional): Callbacks executed after the forward pass during testing. Defaults to None.
            on_test_start (list[Callable], optional): Callbacks executed before testing starts. Defaults to None.
            on_test_end (list[Callable], optional): Callbacks executed after testing ends. Defaults to None.

        Notes:
            - If phase-specific callbacks (train/valid/test) are not provided, the global `on_batch_start` and `on_batch_end` are used.
            - Training metrics, loss adjustments, and constraint satisfaction ratios are automatically logged via the metric manager.
            - The final model checkpoint is saved if a checkpoint manager is configured.
        """
        # Type checking
        validate_type("start_epoch", start_epoch, int)
        validate_type("max_epochs", max_epochs, int)
        validate_type("test_model", test_model, bool)
        validate_callable_iterable("on_batch_start", on_batch_start, allow_none=True)
        validate_callable_iterable("on_batch_end", on_batch_end, allow_none=True)
        validate_callable_iterable("on_train_batch_start", on_train_batch_start, allow_none=True)
        validate_callable_iterable("on_train_batch_end", on_train_batch_end, allow_none=True)
        validate_callable_iterable("on_valid_batch_start", on_valid_batch_start, allow_none=True)
        validate_callable_iterable("on_valid_batch_end", on_valid_batch_end, allow_none=True)
        validate_callable_iterable("on_test_batch_start", on_test_batch_start, allow_none=True)
        validate_callable_iterable("on_test_batch_end", on_test_batch_end, allow_none=True)
        validate_callable_iterable("on_epoch_start", on_epoch_start, allow_none=True)
        validate_callable_iterable("on_epoch_end", on_epoch_end, allow_none=True)
        validate_callable_iterable("on_train_start", on_train_start, allow_none=True)
        validate_callable_iterable("on_train_end", on_train_end, allow_none=True)
        validate_callable_iterable(
            "on_train_completion_forward_pass",
            on_train_completion_forward_pass,
            allow_none=True,
        )
        validate_callable_iterable(
            "on_val_completion_forward_pass",
            on_val_completion_forward_pass,
            allow_none=True,
        )
        validate_callable_iterable(
            "on_test_completion_forward_pass",
            on_test_completion_forward_pass,
            allow_none=True,
        )
        validate_callable_iterable("on_test_start", on_test_start, allow_none=True)
        validate_callable_iterable("on_test_end", on_test_end, allow_none=True)

        # Use global batch callback if phase-specific callback is unset
        # Init callbacks as empty list if None
        on_train_batch_start = on_train_batch_start or on_batch_start or []
        on_train_batch_end = on_train_batch_end or on_batch_end or []
        on_valid_batch_start = on_valid_batch_start or on_batch_start or []
        on_valid_batch_end = on_valid_batch_end or on_batch_end or []
        on_test_batch_start = on_test_batch_start or on_batch_start or []
        on_test_batch_end = on_test_batch_end or on_batch_end or []
        on_batch_start = on_batch_start or []
        on_batch_end = on_batch_end or []
        on_epoch_start = on_epoch_start or []
        on_epoch_end = on_epoch_end or []
        on_train_start = on_train_start or []
        on_train_end = on_train_end or []
        on_train_completion_forward_pass = on_train_completion_forward_pass or []
        on_val_completion_forward_pass = on_val_completion_forward_pass or []
        on_test_completion_forward_pass = on_test_completion_forward_pass or []
        on_test_start = on_test_start or []
        on_test_end = on_test_end or []

        # Keep track of epoch
        epoch = start_epoch

        # Execute training start hook if set
        for callback in on_train_start:
            callback(epoch)

        for i in tqdm(
            range(epoch, max_epochs),
            initial=epoch,
            desc="Epoch",
            disable=self.disable_progress_bar_epoch,
        ):
            epoch = i

            # Execute epoch start hook if set
            for callback in on_epoch_start:
                callback(epoch)

            # Execute training and validation epoch
            self._train_epoch(
                on_train_batch_start,
                on_train_batch_end,
                on_train_completion_forward_pass,
            )
            self._validate_epoch(
                on_valid_batch_start,
                on_valid_batch_end,
                on_val_completion_forward_pass,
            )

            # Checkpointing
            if self.checkpoint_manager:
                self.checkpoint_manager.evaluate_criteria(epoch)

            # Execute epoch end hook if set
            for callback in on_epoch_end:
                callback(epoch)

        # Execute training end hook if set
        for callback in on_train_end:
            callback(epoch)

        # Evaluate model performance on unseen test set if required
        if test_model:
            # Execute test end hook if set
            for callback in on_test_start:
                callback(epoch)

            self._test_model(
                on_test_batch_start,
                on_test_batch_end,
                on_test_completion_forward_pass,
            )

            # Execute test end hook if set
            for callback in on_test_end:
                callback(epoch)

        # Save final model
        if self.checkpoint_manager:
            self.checkpoint_manager.save(epoch, "checkpoint_final.pth")

    def _train_epoch(
        self,
        on_train_batch_start: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]], ...],
        on_train_batch_end: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]], ...],
        on_train_completion_forward_pass: tuple[
            Callable[[dict[str, Tensor]], dict[str, Tensor]], ...
        ],
    ) -> None:
        """Perform a single training epoch over all batches.

        This method sets the network to training mode, iterates over the training
        DataLoader, computes predictions, evaluates losses, applies constraint-based
        adjustments, performs backpropagation, and updates model parameters. It also
        supports executing optional callbacks at different stages of the batch
        processing.

        Args:
            on_train_batch_start (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]], ...]):
                Callbacks executed at the start of each batch. Each callback receives the
                data dictionary and returns updated versions.
            on_train_batch_end (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]], ...]):
                Callbacks executed at the end of each batch. Each callback receives the
                data dictionary and returns updated versions.
            on_train_completion_forward_pass (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]], ...]):
                Callbacks executed immediately after the forward pass of the batch.
                Each callback receives the data dictionary and returns updated versions.

        Returns:
            None
        """
        # Set model in training mode
        self.network.train()

        for data in tqdm(
            self.train_loader,
            desc="Training batches",
            leave=False,
            disable=self.disable_progress_bar_batch,
        ):
            # Transfer batch data to GPU
            data: dict[str, Tensor] = {key: value.to(self.device) for key, value in data.items()}

            # Execute on batch start callbacks
            for callback in on_train_batch_start:
                data = callback(data)

            # Model computations
            data = self.network(data)

            # Execute on completion forward pass callbacks
            for callback in on_train_completion_forward_pass:
                data = callback(data)

            # Calculate loss
            loss = self.criterion(
                data["output"],
                data["target"],
                data=data,
            )
            self.metric_manager.accumulate("Loss/train", loss.unsqueeze(0))

            # Adjust loss based on constraints
            combined_loss = self.train_step(
                data,
                loss,
                self.constraints,
                self.descriptor,
                self.metric_manager,
                self.device,
                constraint_aggregator=self.constraint_aggregator,
                epsilon=self.epsilon,
                enforce_all=self.enforce_all,
            )

            # Backprop
            self.optimizer.zero_grad()
            combined_loss.backward(retain_graph=False, inputs=list(self.network.parameters()))
            self.optimizer.step()

            # Execute on batch end callbacks
            for callback in on_train_batch_end:
                data = callback(data)

    def _validate_epoch(
        self,
        on_valid_batch_start: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]],
        on_valid_batch_end: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]],
        on_valid_completion_forward_pass: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]],
    ) -> None:
        """Perform a single validation epoch over all batches.

        This method sets the network to evaluation mode, iterates over the validation
        DataLoader, computes predictions, evaluates losses, and logs constraint
        satisfaction. Optional callbacks can be executed at the start and end of each
        batch, as well as after the forward pass.

        Args:
            on_valid_batch_start (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]]):
                Callbacks executed at the start of each validation batch. Each callback
                receives the data dictionary and returns updated versions.
            on_valid_batch_end (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]]):
                Callbacks executed at the end of each validation batch. Each callback
                receives the data dictionary and returns updated versions.
            on_valid_completion_forward_pass (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]]):
                Callbacks executed immediately after the forward pass of the validation batch.
                Each callback receives the data dictionary and returns updated versions.

        Returns:
            None
        """
        # Set model in evaluation mode
        self.network.eval()

        # Enable or disable gradient tracking for validation pass
        with torch.set_grad_enabled(self.network_uses_grad):
            # Loop over validation batches
            for data in tqdm(
                self.valid_loader,
                desc="Validation batches",
                leave=False,
                disable=self.disable_progress_bar_batch,
            ):
                # Transfer batch data to GPU
                data: dict[str, Tensor] = {
                    key: value.to(self.device) for key, value in data.items()
                }

                # Execute on batch start callbacks
                for callback in on_valid_batch_start:
                    data = callback(data)

                # Model computations
                data = self.network(data)

                # Execute on completion forward pass callbacks
                for callback in on_valid_completion_forward_pass:
                    data = callback(data)

                # Calculate loss
                loss = self.criterion(
                    data["output"],
                    data["target"],
                    data=data,
                )
                self.metric_manager.accumulate("Loss/valid", loss.unsqueeze(0))

                # Validate constraints
                self.valid_step(
                    data,
                    loss,
                    self.constraints,
                    self.metric_manager,
                )

                # Execute on batch end callbacks
                for callback in on_valid_batch_end:
                    data = callback(data)

    def _test_model(
        self,
        on_test_batch_start: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]],
        on_test_batch_end: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]],
        on_test_completion_forward_pass: tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]],
    ) -> None:
        """Evaluate the model on the test dataset.

        This method sets the network to evaluation mode, iterates over the test
        DataLoader, computes predictions, evaluates losses, and logs constraint
        satisfaction. Optional callbacks can be executed at the start and end of
        each batch, as well as after the forward pass.

        Args:
            on_test_batch_start (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]]):
                Callbacks executed at the start of each test batch. Each callback
                receives the data dictionary and returns updated versions.
            on_test_batch_end (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]]):
                Callbacks executed at the end of each test batch. Each callback
                receives the data dictionary and returns updated versions.
            on_test_completion_forward_pass (tuple[Callable[[dict[str, Tensor]], dict[str, Tensor]]]):
                Callbacks executed immediately after the forward pass of the test batch.
                Each callback receives the data dictionary and returns updated versions.

        Returns:
            None
        """
        # Set model in evaluation mode
        self.network.eval()

        # Enable or disable gradient tracking for validation pass
        with torch.set_grad_enabled(self.network_uses_grad):
            # Loop over test batches
            for data in tqdm(
                self.test_loader,
                desc="Test batches",
                leave=False,
                disable=self.disable_progress_bar_batch,
            ):
                # Transfer batch data to GPU
                data: dict[str, Tensor] = {
                    key: value.to(self.device) for key, value in data.items()
                }

                # Execute on batch start callbacks
                for callback in on_test_batch_start:
                    data = callback(data)

                # Model computations
                data = self.network(data)

                # Execute on completion forward pass callbacks
                for callback in on_test_completion_forward_pass:
                    data = callback(data)

                # Calculate loss
                loss = self.criterion(
                    data["output"],
                    data["target"],
                    data=data,
                )
                self.metric_manager.accumulate("Loss/test", loss.unsqueeze(0))

                # Validate constraints
                self.test_step(
                    data,
                    loss,
                    self.constraints,
                    self.metric_manager,
                )

                # Execute on batch end callbacks
                for callback in on_test_batch_end:
                    data = callback(data)

    @staticmethod
    def train_step(
        data: dict[str, Tensor],
        loss: Tensor,
        constraints: list[Constraint],
        descriptor: Descriptor,
        metric_manager: MetricManager,
        device: torch.device,
        constraint_aggregator: Callable = torch.sum,
        epsilon: float = 1e-6,
        enforce_all: bool = True,
    ) -> Tensor:
        """Adjust the training loss based on constraints and compute the combined loss.

        This method calculates the directions in which the network outputs should be
        adjusted to satisfy constraints, scales these adjustments according to the
        constraint's rescale factor and gradient norms, and adds the result to the
        base loss. It also logs the constraint satisfaction ratio (CSR) for monitoring.

        Args:
            data (dict[str, Tensor]): Dictionary containing the batch data, predictions and additional data.
            loss (Tensor): The base loss computed by the criterion.
            constraints (list[Constraint]): List of constraints to enforce during training.
            descriptor (Descriptor): Descriptor containing layer metadata and variable/loss layer info.
            metric_manager (MetricManager): Metric manager for logging loss and CSR.
            device (torch.device): Device on which computations are performed.
            constraint_aggregator (Callable, optional): Function to aggregate per-layer rescaled losses. Defaults to `torch.mean`.
            epsilon (float, optional): Small value to prevent division by zero in gradient normalization. Defaults to 1e-6.
            enforce_all (bool, optional): If False, constraints are only monitored and do not influence the loss. Defaults to True.

        Returns:
            Tensor: The combined loss including the original loss and constraint-based adjustments.
        """
        # Init scalar tensor for loss
        total_rescale_loss = tensor(0, dtype=float32, device=device)
        norm_loss_grad: dict[str, Tensor] = {}

        # Precalculate loss gradients for each variable layer
        for key in descriptor.variable_keys & descriptor.affects_loss_keys:
            # Calculate gradients of loss w.r.t. predictions
            grad = torch.autograd.grad(
                outputs=loss, inputs=data[key], retain_graph=True, allow_unused=True
            )[0]

            # If gradients is None, report error
            if grad is None:
                raise RuntimeError(
                    f"Unable to compute loss gradients for layer '{key}'. "
                    "For layers not connected to the loss, set has_loss=False "
                    "when defining them in the Descriptor."
                )

            # Flatten batch and compute L2 norm along each item
            grad_flat = grad.view(grad.shape[0], -1)
            norm_loss_grad[key] = (
                vector_norm(grad_flat, dim=1, ord=2, keepdim=True).clamp(min=epsilon).detach()
            )

        for constraint in constraints:
            # Check if constraints are satisfied and calculate directions
            checks, mask = constraint.check_constraint(data)
            directions = constraint.calculate_direction(data)

            # Log constraint satisfaction ratio
            csr = (sum(checks * mask) / sum(mask)).unsqueeze(0)
            metric_manager.accumulate(f"{constraint.name}/train", csr)
            metric_manager.accumulate("CSR/train", csr)

            # Only do adjusting calculation if constraint is not observant
            if not enforce_all or not constraint.enforce:
                continue

            # Only do direction calculations for variable layers affecting constraint
            for key in constraint.layers & descriptor.variable_keys:
                with no_grad():
                    # Multiply direction modifiers with constraint result
                    constraint_result = (1 - checks) * directions[key]

                    # Multiply result with rescale factor of constraint
                    constraint_result *= constraint.rescale_factor

                # Calculate rescale loss
                total_rescale_loss += constraint_aggregator(
                    data[key] * constraint_result * norm_loss_grad[key],
                )

        # Return combined loss
        return loss + total_rescale_loss

    @staticmethod
    def valid_step(
        data: dict[str, Tensor],
        loss: Tensor,
        constraints: list[Constraint],
        metric_manager: MetricManager,
    ) -> Tensor:
        """Evaluate constraints during validation and log constraint satisfaction metrics.

        This method checks whether each constraint is satisfied for the given
        data, computes the constraint satisfaction ratio (CSR),
        and logs it using the metric manager. The base loss is not modified.

        Args:
            data (dict[str, Tensor]): Dictionary containing the batch data, predictions and additional data.
            loss (Tensor): The base loss computed by the criterion.
            constraints (list[Constraint]): List of constraints to evaluate.
            metric_manager (MetricManager): Metric manager for logging CSR and per-constraint metrics.

        Returns:
            Tensor: The original, unchanged base loss.
        """
        # For each constraint in this reference space, calculate directions
        for constraint in constraints:
            # Check if constraints are satisfied for
            checks, mask = constraint.check_constraint(data)

            # Log constraint satisfaction ratio
            csr = (sum(checks * mask) / sum(mask)).unsqueeze(0)
            metric_manager.accumulate(f"{constraint.name}/valid", csr)
            metric_manager.accumulate("CSR/valid", csr)

        # Return original loss
        return loss

    @staticmethod
    def test_step(
        data: dict[str, Tensor],
        loss: Tensor,
        constraints: list[Constraint],
        metric_manager: MetricManager,
    ) -> Tensor:
        """Evaluate constraints during testing and log constraint satisfaction metrics.

        This method checks whether each constraint is satisfied for the given
        data, computes the constraint satisfaction ratio (CSR),
        and logs it using the metric manager. The base loss is not modified.

        Args:
            data (dict[str, Tensor]): Dictionary containing the batch data, predictions and additional data.
            loss (Tensor): The base loss computed by the criterion.
            constraints (list[Constraint]): List of constraints to evaluate.
            metric_manager (MetricManager): Metric manager for logging CSR and per-constraint metrics.

        Returns:
            Tensor: The original, unchanged base loss.
        """
        # For each constraint in this reference space, calculate directions
        for constraint in constraints:
            # Check if constraints are satisfied for
            checks, mask = constraint.check_constraint(data)

            # Log constraint satisfaction ratio
            csr = (sum(checks * mask) / sum(mask)).unsqueeze(0)
            metric_manager.accumulate(f"{constraint.name}/test", csr)
            metric_manager.accumulate("CSR/test", csr)

        # Return original loss
        return loss
