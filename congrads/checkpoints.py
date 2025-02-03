"""
This module provides a `CheckpointManager` class for managing the saving and 
loading of checkpoints during PyTorch model training.

The `CheckpointManager` handles:

- Saving and loading the state of models, optimizers, and metrics.
- Registering and evaluating performance criteria to determine if a model's 
  performance has improved, enabling automated saving of the best-performing 
  model checkpoints.
- Resuming training from a specific checkpoint.

Usage:
    1. Initialize the `CheckpointManager` with a PyTorch model, optimizer, 
       and metric manager.
    2. Register criteria for tracking and evaluating metrics.
    3. Use the `save` and `load` methods to manage checkpoints during training.
    4. Call `evaluate_criteria` to automatically evaluate and save the 
       best-performing checkpoints.

Dependencies:
    - PyTorch (`torch`)
"""

import os
from pathlib import Path
from typing import Callable

from torch import Tensor, gt, load, save
from torch.nn import Module
from torch.optim import Optimizer

from .metrics import MetricManager
from .utils import validate_comparator_pytorch, validate_type


class CheckpointManager:
    """
    A class to handle saving and loading checkpoints for
    PyTorch models and optimizers.

    Args:
        network (torch.nn.Module): The network (model) to save/load.
        optimizer (torch.optim.Optimizer): The optimizer to save/load.
        metric_manager (MetricManager): The metric manager to restore saved
            metric states.
        save_dir (str): Directory where checkpoints will be saved. Defaults
            to 'checkpoints'.
        create_dir (bool): Whether to create the save_dir if it does not exist.
            Defaults to False.

    Raises:
        TypeError: If a provided attribute has an incompatible type.
        FileNotFoundError: If the save directory does not exist and create_dir
            is set to False.
    """

    def __init__(
        self,
        network: Module,
        optimizer: Optimizer,
        metric_manager: MetricManager,
        save_dir: str = "checkpoints",
        create_dir: bool = False,
    ):
        """
        Initialize the checkpoint manager.
        """

        # Type checking
        validate_type("network", network, Module)
        validate_type("optimizer", optimizer, Optimizer)
        validate_type("metric_manager", metric_manager, MetricManager)
        validate_type("create_dir", create_dir, bool)

        # Create path or raise error if create_dir is not found
        if not os.path.exists(save_dir):
            if not create_dir:
                raise FileNotFoundError(
                    f"Save directory '{str(save_dir)}' configured in "
                    "checkpoint manager is not found."
                )
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Initialize objects variables
        self.network = network
        self.optimizer = optimizer
        self.metric_manager = metric_manager
        self.save_dir = save_dir

        self.criteria: dict[str, Callable[[Tensor, Tensor], Tensor]] = {}
        self.best_metrics: dict[str, Tensor] = {}

    def register(
        self,
        metric_name: str,
        comparator: Callable[[Tensor, Tensor], Tensor] = gt,
    ):
        """
        Register a criterion for evaluating a performance metric
        during training.

        Stores the comparator to determine whether the current metric has
        improved relative to the previous best metric value.

        Args:
            metric_name (str): The name of the metric to evaluate.
            comparator (Callable[[Tensor, Tensor], Tensor], optional):
                A function that compares the current metric value against the
                previous best value. Defaults to a greater-than (gt) comparison.

        Raises:
            TypeError: If a provided attribute has an incompatible type.

        """

        validate_type("metric_name", metric_name, str)
        validate_comparator_pytorch("comparator", comparator)
        validate_comparator_pytorch("comparator", comparator)

        self.criteria[metric_name] = comparator

    def resume(
        self, filename: str = "checkpoint.pth", ignore_missing: bool = False
    ) -> int:
        """
        Resumes training from a saved checkpoint file.

        Args:
            filename (str): The name of the checkpoint file to load.
                Defaults to "checkpoint.pth".
            ignore_missing (bool): If True, does not raise an error if the
                checkpoint file is missing and continues without loading,
                starting from epoch 0. Defaults to False.

        Returns:
            int: The epoch number from the loaded checkpoint, or 0 if
                ignore_missing is True and no checkpoint was found.

        Raises:
            TypeError: If a provided attribute has an incompatible type.
            FileNotFoundError: If the specified checkpoint file does not exist.
        """

        # Type checking
        validate_type("filename", filename, str)
        validate_type("ignore_missing", ignore_missing, bool)

        # Return starting epoch, either from checkpoint file or default
        filepath = os.path.join(self.save_dir, filename)
        if os.path.exists(filepath):
            checkpoint = self.load("checkpoint.pth")
            return checkpoint["epoch"]
        elif ignore_missing:
            return 0
        else:
            raise FileNotFoundError(
                f"A checkpoint was not found at {filepath} to resume training."
            )

    def evaluate_criteria(self, epoch: int):
        """
        Evaluate the defined criteria for model performance metrics
        during training.

        Args:
            epoch (int): The current epoch number.

        Compares the current metrics against the previous best metrics using
        predefined comparators. If a criterion is met, saves the model and
        the corresponding best metric values.
        """

        for metric_name, comparator in self.criteria.items():

            current_metric_value = self.metric_manager.metrics[
                metric_name
            ].aggregate()
            best_metric_value = self.best_metrics.get(metric_name)

            # TODO improve efficiency by not checking is None each iteration
            if best_metric_value is None or comparator(
                current_metric_value,
                best_metric_value,
            ):
                self.save(epoch)
                self.best_metrics[metric_name] = current_metric_value

    def save(
        self,
        epoch: int,
        filename: str = "checkpoint.pth",
    ):
        """
        Save a checkpoint.

        Args:
            epoch (int): Current epoch number.
            filename (str): Name of the checkpoint file. Defaults to
                'checkpoint.pth'.
        """

        state = {
            "epoch": epoch,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_metrics": self.best_metrics,
        }
        filepath = os.path.join(self.save_dir, filename)
        save(state, filepath)

    def load(self, filename: str):
        """
        Load a checkpoint and restores the state of the network, optimizer
        and best_metrics.

        Args:
            filename (str): Name of the checkpoint file.

        Returns:
            dict: A dictionary containing the loaded checkpoint
            information (epoch, loss, etc.).
        """

        filepath = os.path.join(self.save_dir, filename)

        checkpoint = load(filepath)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_metrics = checkpoint["best_metrics"]

        return checkpoint
