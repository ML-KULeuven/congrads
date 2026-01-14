"""Module for managing PyTorch model checkpoints.

Provides the `CheckpointManager` class to save and load model and optimizer
states during training, track the best metric values, and optionally report
checkpoint events.
"""

import os
from collections.abc import Callable
from pathlib import Path

from torch import Tensor, load, save
from torch.nn import Module
from torch.optim import Optimizer

from .metrics import MetricManager
from .utils.validation import validate_callable, validate_type


class CheckpointManager:
    """Manage saving and loading checkpoints for PyTorch models and optimizers.

    Handles checkpointing based on a criteria function, restores metric
    states, and optionally reports when a checkpoint is saved.
    """

    def __init__(
        self,
        criteria_function: Callable[[dict[str, Tensor], dict[str, Tensor]], bool],
        network: Module,
        optimizer: Optimizer,
        metric_manager: MetricManager,
        save_dir: str = "checkpoints",
        create_dir: bool = False,
        report_save: bool = False,
    ):
        """Initialize the CheckpointManager.

        Args:
            criteria_function (Callable[[dict[str, Tensor], dict[str, Tensor]], bool]):
                Function that determines if the current checkpoint should be
                saved based on the current and best metric values.
            network (torch.nn.Module): The model to save/load.
            optimizer (torch.optim.Optimizer): The optimizer to save/load.
            metric_manager (MetricManager): Manages metric states for checkpointing.
            save_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
            create_dir (bool, optional): Whether to create `save_dir` if it does not exist.
                Defaults to False.
            report_save (bool, optional): Whether to report when a checkpoint is saved.
                Defaults to False.

        Raises:
            TypeError: If any provided attribute has an incompatible type.
            FileNotFoundError: If `save_dir` does not exist and `create_dir` is False.
        """
        # Type checking
        validate_callable("criteria_function", criteria_function)
        validate_type("network", network, Module)
        validate_type("optimizer", optimizer, Optimizer)
        validate_type("metric_manager", metric_manager, MetricManager)
        validate_type("create_dir", create_dir, bool)
        validate_type("report_save", report_save, bool)

        # Create path or raise error if create_dir is not found
        if not os.path.exists(save_dir):
            if not create_dir:
                raise FileNotFoundError(
                    f"Save directory '{save_dir}' configured in checkpoint manager is not found."
                )
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Initialize objects variables
        self.criteria_function = criteria_function
        self.network = network
        self.optimizer = optimizer
        self.metric_manager = metric_manager
        self.save_dir = save_dir
        self.report_save = report_save

        self.best_metric_values: dict[str, Tensor] = {}

    def evaluate_criteria(self, epoch: int, metric_group: str = "during_training"):
        """Evaluate the criteria function to determine if a better model is found.

        Aggregates the current metric values during training and applies the
        criteria function. If the criteria function indicates improvement, the
        best metric values are updated, a checkpoint is saved, and a message is
        optionally printed.

        Args:
            epoch (int): The current epoch number.
            metric_group (str, optional): The metric group to evaluate. Defaults to 'during_training'.
        """
        current_metric_values = self.metric_manager.aggregate(metric_group)
        if self.criteria_function is not None and self.criteria_function(
            current_metric_values, self.best_metric_values
        ):
            # Print message if a new checkpoint is saved
            if self.report_save:
                print(f"New checkpoint saved at epoch {epoch}.")

            # Update current best metric values
            for metric_name, metric_value in current_metric_values.items():
                self.best_metric_values[metric_name] = metric_value

            # Save the current state
            self.save(epoch)

    def resume(self, filename: str = "checkpoint.pth", ignore_missing: bool = False) -> int:
        """Resumes training from a saved checkpoint file.

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
            checkpoint = self.load(filename)
            return checkpoint["epoch"]
        elif ignore_missing:
            return 0
        else:
            raise FileNotFoundError(f"A checkpoint was not found at {filepath} to resume training.")

    def save(self, epoch: int, filename: str = "checkpoint.pth"):
        """Save a checkpoint.

        Args:
            epoch (int): Current epoch number.
            filename (str): Name of the checkpoint file. Defaults to
                'checkpoint.pth'.
        """
        state = {
            "epoch": epoch,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_metrics": self.best_metric_values,
        }
        filepath = os.path.join(self.save_dir, filename)
        save(state, filepath)

    def load(self, filename: str):
        """Load a checkpoint and restore the training state.

        Loads the checkpoint from the specified file and restores the network
        weights, optimizer state, and best metric values.

        Args:
            filename (str): Name of the checkpoint file.

        Returns:
            dict: A dictionary containing the loaded checkpoint information,
                including epoch, loss, and other relevant training state.
        """
        filepath = os.path.join(self.save_dir, filename)

        checkpoint = load(filepath, weights_only=True)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_metric_values = checkpoint["best_metrics"]

        return checkpoint
