"""Holds all callback implementations for use in the training workflow.

This module acts as a central registry for defining and storing different
callback classes, such as logging, checkpointing, or custom behaviors
triggered during training, validation, or testing. It is intended to
collect all callback implementations in one place for easy reference
and import, and can be extended as new callbacks are added.
"""

from torch.utils.tensorboard import SummaryWriter

from ..callbacks.base import Callback
from ..metrics import MetricManager
from ..utils.utility import CSVLogger


class LoggerCallback(Callback):
    """Callback to log metrics to TensorBoard and CSV after each epoch or test.

    This callback queries a MetricManager for aggregated metrics, writes them
    to TensorBoard using SummaryWriter, and logs them to a CSV file via CSVLogger.
    It also flushes loggers and resets metrics after logging.

    Methods implemented:
    - on_epoch_end: Logs metrics at the end of a training epoch.
    - on_test_end: Logs metrics at the end of testing.
    """

    def __init__(
        self,
        metric_manager: MetricManager,
        tensorboard_logger: SummaryWriter,
        csv_logger: CSVLogger,
    ):
        """Initialize the LoggerCallback.

        Args:
            metric_manager: Instance of MetricManager used to collect metrics.
            tensorboard_logger: TensorBoard SummaryWriter instance for logging scalars.
            csv_logger: CSVLogger instance for logging metrics to CSV files.
        """
        super().__init__()
        self.metric_manager = metric_manager
        self.tensorboard_logger = tensorboard_logger
        self.csv_logger = csv_logger

    def on_epoch_end(self, data: dict[str, any], ctx: dict[str, any]):
        """Log training metrics at the end of an epoch.

        Aggregates metrics from the MetricManager under the 'during_training' category,
        writes them to TensorBoard and CSV, flushes the loggers, and resets the metrics
        for the next epoch.

        Args:
            data: Dictionary containing batch/epoch context (must include 'epoch').
            ctx: Additional context dictionary (unused in this implementation).

        Returns:
            data: The same input dictionary, unmodified.
        """
        epoch = data["epoch"]

        # Log training metrics
        metrics = self.metric_manager.aggregate("during_training")
        for name, value in metrics.items():
            self.tensorboard_logger.add_scalar(name, value.item(), epoch)
            self.csv_logger.add_value(name, value.item(), epoch)

        # Flush/save
        self.tensorboard_logger.flush()
        self.csv_logger.save()

        # Reset metric manager for training
        self.metric_manager.reset("during_training")

        return data

    def on_test_end(self, data: dict[str, any], ctx: dict[str, any]):
        """Log test metrics at the end of testing.

        Aggregates metrics from the MetricManager under the 'after_training' category,
        writes them to TensorBoard and CSV, flushes the loggers, and resets the metrics.

        Args:
            data: Dictionary containing test context (must include 'epoch').
            ctx: Additional context dictionary (unused in this implementation).

        Returns:
            data: The same input dictionary, unmodified.
        """
        epoch = data["epoch"]

        # Log test metrics
        metrics = self.metric_manager.aggregate("after_training")
        for name, value in metrics.items():
            self.tensorboard_logger.add_scalar(name, value.item(), epoch)
            self.csv_logger.add_value(name, value.item(), epoch)

        # Flush/save
        self.tensorboard_logger.flush()
        self.csv_logger.save()

        # Reset metric manager for test
        self.metric_manager.reset("after_training")

        return data
