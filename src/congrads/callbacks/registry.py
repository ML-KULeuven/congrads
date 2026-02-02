"""Holds all callback implementations for use in the training workflow.

This module acts as a central registry for defining and storing different
callback classes, such as logging, checkpointing, or custom behaviors
triggered during training, validation, or testing. It is intended to
collect all callback implementations in one place for easy reference
and import, and can be extended as new callbacks are added.
"""

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..callbacks.base import Callback
from ..metrics import MetricManager
from ..utils.utility import CSVLogger

__all__ = ["LoggerCallback"]


class LoggerCallback(Callback):
    """Callback to periodically aggregate and store metrics during training and testing.

    This callback works in conjunction with a MetricManager that accumulates metrics
    internally (e.g. per batch). Metrics are:

    - Aggregated at a configurable epoch interval (`aggregate_interval`)
    - Cached in memory (GPU-resident tensors)
    - Written to TensorBoard and CSV at a separate interval (`store_interval`)

    Aggregation and storage are decoupled to avoid unnecessary GPU-to-CPU
    synchronization. Any remaining cached metrics are flushed at the end of training.

    Methods implemented:
    - on_epoch_end: Periodically aggregates and stores training metrics.
    - on_train_end: Flushes any remaining cached training metrics.
    - on_test_end: Aggregates and stores test metrics immediately.
    """

    def __init__(
        self,
        metric_manager: MetricManager,
        tensorboard_logger: SummaryWriter,
        csv_logger: CSVLogger,
        *,
        aggregate_interval: int = 1,
        store_interval: int = 1,
    ):
        """Initialize the LoggerCallback.

        Args:
            metric_manager: Instance of MetricManager used to collect metrics.
            tensorboard_logger: TensorBoard SummaryWriter instance for logging scalars.
            csv_logger: CSVLogger instance for logging metrics to CSV files.
            aggregate_interval: Number of epochs between metric aggregation.
            store_interval: Number of epochs between metric storage.
        """
        super().__init__()

        # Input validation
        if aggregate_interval <= 0 or store_interval <= 0:
            raise ValueError("Intervals must be positive integers")

        if store_interval % aggregate_interval != 0:
            raise ValueError("store_interval must be a multiple of aggregate_interval")

        # Store references
        self.metric_manager = metric_manager
        self.tensorboard_logger = tensorboard_logger
        self.csv_logger = csv_logger
        self.aggregate_interval = aggregate_interval
        self.store_interval = store_interval

        # Cached metrics on GPU by epoch
        self._accumulated_metrics: dict[int, dict[str, Tensor]] = {}

    def on_epoch_end(self, data: dict[str, any], ctx: dict[str, any]):
        """Handle end-of-epoch training logic.

        At the end of each epoch, this method may:
        - Aggregate training metrics from the MetricManager (every `aggregate_interval` epochs)
        - Cache aggregated metrics keyed by epoch
        - Store cached metrics to disk (every `store_interval` epochs)

        Metric aggregation resets the MetricManager accumulation state.
        Metric storage triggers GPU-to-CPU synchronization and writes to loggers.

        Args:
            data: Dictionary containing epoch context (must include 'epoch').
            ctx: Additional context dictionary (unused).

        Returns:
            data: The same input dictionary, unmodified.
        """
        epoch = data["epoch"]

        # Cache training metrics
        if epoch % self.aggregate_interval == 0:
            metrics = self.metric_manager.aggregate("during_training")
            self._accumulated_metrics[epoch] = metrics
            self.metric_manager.reset("during_training")

        # Store metrics to disk
        if epoch % self.store_interval == 0:
            self._save(self._accumulated_metrics)
            self._accumulated_metrics.clear()

        return data

    def on_train_end(self, data, ctx):
        """Flush any remaining cached training metrics at the end of training.

        This ensures that aggregated metrics that were not yet written due to
        `store_interval` alignment are persisted before training terminates.

        Args:
            data: Dictionary containing training context (unused).
            ctx: Additional context dictionary (unused).

        Returns:
            data: The same input dictionary, unmodified.
        """
        if self._accumulated_metrics:
            self._save(self._accumulated_metrics)
            self._accumulated_metrics.clear()

        return data

    def on_test_end(self, data: dict[str, any], ctx: dict[str, any]):
        """Aggregate and store test metrics at the end of testing.

        Test metrics are aggregated once and written immediately to disk.
        Interval-based aggregation and caching are not applied to testing.

        Args:
            data: Dictionary containing test context (must include 'epoch').
            ctx: Additional context dictionary (unused).

        Returns:
            data: The same input dictionary, unmodified.
        """
        epoch = data["epoch"]

        # Save test metrics
        metrics = self.metric_manager.aggregate("after_training")
        self._save({epoch: metrics})
        self.metric_manager.reset("after_training")

        return data

    def _save(self, metrics: dict[int, dict[str, Tensor]]):
        """Write aggregated metrics to TensorBoard and CSV loggers.

        Args:
            metrics: Mapping from epoch to a dictionary of metric name to scalar tensor.
                Tensors are expected to be detached and graph-free.
        """
        for epoch, metrics_by_name in metrics.items():
            for name, value in metrics_by_name.items():
                cpu_value = value.item()
                self.tensorboard_logger.add_scalar(name, cpu_value, epoch)
                self.csv_logger.add_value(name, cpu_value, epoch)

        # Flush/save
        self.tensorboard_logger.flush()
        self.csv_logger.save()
