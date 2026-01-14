"""Module for managing metrics during training.

Provides the `Metric` and `MetricManager` classes for accumulating,
aggregating, and resetting metrics over training batches. Supports
grouping metrics and using custom accumulation functions.
"""

from collections.abc import Callable

from torch import Tensor, cat, nanmean, tensor

from .utils.validation import validate_callable, validate_type


class Metric:
    """Represents a single metric to be accumulated and aggregated.

    Stores metric values over multiple batches and computes an aggregated
    result using a specified accumulation function.
    """

    def __init__(self, name: str, accumulator: Callable[..., Tensor] = nanmean) -> None:
        """Initialize a Metric instance.

        Args:
            name (str): Name of the metric.
            accumulator (Callable[..., Tensor], optional): Function to aggregate
                accumulated values. Defaults to `torch.nanmean`.
        """
        # Type checking
        validate_type("name", name, str)
        validate_callable("accumulator", accumulator)

        self.name = name
        self.accumulator = accumulator
        self.values: list[Tensor] = []
        self.sample_count = 0

    def accumulate(self, value: Tensor) -> None:
        """Accumulate a new value for the metric.

        Args:
            value (Tensor): Metric values for the current batch.
        """
        self.values.append(value.detach().clone())
        self.sample_count += value.size(0)

    def aggregate(self) -> Tensor:
        """Compute the aggregated value of the metric.

        Returns:
            Tensor: The aggregated metric value. Returns NaN if no values
                have been accumulated.
        """
        if not self.values:
            return tensor(float("nan"))

        combined = cat(self.values)
        return self.accumulator(combined)

    def reset(self) -> None:
        """Reset the accumulated values and sample count for the metric."""
        self.values = []
        self.sample_count = 0


class MetricManager:
    """Manages multiple metrics and groups for training or evaluation.

    Supports registering metrics, accumulating values by name, aggregating
    metrics by group, and resetting metrics by group.
    """

    def __init__(self) -> None:
        """Initialize a MetricManager instance."""
        self.metrics: dict[str, Metric] = {}
        self.groups: dict[str, str] = {}

    def register(
        self, name: str, group: str = "default", accumulator: Callable[..., Tensor] = nanmean
    ) -> None:
        """Register a new metric under a specified group.

        Args:
            name (str): Name of the metric.
            group (str, optional): Group name for the metric. Defaults to "default".
            accumulator (Callable[..., Tensor], optional): Function to aggregate
                accumulated values. Defaults to `torch.nanmean`.
        """
        # Type checking
        validate_type("name", name, str)
        validate_type("group", group, str)
        validate_callable("accumulator", accumulator)

        self.metrics[name] = Metric(name, accumulator)
        self.groups[name] = group

    def accumulate(self, name: str, value: Tensor) -> None:
        """Accumulate a value for a specific metric by name.

        Args:
            name (str): Name of the metric.
            value (Tensor): Metric values for the current batch.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' is not registered.")

        self.metrics[name].accumulate(value)

    def aggregate(self, group: str = "default") -> dict[str, Tensor]:
        """Aggregate all metrics in a specified group.

        Args:
            group (str, optional): The group of metrics to aggregate. Defaults to "default".

        Returns:
            dict[str, Tensor]: Dictionary mapping metric names to their
                aggregated values.
        """
        return {
            name: metric.aggregate()
            for name, metric in self.metrics.items()
            if self.groups[name] == group
        }

    def reset(self, group: str = "default") -> None:
        """Reset all metrics in a specified group.

        Args:
            group (str, optional): The group of metrics to reset. Defaults to "default".
        """
        for name, metric in self.metrics.items():
            if self.groups[name] == group:
                metric.reset()

    def reset_all(self) -> None:
        """Reset all metrics across all groups."""
        for metric in self.metrics.values():
            metric.reset()
