"""
This module defines the `Metric` and `MetricManager` classes, which are 
used to track and aggregate performance metrics during model training or 
evaluation in machine learning. These classes support the accumulation of 
metric values, aggregation using customizable functions (such as mean), 
and resetting of the metrics.

Classes:

    - Metric: A class that tracks and aggregates a specific metric over 
      multiple samples, allowing for accumulation, aggregation, and 
      resetting of values.
    - MetricManager: A class that manages and tracks multiple metrics 
      during model training or evaluation, supporting registration, 
      accumulation, aggregation, and resetting of metrics.

Key Methods:

    - `Metric.__init__`: Initializes a metric with a specified name and 
      optional accumulator function (defaults to `nanmean`).
    - `Metric.accumulate`: Accumulates a new value for the metric, 
      typically a tensor of model output or performance.
    - `Metric.aggregate`: Aggregates the accumulated values using the 
      specified accumulator function.
    - `Metric.reset`: Resets the accumulated values and sample count for 
      the metric.
    - `MetricManager.__init__`: Initializes a manager for multiple metrics.
    - `MetricManager.register`: Registers a new metric with a name, group, 
      and optional accumulator function.
    - `MetricManager.accumulate`: Accumulates a new value for the specified 
      metric.
    - `MetricManager.aggregate`: Aggregates all metrics in a specified group.
    - `MetricManager.reset`: Resets all registered metrics in a specified 
      group.

Each class provides functionality to efficiently track, aggregate, and reset 
metrics during the training and evaluation phases of machine learning tasks, 
supporting flexible aggregation strategies and group-based management of 
metrics.
"""

from typing import Callable

from torch import Tensor, cat, nanmean

from .utils import validate_callable, validate_type


class Metric:
    """
    A class that tracks and aggregates a specific metric over multiple samples.

    This class allows the accumulation of values, their aggregation using a
    specified function (e.g., mean), and the ability to reset the metrics.
    It is typically used to track performance metrics during training or
    evaluation processes in machine learning.

    Args:
        name (str): The name of the metric.
        accumulator (Callable[..., Tensor], optional): A function used to
        aggregate values (defaults to `nanmean`).

    Attributes:
        name (str): The name of the metric.
        accumulator (Callable[..., Tensor]): The function used to aggregate
            values.
        values (list): A list to store accumulated values.
        sample_count (int): The count of accumulated samples.

    """

    def __init__(
        self,
        name: str,
        accumulator: Callable[..., Tensor] = nanmean,
    ) -> None:
        """
        Constructor method
        """

        # Type checking
        validate_type("name", name, str)
        validate_callable("accumulator", accumulator)

        self.name = name
        self.accumulator = accumulator

        self.values = []
        self.sample_count = 0

    def accumulate(self, value: Tensor) -> None:
        """
        Accumulates a new value for the metric.

        Args:
            value (Tensor): The new value to accumulate, typically a
                tensor of model output or performance.
        """

        self.values.append(value)
        self.sample_count += value.size(0)

    def aggregate(self) -> Tensor:
        """
        Aggregates the accumulated values using the specified
        accumulator function.

        Returns:
            Tensor: The aggregated result of the accumulated values.
        """

        combined = cat(self.values)
        return self.accumulator(combined)

    def reset(self) -> None:
        """
        Resets the accumulated values and sample count for the metric.
        """

        self.values = []
        self.sample_count = 0


class MetricManager:
    """
    A class to manage and track multiple metrics during model
    training or evaluation.

    This class allows registering metrics, accumulating values for each metric,
    and recording the aggregated values. It also supports the reset of metrics
    after each epoch or training step.

    Attributes:
        metrics (dict[str, Metric]): A dictionary of registered metrics.
        groups (dict[str, str]): A dictionary mapping metric names to groups.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """

        self.metrics: dict[str, Metric] = {}
        self.groups: dict[str, str] = {}

    def register(
        self,
        name: str,
        group: str,
        accumulator: Callable[..., Tensor] = nanmean,
    ) -> None:
        """
        Registers a new metric with the specified name and accumulator function.

        Args:
            name (str): The name of the metric to register.
            group (str): The name of the group to assign the metric to.
            accumulator (Callable[..., Tensor], optional): The function used
                to aggregate values for the metric (defaults to `nanmean`).
        """

        # Type checking
        validate_type("name", name, str)
        validate_type("group", group, str)
        validate_callable("accumulator", accumulator)

        self.metrics[name] = Metric(name, accumulator)
        self.groups[name] = group

    def accumulate(self, name: str, value: Tensor) -> None:
        """
        Accumulates a new value for the specified metric.

        Args:
            name (str): The name of the metric.
            value (Tensor): The new value to accumulate.
        """

        self.metrics[name].accumulate(value)

    def aggregate(self, group: str) -> dict[str, Tensor]:
        """
        Aggregates all metrics in a group using the accumulators
        specified during registration.

        Args:
            group (str): The name of the group.

        Returns:
            dict[str, Tensor]: A dictionary with the metric names and the
                corresponding aggregated values of the selected group.
        """

        return {
            name: metric.aggregate()
            for name, metric in self.metrics.items()
            if self.groups[name] == group
        }

    def reset(self, group: str) -> None:
        """
        Resets all registered metrics in a group.

        Args:
            group (str): The name of the group.
        """

        for name, metric in self.metrics.items():
            if self.groups[name] == group:
                metric.reset()
                metric.reset()
