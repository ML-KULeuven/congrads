"""Rescale Strategies for Constraint-Based Loss Adjustment.

This module provides classes to compute rescale factors for constraints
during neural network training. Strategies can be static or adaptive,
and multiple strategies can be chained together to modify the scaling
effectively.
"""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Union

from torch import Tensor

__all__ = [
    "RescaleStrategy",
    "StaticStrategy",
    "ImplicationBalancedStrategy",
]


class RescaleStrategy(ABC):
    """Abstract base class for adaptive rescale strategies used with constraints.

    A `RescaleStrategy` defines how a rescale factor should be computed during training.
    It supports chaining, allowing one strategy to build on top of another.

    Args:
        base (Number | RescaleStrategy, optional): Either a numeric rescale factor value to act
        as a base or another RescaleStrategy to chain. Defaults to 1.5.
    """

    def __init__(self, base: Union[Number, "RescaleStrategy"] = 1.5):
        """Initialize the rescale strategy with a base value or strategy."""
        if isinstance(base, RescaleStrategy):
            self._inner = base
        elif isinstance(base, Number):
            self._inner = StaticStrategy(base)
        else:
            raise ValueError("Base must be a number or a RescaleStrategy instance.")

    def compute(
        self,
        data: dict[str, Tensor],
        constraint_checks: Tensor,
        constraint_mask: Tensor,
        constraint_directions: dict[str, Tensor],
        loss: Tensor,
    ) -> Tensor:
        """Compute the effective rescale factor for the given batch.

        The resulting rescale factor combines the output of the base strategy (which may itself
        be a chain of strategies) with the modifier computed by this strategy, by multiplying.

        Args:
            data (dict[str, Tensor]): Batch data including but not limiting to inputs, outputs and targets.
            constraint_checks (Tensor): Tensor indicating which batch samples are satisfied.
            constraint_mask (Tensor): Tensor indicating relevant batch samples.
            constraint_directions (dict[str, Tensor]): Adjustment directions per layer.
            loss (Tensor): Original loss tensor.

        Returns:
            Tensor: The computed rescale factor to apply to the loss.
        """
        base_value = self._inner.compute(
            data, constraint_checks, constraint_mask, constraint_directions, loss
        )
        modifier = self.compute_modifier(
            data, constraint_checks, constraint_mask, constraint_directions, loss
        )
        return base_value * modifier

    @abstractmethod
    def compute_modifier(
        self,
        data: dict[str, Tensor],
        constraint_checks: Tensor,
        constraint_mask: Tensor,
        constraint_directions: dict[str, Tensor],
        loss: Tensor,
    ) -> Tensor | Number:
        """Compute the modifier relative to the base strategy.

        This method should only return the factor to multiply the base value
        with. Subclasses implement their specific logic here.

        Args:
            data (dict[str, Tensor]): Batch data including inputs and model outputs.
            constraint_checks (Tensor): Tensor indicating which batch samples are satisfied.
            constraint_mask (Tensor): Tensor indicating relevant batch samples.
            constraint_directions (dict[str, Tensor]): Adjustment directions per layer.
            loss (Tensor): Original loss tensor.

        Returns:
            Tensor | Number: The multiplicative modifier for the base rescale factor.
        """
        pass


class StaticStrategy(RescaleStrategy):
    """A leaf rescale strategy that always returns a fixed value.

    This is the simplest rescale strategy and is typically used as a base
    in a chain or when a fixed scaling factor is desired.

    Args:
        value (Number): The fixed rescale factor to apply.
    """

    def __init__(self, value: Number):
        """Initialize the static strategy with a fixed value."""
        self.value = value

    def compute(self, *args, **kwargs):
        """Return the fixed base value as a tensor-compatible number."""
        return self.value

    def compute_modifier(self, *args, **kwargs):
        """Static strategy has no modifier; always returns 1.0."""
        return 1.0


class ImplicationBalancedStrategy(RescaleStrategy):
    """Rescale strategy that adjusts constraint strength based on sample relevance.

    The effective rescale factor increases when fewer samples are relevant, ensuring constraints
    that apply to a subset of the batch are weighted comparably to those that apply to the entire batch.

    Example:
        modifier = num_total_samples / num_relevant_samples

    Chaining with a base strategy is supported.

    Args:
        base (Number | RescaleStrategy, optional): Base value or strategy to chain.
            Defaults to 1.5.
    """

    def compute_modifier(
        self, data, constraint_checks, constraint_mask, constraint_directions, loss
    ):
        """Compute modifier as inverse proportion of relevant samples.

        Returns:
            Tensor | Number: Scaling factor that amplifies constraint effect
            when fewer samples are relevant.
        """
        return constraint_mask.numel() / constraint_mask.sum().clamp_min(1.0)
