"""Module holding specific transformation implementations."""

from numbers import Number

from torch import Tensor

from ..utils.validation import validate_callable, validate_type
from .base import Transformation


class IdentityTransformation(Transformation):
    """A transformation that returns the input unchanged."""

    def __call__(self, data: Tensor) -> Tensor:
        """Return the input tensor without any modification.

        Args:
            data (Tensor): Input tensor.

        Returns:
            Tensor: The same input tensor.
        """
        return data


class DenormalizeMinMax(Transformation):
    """A transformation that denormalizes data using min-max scaling."""

    def __init__(self, tag: str, min: Number, max: Number):
        """Initialize a min-max denormalization transformation.

        Args:
            tag (str): Tag this transformation applies to.
            min (Number): Minimum value used for denormalization.
            max (Number): Maximum value used for denormalization.
        """
        validate_type("min", min, Number)
        validate_type("max", max, Number)

        super().__init__(tag)

        self.min = min
        self.max = max

    def __call__(self, data: Tensor) -> Tensor:
        """Denormalize the input tensor using the min-max range.

        Args:
            data (Tensor): Normalized input tensor (typically in range [0, 1]).

        Returns:
            Tensor: Denormalized tensor in the range [min, max].
        """
        return data * (self.max - self.min) + self.min


class ApplyOperator(Transformation):
    """A transformation that applies a binary operator to the input tensor."""

    def __init__(self, tag: str, operator: callable, value: Number):
        """Initialize an operator-based transformation.

        Args:
            tag (str): Tag this transformation applies to.
            operator (callable): A callable that takes two arguments (tensor, value)
                and returns a tensor.
            value (Number): The value to use as the second argument in the operator.
        """
        validate_callable("operator", operator)
        validate_type("value", value, Number)

        super().__init__(tag)

        self.operator = operator
        self.value = value

    def __call__(self, data: Tensor) -> Tensor:
        """Apply the operator to the input tensor and the specified value.

        Args:
            data (Tensor): Input tensor.

        Returns:
            Tensor: Result of applying `operator(data, value)`.
        """
        return self.operator(data, self.value)
