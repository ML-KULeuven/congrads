"""
This module defines the abstract base class `Transformation` and two
specific transformations: `IdentityTransformation` and `DenormalizeMinMax`.
These transformations are used to apply operations to neuron data.

Classes:

    - Transformation: An abstract base class for transformations that
      can be applied to neuron data. Subclasses must implement the
      `__call__` method to apply the transformation.
    - IdentityTransformation: A subclass of `Transformation` that
      returns the input data unchanged.
    - DenormalizeMinMax: A subclass of `Transformation` that denormalizes
      input data based on specified minimum and maximum values.

Key Methods:

    - `__call__(data: Tensor) -> Tensor`: Abstract method in the 
      `Transformation` class that must be implemented by subclasses to apply 
      a transformation to the input data.
    - `__init__(neuron_name: str)`: Initializes the transformation with the
      associated neuron name.
    - `IdentityTransformation.__call__(data: Tensor) -> Tensor`: Returns
      the input data without applying any transformation.
    - `DenormalizeMinMax.__call__(data: Tensor) -> Tensor`: Denormalizes
      the input data by scaling it based on the specified min and max values.

The `Transformation` class is intended as a base class for creating
custom transformations for neuron data, while the `IdentityTransformation`
is used when no transformation is desired, and `DenormalizeMinMax` is used
for reversing the normalization process by using a min-max scaling approach.
"""

from abc import ABC, abstractmethod
from numbers import Number

from torch import Tensor

from .utils import validate_type


class Transformation(ABC):
    """
    Abstract base class for transformations applied to neuron data.

    Args:
        neuron_name (str): The name of the neuron associated with
            the transformation.

    Methods:
        __call__(data: Tensor) -> Tensor:
            Applies the transformation to the provided data.
            Must be implemented by subclasses.
    """

    def __init__(self, neuron_name: str):
        validate_type("neuron_name", neuron_name, str)

        super().__init__()
        self.neuron_name = neuron_name

    @abstractmethod
    def __call__(self, data: Tensor) -> Tensor:
        """
        Abstract method to apply the transformation to the given data.

        Args:
            data (Tensor): The input data to be transformed.

        Returns:
            Tensor: The transformed data.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class IdentityTransformation(Transformation):
    """
    A transformation that returns the data unchanged (identity transformation).

    Inherits from the Transformation class and implements the
    __call__ method to return the input data as is.
    """

    def __call__(self, data: Tensor) -> Tensor:
        """
        Returns the input data without any transformation.

        Args:
            data (Tensor): The input data to be returned as is.

        Returns:
            Tensor: The unchanged input data.
        """
        return data


class DenormalizeMinMax(Transformation):
    """
    A transformation that denormalizes data based on a
    specified min and max value.

    This transformation scales the data by the range of the min and max values,
    then adds the min value to denormalize it back to the original scale.

    Args:
        neuron_name (str): The name of the neuron associated with
            the transformation.
        min (Number): The minimum value to scale the data.
        max (Number): The maximum value to scale the data.

    Methods:
        __call__(data: Tensor) -> Tensor:
            Applies the denormalization to the given data by scaling it
            with the min and max values.
    """

    # pylint: disable-next=redefined-builtin
    def __init__(self, neuron_name: str, min: Number, max: Number):
        validate_type("min", min, Number)
        validate_type("max", max, Number)

        super().__init__(neuron_name)

        self.min = min
        self.max = max

    def __call__(self, data: Tensor) -> Tensor:
        """
        Denormalizes the input data based on the min and max values.

        Args:
            data (Tensor): The normalized input data to be denormalized.

        Returns:
            Tensor: The denormalized data.
        """
        return data * (self.max - self.min) + self.min
