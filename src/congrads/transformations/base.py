"""Module defining transformations and components."""

from abc import ABC, abstractmethod

from torch import Tensor

from ..utils.validation import validate_type


class Transformation(ABC):
    """Abstract base class for tag data transformations."""

    def __init__(self, tag: str):
        """Initialize a Transformation.

        Args:
            tag (str): Tag this transformation applies to.
        """
        validate_type("tag", tag, str)

        super().__init__()
        self.tag = tag

    @abstractmethod
    def __call__(self, data: Tensor) -> Tensor:
        """Apply the transformation to the input tensor.

        Args:
            data (Tensor): Input tensor representing network data.

        Returns:
            Tensor: Transformed tensor.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
