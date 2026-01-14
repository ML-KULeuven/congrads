"""Defines the abstract base class `Constraint` for specifying constraints on neural network outputs.

A `Constraint` monitors whether the network predictions satisfy certain
conditions during training, validation, and testing. It can optionally
adjust the loss to enforce constraints, and logs the relevant metrics.

Responsibilities:
- Track which network layers/tags the constraint applies to
- Check constraint satisfaction for a batch of predictions
- Compute adjustment directions to enforce the constraint
- Provide a rescale factor and enforcement flag to influence loss adjustment

Subclasses must implement the abstract methods:
- `check_constraint(data)`: Evaluate constraint satisfaction for a batch
- `calculate_direction(data)`: Compute directions to adjust predictions
"""

import random
import string
import warnings
from abc import ABC, abstractmethod
from numbers import Number

from torch import Tensor

from congrads.descriptor import Descriptor
from congrads.utils.validation import validate_iterable, validate_type


class Constraint(ABC):
    """Abstract base class for defining constraints applied to neural networks.

    A `Constraint` specifies conditions that the neural network outputs
    should satisfy. It supports monitoring constraint satisfaction
    during training and can adjust loss to enforce constraints. Subclasses
    must implement the `check_constraint` and `calculate_direction` methods.

    Args:
        tags (set[str]): Tags referencing parts of the network where this constraint applies to.
        name (str, optional): A unique name for the constraint. If not provided,
            a name is generated based on the class name and a random suffix.
        enforce (bool, optional): If False, only monitor the constraint
            without adjusting the loss. Defaults to True.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5. Should be greater
            than 1 to give weight to the constraint.

    Raises:
        TypeError: If a provided attribute has an incompatible type.
        ValueError: If any tag in `tags` is not
            defined in the `descriptor`.

    Note:
        - If `rescale_factor <= 1`, a warning is issued.
        - If `name` is not provided, a name is auto-generated,
          and a warning is logged.

    """

    descriptor: Descriptor = None
    device = None

    def __init__(
        self, tags: set[str], name: str = None, enforce: bool = True, rescale_factor: Number = 1.5
    ) -> None:
        """Initializes a new Constraint instance.

        Args:
            tags (set[str]): Tags referencing parts of the network where this constraint applies to.
            name (str, optional): A unique name for the constraint. If not
                provided, a name is generated based on the class name and a
                random suffix.
            enforce (bool, optional): If False, only monitor the constraint
                without adjusting the loss. Defaults to True.
            rescale_factor (Number, optional): Factor to scale the
                constraint-adjusted loss. Defaults to 1.5. Should be greater
                than 1 to give weight to the constraint.

        Raises:
            TypeError: If a provided attribute has an incompatible type.
            ValueError: If any tag in `tags` is not defined in the `descriptor`.

        Note:
            - If `rescale_factor <= 1`, a warning is issued.
            - If `name` is not provided, a name is auto-generated, and a
            warning is logged.
        """
        # Init parent class
        super().__init__()

        # Type checking
        validate_iterable("tags", tags, str)
        validate_type("name", name, str, allow_none=True)
        validate_type("enforce", enforce, bool)
        validate_type("rescale_factor", rescale_factor, Number)

        # Init object variables
        self.tags = tags
        self.rescale_factor = rescale_factor
        self.initial_rescale_factor = rescale_factor
        self.enforce = enforce

        # Perform checks
        if rescale_factor <= 1:
            warnings.warn(
                f"Rescale factor for constraint {name} is <= 1. The network "
                "will favor general loss over the constraint-adjusted loss. "
                "Is this intended behavior? Normally, the rescale factor "
                "should always be larger than 1.",
                stacklevel=2,
            )

        # If no constraint_name is set, generate one based
        # on the class name and a random suffix
        if name:
            self.name = name
        else:
            random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            self.name = f"{self.__class__.__name__}_{random_suffix}"
            warnings.warn(f"Name for constraint is not set. Using {self.name}.", stacklevel=2)

        # Infer layers from descriptor and tags
        self.layers = set()
        for tag in self.tags:
            if not self.descriptor.exists(tag):
                raise ValueError(
                    f"The tag {tag} used with constraint "
                    f"{self.name} is not defined in the descriptor. Please "
                    "add it to the correct layer using "
                    "descriptor.add('layer', ...)."
                )

            layer, _ = self.descriptor.location(tag)
            self.layers.add(layer)

    @abstractmethod
    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluates whether the given model predictions satisfy the constraint.

        1 IS SATISFIED, 0 IS NOT SATISFIED

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            tuple[Tensor, Tensor]: A tuple where the first element is a tensor of floats
            indicating whether the constraint is satisfied (with value 1.0
            for satisfaction, and 0.0 for non-satisfaction, and the second element is a tensor
            mask that indicates the relevance of each sample (`True` for relevant
            samples and `False` for irrelevant ones).

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute adjustment directions to better satisfy the constraint.

        Given the model predictions, input batch, and context, this method calculates the direction
        in which the predictions referenced by a tag should be adjusted to satisfy the constraint.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            dict[str, Tensor]: Dictionary mapping network layers to tensors that
                specify the adjustment direction for each tag.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
