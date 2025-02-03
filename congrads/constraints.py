"""
This module provides a set of constraint classes for guiding neural network 
training by enforcing specific conditions on the network's outputs.

The constraints in this module include:

- `Constraint`: The base class for all constraint types, defining the 
  interface and core behavior.
- `ImplicationConstraint`: A constraint that enforces one condition only if 
  another condition is met, useful for modeling implications between network 
  outputs.
- `ScalarConstraint`: A constraint that enforces scalar-based comparisons on 
  a network's output.
- `BinaryConstraint`: A constraint that enforces a binary comparison between
  two neurons in the network, using a comparison function (e.g., less than,
  greater than).
- `SumConstraint`: A constraint that enforces that the sum of certain neurons'
  outputs equals a specified value, which can be used to control total output.
- `PythagoreanConstraint`: A constraint that enforces the Pythagorean theorem
  on a set of neurons, ensuring that the square of one neuron's output is equal
  to the sum of the squares of other outputs.

These constraints can be used to steer the learning process by applying 
conditions such as logical implications or numerical bounds.

Usage:
    1. Define a custom constraint class by inheriting from `Constraint`.
    2. Apply the constraint to your neural network during training to 
       enforce desired output behaviors.
    3. Use the helper classes like `IdentityTransformation` for handling 
       transformations and comparisons in constraints.

Dependencies:
    - PyTorch (`torch`)
"""

import random
import string
import warnings
from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Dict, Union

from torch import (
    Tensor,
    count_nonzero,
    ge,
    gt,
    isclose,
    le,
    logical_not,
    logical_or,
    lt,
    numel,
    ones,
    ones_like,
    reshape,
    sign,
    sqrt,
    square,
    stack,
    tensor,
    zeros_like,
)
from torch.nn.functional import normalize

from .descriptor import Descriptor
from .transformations import IdentityTransformation, Transformation
from .utils import validate_comparator_pytorch, validate_iterable, validate_type


class Constraint(ABC):
    """
    Abstract base class for defining constraints applied to neural networks.

    A `Constraint` specifies conditions that the neural network outputs
    should satisfy. It supports monitoring constraint satisfaction
    during training and can adjust loss to enforce constraints. Subclasses
    must implement the `check_constraint` and `calculate_direction` methods.

    Args:
        neurons (set[str]): Names of the neurons this constraint applies to.
        name (str, optional): A unique name for the constraint. If not provided,
            a name is generated based on the class name and a random suffix.
        monitor_only (bool, optional): If True, only monitor the constraint
            without adjusting the loss. Defaults to False.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5. Should be greater
            than 1 to give weight to the constraint.

    Raises:
        TypeError: If a provided attribute has an incompatible type.
        ValueError: If any neuron in `neurons` is not
            defined in the `descriptor`.

    Note:
        - If `rescale_factor <= 1`, a warning is issued, and the value is
          adjusted to a positive value greater than 1.
        - If `name` is not provided, a name is auto-generated,
          and a warning is logged.

    """

    descriptor: Descriptor = None
    device = None

    def __init__(
        self,
        neurons: set[str],
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """
        Initializes a new Constraint instance.
        """

        # Init parent class
        super().__init__()

        # Type checking
        validate_iterable("neurons", neurons, str)
        validate_type("name", name, (str, type(None)))
        validate_type("monitor_only", monitor_only, bool)
        validate_type("rescale_factor", rescale_factor, Number)

        # Init object variables
        self.neurons = neurons
        self.rescale_factor = rescale_factor
        self.monitor_only = monitor_only

        # Perform checks
        if rescale_factor <= 1:
            warnings.warn(
                "Rescale factor for constraint %s is <= 1. The network \
                    will favor general loss over the constraint-adjusted loss. \
                    Is this intended behavior? Normally, the loss should \
                    always be larger than 1.",
                name,
            )

        # If no constraint_name is set, generate one based
        # on the class name and a random suffix
        if name:
            self.name = name
        else:
            random_suffix = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=6)
            )
            self.name = f"{self.__class__.__name__}_{random_suffix}"
            warnings.warn(
                "Name for constraint is not set. Using %s.", self.name
            )

        # If rescale factor is not larger than 1, warn user and adjust
        if rescale_factor <= 1:
            self.rescale_factor = abs(rescale_factor) + 1.5
            warnings.warn(
                "Rescale factor for constraint %s is < 1, adjusted value \
                    %s to %s.",
                name,
                rescale_factor,
                self.rescale_factor,
            )
        else:
            self.rescale_factor = rescale_factor

        # Infer layers from descriptor and neurons
        self.layers = set()
        for neuron in self.neurons:
            if neuron not in self.descriptor.neuron_to_layer.keys():
                raise ValueError(
                    f'The neuron name {neuron} used with constraint \
                        {self.name} is not defined in the descriptor. Please \
                        add it to the correct layer using \
                        descriptor.add("layer", ...).'
                )

            self.layers.add(self.descriptor.neuron_to_layer[neuron])

    @abstractmethod
    def check_constraint(
        self, prediction: dict[str, Tensor]
    ) -> tuple[Tensor, int]:
        """
        Evaluates whether the given model predictions satisfy the constraint.

        Args:
            prediction (dict[str, Tensor]): Model predictions for the neurons.

        Returns:
            tuple[Tensor, int]: A tuple where the first element is a tensor
            indicating whether the constraint is satisfied (with `True`
            for satisfaction, `False` for non-satisfaction, and `torch.nan`
            for irrelevant results), and the second element is an integer
            value representing the number of relevant constraints.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def calculate_direction(
        self, prediction: dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Calculates adjustment directions for neurons to
        better satisfy the constraint.

        Args:
            prediction (dict[str, Tensor]): Model predictions for the neurons.

        Returns:
            Dict[str, Tensor]: Dictionary mapping neuron layers to tensors
            specifying the adjustment direction for each neuron.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """

        raise NotImplementedError


class ImplicationConstraint(Constraint):
    """
    Represents an implication constraint between two
    constraints (head and body).

    The implication constraint ensures that the `body` constraint only applies
    when the `head` constraint is satisfied. If the `head` constraint is not
    satisfied, the `body` constraint does not apply.

    Args:
        head (Constraint): The head of the implication. If this constraint
            is satisfied, the body constraint must also be satisfied.
        body (Constraint): The body of the implication. This constraint
            is enforced only when the head constraint is satisfied.
        name (str, optional): A unique name for the constraint. If not
            provided, the name is generated in the format
            "{body.name} if {head.name}". Defaults to None.
        monitor_only (bool, optional): If True, the constraint is only
            monitored without adjusting the loss. Defaults to False.
        rescale_factor (Number, optional): The scaling factor for the
            constraint-adjusted loss. Defaults to 1.5.

    Raises:
        TypeError: If a provided attribute has an incompatible type.

    """

    def __init__(
        self,
        head: Constraint,
        body: Constraint,
        name=None,
        monitor_only=False,
        rescale_factor=1.5,
    ):
        """
        Initializes an ImplicationConstraint instance.
        """

        # Type checking
        validate_type("head", head, Constraint)
        validate_type("body", body, Constraint)

        # Compose constraint name
        name = f"{body.name} if {head.name}"

        # Init parent class
        super().__init__(
            head.neurons | body.neurons,
            name,
            monitor_only,
            rescale_factor,
        )

        self.head = head
        self.body = body

    def check_constraint(
        self, prediction: dict[str, Tensor]
    ) -> tuple[Tensor, int]:

        # Check satisfaction of head and body constraints
        head_satisfaction, _ = self.head.check_constraint(prediction)
        body_satisfaction, _ = self.body.check_constraint(prediction)

        # If head constraint is satisfied (returning 1),
        # the body constraint matters (and should return 0/1 based on body)
        # If head constraint is not satisfied (returning 0),
        # the body constraint does not apply (and should return 1)
        result = logical_or(
            logical_not(head_satisfaction), body_satisfaction
        ).float()

        return result, count_nonzero(head_satisfaction)

    def calculate_direction(
        self, prediction: dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # NOTE currently only works for dense layers
        # due to neuron to index translation

        # Use directions of constraint body as update vector
        return self.body.calculate_direction(prediction)


class ScalarConstraint(Constraint):
    """
    A constraint that enforces scalar-based comparisons on a specific neuron.

    This class ensures that the output of a specified neuron satisfies a scalar
    comparison operation (e.g., less than, greater than, etc.). It uses a
    comparator function to validate the condition and calculates adjustment
    directions accordingly.

    Args:
        operand (Union[str, Transformation]): Name of the neuron or a
            transformation to apply.
        comparator (Callable[[Tensor, Number], Tensor]): A comparison
            function (e.g., `torch.ge`, `torch.lt`).
        scalar (Number): The scalar value to compare against.
        name (str, optional): A unique name for the constraint. If not
            provided, a name is auto-generated in the format
            "<neuron_name> <comparator> <scalar>".
        monitor_only (bool, optional): If True, only monitor the constraint
            without adjusting the loss. Defaults to False.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5.

    Raises:
        TypeError: If a provided attribute has an incompatible type.

    Notes:
        - The `neuron_name` must be defined in the `descriptor` mapping.
        - The constraint name is composed using the neuron name,
          comparator, and scalar value.

    """

    def __init__(
        self,
        operand: Union[str, Transformation],
        comparator: Callable[[Tensor, Number], Tensor],
        scalar: Number,
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """
        Initializes a ScalarConstraint instance.
        """

        # Type checking
        validate_type("operand", operand, (str, Transformation))
        validate_comparator_pytorch("comparator", comparator)
        validate_comparator_pytorch("comparator", comparator)
        validate_type("scalar", scalar, Number)

        # If transformation is provided, get neuron name,
        # else use IdentityTransformation
        if isinstance(operand, Transformation):
            neuron_name = operand.neuron_name
            transformation = operand
        else:
            neuron_name = operand
            transformation = IdentityTransformation(neuron_name)

        # Compose constraint name
        name = f"{neuron_name} {comparator.__name__} {str(scalar)}"

        # Init parent class
        super().__init__({neuron_name}, name, monitor_only, rescale_factor)

        # Init variables
        self.comparator = comparator
        self.scalar = scalar
        self.transformation = transformation

        # Get layer name and feature index from neuron_name
        self.layer = self.descriptor.neuron_to_layer[neuron_name]
        self.index = self.descriptor.neuron_to_index[neuron_name]

        # Calculate directions based on constraint operator
        if self.comparator in [lt, le]:
            self.direction = -1
        elif self.comparator in [gt, ge]:
            self.direction = 1

    def check_constraint(
        self, prediction: dict[str, Tensor]
    ) -> tuple[Tensor, int]:

        # Select relevant columns
        selection = prediction[self.layer][:, self.index]

        # Apply transformation
        selection = self.transformation(selection)

        # Calculate current constraint result
        result = self.comparator(selection, self.scalar).float()
        return result, numel(result)

    def calculate_direction(
        self, prediction: dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # NOTE currently only works for dense layers due
        # to neuron to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(prediction[layer][0], device=self.device)

        output[self.layer][self.index] = self.direction

        for layer in self.layers:
            output[layer] = normalize(reshape(output[layer], [1, -1]), dim=1)

        return output


class BinaryConstraint(Constraint):
    """
    A constraint that enforces a binary comparison between two neurons.

    This class ensures that the output of one neuron satisfies a comparison
    operation with the output of another neuron
    (e.g., less than, greater than, etc.). It uses a comparator function to
    validate the condition and calculates adjustment directions accordingly.

    Args:
        operand_left (Union[str, Transformation]): Name of the left
            neuron or a transformation to apply.
        comparator (Callable[[Tensor, Number], Tensor]): A comparison
            function (e.g., `torch.ge`, `torch.lt`).
        operand_right (Union[str, Transformation]): Name of the right
            neuron or a transformation to apply.
        name (str, optional): A unique name for the constraint. If not
            provided, a name is auto-generated in the format
            "<neuron_name_left> <comparator> <neuron_name_right>".
        monitor_only (bool, optional): If True, only monitor the constraint
            without adjusting the loss. Defaults to False.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5.

    Raises:
        TypeError: If a provided attribute has an incompatible type.

    Notes:
        - The neuron names must be defined in the `descriptor` mapping.
        - The constraint name is composed using the left neuron name,
          comparator, and right neuron name.

    """

    def __init__(
        self,
        operand_left: Union[str, Transformation],
        comparator: Callable[[Tensor, Number], Tensor],
        operand_right: Union[str, Transformation],
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """
        Initializes a BinaryConstraint instance.
        """

        # Type checking
        validate_type("operand_left", operand_left, (str, Transformation))
        validate_comparator_pytorch("comparator", comparator)
        validate_comparator_pytorch("comparator", comparator)
        validate_type("operand_right", operand_right, (str, Transformation))

        # If transformation is provided, get neuron name,
        # else use IdentityTransformation
        if isinstance(operand_left, Transformation):
            neuron_name_left = operand_left.neuron_name
            transformation_left = operand_left
        else:
            neuron_name_left = operand_left
            transformation_left = IdentityTransformation(neuron_name_left)

        if isinstance(operand_right, Transformation):
            neuron_name_right = operand_right.neuron_name
            transformation_right = operand_right
        else:
            neuron_name_right = operand_right
            transformation_right = IdentityTransformation(neuron_name_right)

        # Compose constraint name
        name = f"{neuron_name_left} {comparator.__name__} {neuron_name_right}"

        # Init parent class
        super().__init__(
            {neuron_name_left, neuron_name_right},
            name,
            monitor_only,
            rescale_factor,
        )

        # Init variables
        self.comparator = comparator
        self.transformation_left = transformation_left
        self.transformation_right = transformation_right

        # Get layer name and feature index from neuron_name
        self.layer_left = self.descriptor.neuron_to_layer[neuron_name_left]
        self.layer_right = self.descriptor.neuron_to_layer[neuron_name_right]
        self.index_left = self.descriptor.neuron_to_index[neuron_name_left]
        self.index_right = self.descriptor.neuron_to_index[neuron_name_right]

        # Calculate directions based on constraint operator
        if self.comparator in [lt, le]:
            self.direction_left = -1
            self.direction_right = 1
        else:
            self.direction_left = 1
            self.direction_right = -1

    def check_constraint(
        self, prediction: dict[str, Tensor]
    ) -> tuple[Tensor, int]:

        # Select relevant columns
        selection_left = prediction[self.layer_left][:, self.index_left]
        selection_right = prediction[self.layer_right][:, self.index_right]

        # Apply transformations
        selection_left = self.transformation_left(selection_left)
        selection_right = self.transformation_right(selection_right)

        result = self.comparator(selection_left, selection_right).float()

        return result, numel(result)

    def calculate_direction(
        self, prediction: dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # NOTE currently only works for dense layers due
        # to neuron to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(prediction[layer][0], device=self.device)

        output[self.layer_left][self.index_left] = self.direction_left
        output[self.layer_right][self.index_right] = self.direction_right

        for layer in self.layers:
            output[layer] = normalize(reshape(output[layer], [1, -1]), dim=1)

        return output


class SumConstraint(Constraint):
    """
    A constraint that enforces a weighted summation comparison
    between two groups of neurons.

    This class evaluates whether the weighted sum of outputs from one set of
    neurons satisfies a comparison operation with the weighted sum of
    outputs from another set of neurons.

    Args:
        operands_left (list[Union[str, Transformation]]): List of neuron
            names or transformations on the left side.
        comparator (Callable[[Tensor, Number], Tensor]): A comparison
            function for the constraint.
        operands_right (list[Union[str, Transformation]]): List of neuron
            names or transformations on the right side.
        weights_left (list[Number], optional): Weights for the left neurons.
            Defaults to None.
        weights_right (list[Number], optional): Weights for the right
            neurons. Defaults to None.
        name (str, optional): Unique name for the constraint.
            If None, it's auto-generated. Defaults to None.
        monitor_only (bool, optional): If True, only monitor the constraint
            without adjusting the loss. Defaults to False.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5.

    Raises:
        TypeError: If a provided attribute has an incompatible type.
        ValueError: If the dimensions of neuron names and weights mismatch.

    """

    def __init__(
        self,
        operands_left: list[Union[str, Transformation]],
        comparator: Callable[[Tensor, Number], Tensor],
        operands_right: list[Union[str, Transformation]],
        weights_left: list[Number] = None,
        weights_right: list[Number] = None,
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """
        Initializes the SumConstraint.
        """

        # Type checking
        validate_iterable("operands_left", operands_left, (str, Transformation))
        validate_comparator_pytorch("comparator", comparator)
        validate_comparator_pytorch("comparator", comparator)
        validate_iterable(
            "operands_right", operands_right, (str, Transformation)
        )
        validate_iterable("weights_left", weights_left, Number, allow_none=True)
        validate_iterable(
            "weights_right", weights_right, Number, allow_none=True
        )

        # If transformation is provided, get neuron name,
        # else use IdentityTransformation
        neuron_names_left: list[str] = []
        transformations_left: list[Transformation] = []
        for operand_left in operands_left:
            if isinstance(operand_left, Transformation):
                neuron_name_left = operand_left.neuron_name
                neuron_names_left.append(neuron_name_left)
                transformations_left.append(operand_left)
            else:
                neuron_name_left = operand_left
                neuron_names_left.append(neuron_name_left)
                transformations_left.append(
                    IdentityTransformation(neuron_name_left)
                )

        neuron_names_right: list[str] = []
        transformations_right: list[Transformation] = []
        for operand_right in operands_right:
            if isinstance(operand_right, Transformation):
                neuron_name_right = operand_right.neuron_name
                neuron_names_right.append(neuron_name_right)
                transformations_right.append(operand_right)
            else:
                neuron_name_right = operand_right
                neuron_names_right.append(neuron_name_right)
                transformations_right.append(
                    IdentityTransformation(neuron_name_right)
                )

        # Compose constraint name
        w_left = weights_left or [""] * len(neuron_names_left)
        w_right = weights_right or [""] * len(neuron_names_right)
        left_expr = " + ".join(
            f"{w}{n}" for w, n in zip(w_left, neuron_names_left)
        )
        right_expr = " + ".join(
            f"{w}{n}" for w, n in zip(w_right, neuron_names_right)
        )
        comparator_name = comparator.__name__
        name = f"{left_expr} {comparator_name} {right_expr}"

        # Init parent class
        neuron_names = set(neuron_names_left) | set(neuron_names_right)
        super().__init__(neuron_names, name, monitor_only, rescale_factor)

        # Init variables
        self.comparator = comparator
        self.neuron_names_left = neuron_names_left
        self.neuron_names_right = neuron_names_right
        self.transformations_left = transformations_left
        self.transformations_right = transformations_right

        # If feature list dimensions don't match
        # weight list dimensions, raise error
        if weights_left and (len(neuron_names_left) != len(weights_left)):
            raise ValueError(
                "The dimensions of neuron_names_left don't match with the \
                    dimensions of weights_left."
            )
        if weights_right and (len(neuron_names_right) != len(weights_right)):
            raise ValueError(
                "The dimensions of neuron_names_right don't match with the \
                    dimensions of weights_right."
            )

        # If weights are provided for summation, transform them to Tensors
        if weights_left:
            self.weights_left = tensor(weights_left, device=self.device)
        else:
            self.weights_left = ones(len(neuron_names_left), device=self.device)
        if weights_right:
            self.weights_right = tensor(weights_right, device=self.device)
        else:
            self.weights_right = ones(
                len(neuron_names_right), device=self.device
            )

        # Calculate directions based on constraint operator
        if self.comparator in [lt, le]:
            self.direction_left = -1
            self.direction_right = 1
        else:
            self.direction_left = 1
            self.direction_right = -1

    def check_constraint(
        self, prediction: dict[str, Tensor]
    ) -> tuple[Tensor, int]:

        def compute_weighted_sum(
            neuron_names: list[str],
            transformations: list[Transformation],
            weights: tensor,
        ) -> tensor:
            layers = [
                self.descriptor.neuron_to_layer[neuron_name]
                for neuron_name in neuron_names
            ]
            indices = [
                self.descriptor.neuron_to_index[neuron_name]
                for neuron_name in neuron_names
            ]

            # Select relevant column
            selections = [
                prediction[layer][:, index]
                for layer, index in zip(layers, indices)
            ]

            # Apply transformations
            results = []
            for transformation, selection in zip(transformations, selections):
                results.append(transformation(selection))

            # Extract predictions for all neurons and apply weights in bulk
            predictions = stack(
                results,
                dim=1,
            )

            # Calculate weighted sum
            return (predictions * weights.unsqueeze(0)).sum(dim=1)

        # Compute weighted sums
        weighted_sum_left = compute_weighted_sum(
            self.neuron_names_left,
            self.transformations_left,
            self.weights_left,
        )
        weighted_sum_right = compute_weighted_sum(
            self.neuron_names_right,
            self.transformations_right,
            self.weights_right,
        )

        # Apply the comparator and calculate the result
        result = self.comparator(weighted_sum_left, weighted_sum_right).float()

        return result, numel(result)

    def calculate_direction(
        self, prediction: dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # NOTE currently only works for dense layers
        # due to neuron to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(prediction[layer][0], device=self.device)

        for neuron_name_left in self.neuron_names_left:
            layer = self.descriptor.neuron_to_layer[neuron_name_left]
            index = self.descriptor.neuron_to_index[neuron_name_left]
            output[layer][index] = self.direction_left

        for neuron_name_right in self.neuron_names_right:
            layer = self.descriptor.neuron_to_layer[neuron_name_right]
            index = self.descriptor.neuron_to_index[neuron_name_right]
            output[layer][index] = self.direction_right

        for layer in self.layers:
            output[layer] = normalize(reshape(output[layer], [1, -1]), dim=1)

        return output


class PythagoreanIdentityConstraint(Constraint):
    """
    A constraint that enforces the Pythagorean identity: a² + b² ≈ 1,
    where `a` and `b` are neurons or transformations.

    This constraint checks that the sum of the squares of two specified
    neurons (or their transformations) is approximately equal to 1.
    The constraint is evaluated using relative and absolute
    tolerance (`rtol` and `atol`) and is applied during the forward pass.

    Args:
        a (Union[str, Transformation]): The first input, either a
            neuron name (str) or a Transformation.
        b (Union[str, Transformation]): The second input, either a
            neuron name (str) or a Transformation.
        rtol (float, optional): The relative tolerance for the
            comparison (default is 0.00001).
        atol (float, optional): The absolute tolerance for the
            comparison (default is 1e-8).
        name (str, optional): The name of the constraint
            (default is None, and it is generated automatically).
        monitor_only (bool, optional): Flag indicating whether the
            constraint is only for monitoring (default is False).
        rescale_factor (Number, optional): A factor used for
            rescaling (default is 1.5).

    Raises:
        TypeError: If a provided attribute has an incompatible type.

    """

    def __init__(
        self,
        a: Union[str, Transformation],
        b: Union[str, Transformation],
        rtol: float = 0.00001,
        atol: float = 1e-8,
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """
        Initialize the PythagoreanIdentityConstraint.
        """

        # Type checking
        validate_type("a", a, (str, Transformation))
        validate_type("b", b, (str, Transformation))
        validate_type("rtol", rtol, float)
        validate_type("atol", atol, float)

        # If transformation is provided, get neuron name,
        # else use IdentityTransformation
        if isinstance(a, Transformation):
            neuron_name_a = a.neuron_name
            transformation_a = a
        else:
            neuron_name_a = a
            transformation_a = IdentityTransformation(neuron_name_a)

        if isinstance(b, Transformation):
            neuron_name_b = b.neuron_name
            transformation_b = b
        else:
            neuron_name_b = b
            transformation_b = IdentityTransformation(neuron_name_b)

        # Compose constraint name
        name = f"{neuron_name_a}² + {neuron_name_b}² ≈ 1"

        # Init parent class
        super().__init__(
            {neuron_name_a, neuron_name_b},
            name,
            monitor_only,
            rescale_factor,
        )

        # Init variables
        self.transformation_a = transformation_a
        self.transformation_b = transformation_b
        self.rtol = rtol
        self.atol = atol

        # Get layer name and feature index from neuron_name
        self.layer_a = self.descriptor.neuron_to_layer[neuron_name_a]
        self.layer_b = self.descriptor.neuron_to_layer[neuron_name_b]
        self.index_a = self.descriptor.neuron_to_index[neuron_name_a]
        self.index_b = self.descriptor.neuron_to_index[neuron_name_b]

    def check_constraint(
        self, prediction: dict[str, Tensor]
    ) -> tuple[Tensor, int]:

        # Select relevant columns
        selection_a = prediction[self.layer_a][:, self.index_a]
        selection_b = prediction[self.layer_b][:, self.index_b]

        # Apply transformations
        selection_a = self.transformation_a(selection_a)
        selection_b = self.transformation_b(selection_b)

        # Calculate result
        result = isclose(
            square(selection_a) + square(selection_b),
            ones_like(selection_a, device=self.device),
            rtol=self.rtol,
            atol=self.atol,
        ).float()

        return result, numel(result)

    def calculate_direction(
        self, prediction: dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # NOTE currently only works for dense layers due
        # to neuron to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(prediction[layer], device=self.device)

        a = prediction[self.layer_a][:, self.index_a]
        b = prediction[self.layer_b][:, self.index_b]
        m = sqrt(square(a) + square(b))

        output[self.layer_a][:, self.index_a] = a / m * sign(1 - m)
        output[self.layer_b][:, self.index_b] = b / m * sign(1 - m)

        return output
