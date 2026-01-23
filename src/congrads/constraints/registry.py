"""Module providing constraint classes for guiding neural network training.

This module defines constraints that enforce specific conditions on network outputs
to steer learning. Available constraint types include:

- `Constraint`: Base class for all constraint types, defining the interface and core
  behavior.
- `ImplicationConstraint`: Enforces one condition only if another condition is met,
  useful for modeling implications between outputs.
- `ScalarConstraint`: Enforces scalar-based comparisons on a network's output.
- `BinaryConstraint`: Enforces a binary comparison between two tags using a
  comparison function (e.g., less than, greater than).
- `SumConstraint`: Ensures the sum of selected tags' outputs equals a specified
  value, controlling total output.

These constraints can steer the learning process by applying logical implications
or numerical bounds.

Usage:
    1. Define a custom constraint class by inheriting from `Constraint`.
    2. Apply the constraint to your neural network during training.
    3. Use helper classes like `IdentityTransformation` for transformations and
       comparisons in constraints.

"""

from collections.abc import Callable
from numbers import Number
from typing import Literal

import torch
from torch import (
    Tensor,
    argsort,
    eq,
    logical_and,
    logical_not,
    logical_or,
    ones,
    ones_like,
    reshape,
    sign,
    sort,
    stack,
    tensor,
    triu,
    unique,
    zeros_like,
)
from torch.nn.functional import normalize

from ..transformations.base import Transformation
from ..transformations.registry import IdentityTransformation
from ..utils.validation import validate_comparator, validate_iterable, validate_type
from .base import Constraint, MonotonicityConstraint

COMPARATOR_MAP: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    ">": torch.gt,
    ">=": torch.ge,
    "<": torch.lt,
    "<=": torch.le,
}


class ImplicationConstraint(Constraint):
    """Represents an implication constraint between two constraints (head and body).

    The implication constraint ensures that the `body` constraint only applies
    when the `head` constraint is satisfied. If the `head` constraint is not
    satisfied, the `body` constraint does not apply.
    """

    def __init__(
        self,
        head: Constraint,
        body: Constraint,
        name: str = None,
    ):
        """Initializes an ImplicationConstraint instance.

        Uses `enforce` and `rescale_factor` from the body constraint.

        Args:
            head (Constraint): Constraint defining the head of the implication.
            body (Constraint): Constraint defining the body of the implication.
            name (str, optional): A unique name for the constraint. If not
                provided, a name is generated based on the class name and a
                random suffix.

        Raises:
            TypeError: If a provided attribute has an incompatible type.

        """
        # Type checking
        validate_type("head", head, Constraint)
        validate_type("body", body, Constraint)

        # Compose constraint name
        name = f"{body.name} if {head.name}"

        # Init parent class
        super().__init__(head.tags | body.tags, name, body.enforce, body.rescale_factor)

        self.head = head
        self.body = body

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Check whether the implication constraint is satisfied.

        Evaluates the `head` and `body` constraints. The `body` constraint
        is enforced only if the `head` constraint is satisfied. If the
        `head` constraint is not satisfied, the `body` constraint does not
        affect the result.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            tuple[Tensor, Tensor]:
                - result: Tensor indicating satisfaction of the implication
                constraint (1 if satisfied, 0 otherwise).
                - head_satisfaction: Tensor indicating satisfaction of the
                head constraint alone.
        """
        # Check satisfaction of head and body constraints
        head_satisfaction, _ = self.head.check_constraint(data)
        body_satisfaction, _ = self.body.check_constraint(data)

        # If head constraint is satisfied (returning 1),
        # the body constraint matters (and should return 0/1 based on body)
        # If head constraint is not satisfied (returning 0),
        # the body constraint does not apply (and should return 1)
        result = logical_or(logical_not(head_satisfaction), body_satisfaction).float()

        return result, head_satisfaction

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute adjustment directions for tags to satisfy the constraint.

        Uses the `body` constraint directions as the update vector. Only
        applies updates if the `head` constraint is satisfied. Currently,
        this method only works for dense layers due to tag-to-index
        translation limitations.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            dict[str, Tensor]: Dictionary mapping tags to tensors
                specifying the adjustment direction for each tag.
        """
        # NOTE currently only works for dense layers
        # due to tag to index translation

        # Use directions of constraint body as update vector
        return self.body.calculate_direction(data)


class ScalarConstraint(Constraint):
    """A constraint that enforces scalar-based comparisons on a specific tag.

    This class ensures that the output of a specified tag satisfies a scalar
    comparison operation (e.g., less than, greater than, etc.). It uses a
    comparator function to validate the condition and calculates adjustment
    directions accordingly.

    Args:
        operand (Union[str, Transformation]): Name of the tag or a
            transformation to apply.
        comparator (Literal[">", "<", ">=", "<="]): Comparison operator used in the constraint.
        scalar (Number): The scalar value to compare against.
        name (str, optional): A unique name for the constraint. If not
            provided, a name is auto-generated in the format
            "<tag> <comparator> <scalar>".
        enforce (bool, optional): If False, only monitor the constraint
            without adjusting the loss. Defaults to True.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5.

    Raises:
        TypeError: If a provided attribute has an incompatible type.

    Notes:
        - The `tag` must be defined in the `descriptor` mapping.
        - The constraint name is composed using the tag, comparator, and scalar value.

    """

    def __init__(
        self,
        operand: str | Transformation,
        comparator: Literal[">", "<", ">=", "<="],
        scalar: Number,
        name: str = None,
        enforce: bool = True,
        rescale_factor: Number = 1.5,
    ) -> None:
        """Initializes a ScalarConstraint instance.

        Args:
            operand (Union[str, Transformation]): Function that needs to be
                performed on the network variables before applying the
                constraint.
            comparator (Literal[">", "<", ">=", "<="]): Comparison operator used in the constraint.
            scalar (Number): Constant to compare the variable to.
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

        Notes:
            - The `tag` must be defined in the `descriptor` mapping.
            - The constraint name is composed using the tag, comparator, and scalar value.
        """
        # Type checking
        validate_type("operand", operand, (str, Transformation))
        validate_comparator("comparator", comparator, COMPARATOR_MAP)
        validate_type("scalar", scalar, Number)

        # If transformation is provided, get tag name, else use IdentityTransformation
        if isinstance(operand, Transformation):
            tag = operand.tag
            transformation = operand
        else:
            tag = operand
            transformation = IdentityTransformation(tag)

        # Compose constraint name
        name = f"{tag} {comparator} {str(scalar)}"

        # Init parent class
        super().__init__({tag}, name, enforce, rescale_factor)

        # Init variables
        self.tag = tag
        self.scalar = scalar
        self.transformation = transformation

        # Calculate directions based on constraint operator
        if comparator in ["<", "<="]:
            self.direction = 1
        elif comparator in [">", ">="]:
            self.direction = -1
        self.comparator = COMPARATOR_MAP[comparator]

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Check if the scalar constraint is satisfied for a given tag.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            tuple[Tensor, Tensor]:
                - result: Tensor indicating whether the tag satisfies the constraint.
                - ones_like(result): Tensor of ones with same shape as `result`.
        """
        # Select relevant columns
        selection = self.descriptor.select(self.tag, data)

        # Apply transformation
        selection = self.transformation(selection)

        # Calculate current constraint result
        result = self.comparator(selection, self.scalar).float()
        return result, ones_like(result)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute adjustment directions to satisfy the scalar constraint.

        Only works for dense layers due to tag-to-index translation.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            dict[str, Tensor]: Dictionary mapping layers to tensors specifying
                the adjustment direction for each tag.
        """
        # NOTE currently only works for dense layers due
        # to tag to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(data[layer][0], device=self.device)

        layer, index = self.descriptor.location(self.tag)
        output[layer][index] = self.direction

        for layer in self.layers:
            output[layer] = normalize(reshape(output[layer], [1, -1]), dim=1)

        return output


class BinaryConstraint(Constraint):
    """A constraint that enforces a binary comparison between two tags.

    This class ensures that the output of one tag satisfies a comparison
    operation with the output of another tag (e.g., less than, greater than, etc.).
    It uses a comparator function to validate the condition and calculates adjustment directions accordingly.

    Args:
        operand_left (Union[str, Transformation]): Name of the left
            tag or a transformation to apply.
        comparator (Literal[">", "<", ">=", "<="]): Comparison operator used in the constraint.
        operand_right (Union[str, Transformation]): Name of the right
            tag or a transformation to apply.
        name (str, optional): A unique name for the constraint. If not
            provided, a name is auto-generated in the format
            "<operand_left> <comparator> <operand_right>".
        enforce (bool, optional): If False, only monitor the constraint
            without adjusting the loss. Defaults to True.
        rescale_factor (Number, optional): Factor to scale the
            constraint-adjusted loss. Defaults to 1.5.

    Raises:
        TypeError: If a provided attribute has an incompatible type.

    Notes:
        - The tags must be defined in the `descriptor` mapping.
        - The constraint name is composed using the left tag, comparator, and right tag.

    """

    def __init__(
        self,
        operand_left: str | Transformation,
        comparator: Literal[">", "<", ">=", "<="],
        operand_right: str | Transformation,
        name: str = None,
        enforce: bool = True,
        rescale_factor: Number = 1.5,
    ) -> None:
        """Initializes a BinaryConstraint instance.

        Args:
            operand_left (Union[str, Transformation]): Name of the left
                tag or a transformation to apply.
            comparator (Literal[">", "<", ">=", "<="]): Comparison operator used in the constraint.
            operand_right (Union[str, Transformation]): Name of the right
                tag or a transformation to apply.
            name (str, optional): A unique name for the constraint. If not
                provided, a name is auto-generated in the format
                "<operand_left> <comparator> <operand_right>".
            enforce (bool, optional): If False, only monitor the constraint
                without adjusting the loss. Defaults to True.
            rescale_factor (Number, optional): Factor to scale the
                constraint-adjusted loss. Defaults to 1.5.

        Raises:
            TypeError: If a provided attribute has an incompatible type.

        Notes:
            - The tags must be defined in the `descriptor` mapping.
            - The constraint name is composed using the left tag,
              comparator, and right tag.
        """
        # Type checking
        validate_type("operand_left", operand_left, (str, Transformation))
        validate_comparator("comparator", comparator, COMPARATOR_MAP)
        validate_type("operand_right", operand_right, (str, Transformation))

        # If transformation is provided, get tag name, else use IdentityTransformation
        if isinstance(operand_left, Transformation):
            tag_left = operand_left.tag
            transformation_left = operand_left
        else:
            tag_left = operand_left
            transformation_left = IdentityTransformation(tag_left)

        if isinstance(operand_right, Transformation):
            tag_right = operand_right.tag
            transformation_right = operand_right
        else:
            tag_right = operand_right
            transformation_right = IdentityTransformation(tag_right)

        # Compose constraint name
        name = f"{tag_left} {comparator} {tag_right}"

        # Init parent class
        super().__init__({tag_left, tag_right}, name, enforce, rescale_factor)

        # Init variables
        self.tag_left = tag_left
        self.tag_right = tag_right
        self.transformation_left = transformation_left
        self.transformation_right = transformation_right

        # Calculate directions based on constraint operator
        if comparator in ["<", "<="]:
            self.direction_left = 1
            self.direction_right = -1
        elif comparator in [">", ">="]:
            self.direction_left = -1
            self.direction_right = 1
        self.comparator = COMPARATOR_MAP[comparator]

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluate whether the binary constraint is satisfied for the current predictions.

        The constraint compares the outputs of two tags using the specified
        comparator function. A result of `1` indicates the constraint is satisfied
        for a sample, and `0` indicates it is violated.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            tuple[Tensor, Tensor]:
                - result (Tensor): Binary tensor indicating constraint satisfaction
                (1 for satisfied, 0 for violated) for each sample.
                - mask (Tensor): Tensor of ones with the same shape as `result`,
                used for constraint aggregation.
        """
        # Select relevant columns
        selection_left = self.descriptor.select(self.tag_left, data)
        selection_right = self.descriptor.select(self.tag_right, data)

        # Apply transformations
        selection_left = self.transformation_left(selection_left)
        selection_right = self.transformation_right(selection_right)

        result = self.comparator(selection_left, selection_right).float()

        return result, ones_like(result)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute adjustment directions for the tags involved in the binary constraint.

        The returned directions indicate how to adjust each tag's output to
        satisfy the constraint. Only currently supported for dense layers.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            dict[str, Tensor]: A mapping from layer names to tensors specifying
            the normalized adjustment directions for each tag involved in the
            constraint.
        """
        # NOTE currently only works for dense layers due
        # to tag to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(data[layer][0], device=self.device)

        layer_left, index_left = self.descriptor.location(self.tag_left)
        layer_right, index_right = self.descriptor.location(self.tag_right)
        output[layer_left][index_left] = self.direction_left
        output[layer_right][index_right] = self.direction_right

        for layer in self.layers:
            output[layer] = normalize(reshape(output[layer], [1, -1]), dim=1)

        return output


class SumConstraint(Constraint):
    """A constraint that enforces a weighted summation comparison between two groups of tags.

    This class evaluates whether the weighted sum of outputs from one set of
    tags satisfies a comparison operation with the weighted sum of
    outputs from another set of tags.
    """

    def __init__(
        self,
        operands_left: list[str | Transformation],
        comparator: Literal[">", "<", ">=", "<="],
        operands_right: list[str | Transformation],
        weights_left: list[Number] = None,
        weights_right: list[Number] = None,
        name: str = None,
        enforce: bool = True,
        rescale_factor: Number = 1.5,
    ) -> None:
        """Initializes the SumConstraint.

        Args:
            operands_left (list[Union[str, Transformation]]): List of tags
                or transformations on the left side.
            comparator (Literal[">", "<", ">=", "<="]): Comparison operator used in the constraint.
            operands_right (list[Union[str, Transformation]]): List of tags
                or transformations on the right side.
            weights_left (list[Number], optional): Weights for the left
                tags. Defaults to None.
            weights_right (list[Number], optional): Weights for the right
                tags. Defaults to None.
            name (str, optional): Unique name for the constraint.
                If None, it's auto-generated. Defaults to None.
            enforce (bool, optional): If False, only monitor the constraint
                without adjusting the loss. Defaults to True.
            rescale_factor (Number, optional): Factor to scale the
                constraint-adjusted loss. Defaults to 1.5.

        Raises:
            TypeError: If a provided attribute has an incompatible type.
            ValueError: If the dimensions of tags and weights mismatch.
        """
        # Type checking
        validate_iterable("operands_left", operands_left, (str, Transformation))
        validate_comparator("comparator", comparator, COMPARATOR_MAP)
        validate_iterable("operands_right", operands_right, (str, Transformation))
        validate_iterable("weights_left", weights_left, Number, allow_none=True)
        validate_iterable("weights_right", weights_right, Number, allow_none=True)

        # If transformation is provided, get tag, else use IdentityTransformation
        tags_left: list[str] = []
        transformations_left: list[Transformation] = []
        for operand_left in operands_left:
            if isinstance(operand_left, Transformation):
                tag_left = operand_left.tag
                tags_left.append(tag_left)
                transformations_left.append(operand_left)
            else:
                tag_left = operand_left
                tags_left.append(tag_left)
                transformations_left.append(IdentityTransformation(tag_left))

        tags_right: list[str] = []
        transformations_right: list[Transformation] = []
        for operand_right in operands_right:
            if isinstance(operand_right, Transformation):
                tag_right = operand_right.tag
                tags_right.append(tag_right)
                transformations_right.append(operand_right)
            else:
                tag_right = operand_right
                tags_right.append(tag_right)
                transformations_right.append(IdentityTransformation(tag_right))

        # Compose constraint name
        w_left = weights_left or [""] * len(tags_left)
        w_right = weights_right or [""] * len(tags_right)
        left_expr = " + ".join(f"{w}{n}" for w, n in zip(w_left, tags_left, strict=False))
        right_expr = " + ".join(f"{w}{n}" for w, n in zip(w_right, tags_right, strict=False))
        name = f"{left_expr} {comparator} {right_expr}"

        # Init parent class
        tags = set(tags_left) | set(tags_right)
        super().__init__(tags, name, enforce, rescale_factor)

        # Init variables
        self.tags_left = tags_left
        self.tags_right = tags_right
        self.transformations_left = transformations_left
        self.transformations_right = transformations_right

        # If feature list dimensions don't match weight list dimensions, raise error
        if weights_left and (len(tags_left) != len(weights_left)):
            raise ValueError(
                "The dimensions of tags_left don't match with the dimensions of weights_left."
            )
        if weights_right and (len(tags_right) != len(weights_right)):
            raise ValueError(
                "The dimensions of tags_right don't match with the dimensions of weights_right."
            )

        # If weights are provided for summation, transform them to Tensors
        if weights_left:
            self.weights_left = tensor(weights_left, device=self.device)
        else:
            self.weights_left = ones(len(tags_left), device=self.device)
        if weights_right:
            self.weights_right = tensor(weights_right, device=self.device)
        else:
            self.weights_right = ones(len(tags_right), device=self.device)

        # Calculate directions based on constraint operator
        if comparator in ["<", "<="]:
            self.direction_left = -1
            self.direction_right = 1
        elif comparator in [">", ">="]:
            self.direction_left = 1
            self.direction_right = -1
        self.comparator = COMPARATOR_MAP[comparator]

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluate whether the weighted sum constraint is satisfied.

        Computes the weighted sum of outputs from the left and right tags,
        applies the specified comparator function, and returns a binary result for
        each sample.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            tuple[Tensor, Tensor]:
                - result (Tensor): Binary tensor indicating whether the constraint
                is satisfied (1) or violated (0) for each sample.
                - mask (Tensor): Tensor of ones, used for constraint aggregation.
        """

        def compute_weighted_sum(
            tags: list[str],
            transformations: list[Transformation],
            weights: Tensor,
        ) -> Tensor:
            # Select relevant columns
            selections = [self.descriptor.select(tag, data) for tag in tags]

            # Apply transformations
            results = []
            for transformation, selection in zip(transformations, selections, strict=False):
                results.append(transformation(selection))

            # Extract predictions for all tags and apply weights in bulk
            predictions = stack(results)

            # Calculate weighted sum
            return (predictions * weights.view(-1, 1, 1)).sum(dim=0)

        # Compute weighted sums
        weighted_sum_left = compute_weighted_sum(
            self.tags_left, self.transformations_left, self.weights_left
        )
        weighted_sum_right = compute_weighted_sum(
            self.tags_right, self.transformations_right, self.weights_right
        )

        # Apply the comparator and calculate the result
        result = self.comparator(weighted_sum_left, weighted_sum_right).float()

        return result, ones_like(result)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute adjustment directions for tags involved in the weighted sum constraint.

        The directions indicate how to adjust each tag's output to satisfy the
        constraint. Only dense layers are currently supported.

        Args:
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            dict[str, Tensor]: Mapping from layer names to normalized tensors
            specifying adjustment directions for each tag involved in the constraint.
        """
        # NOTE currently only works for dense layers
        # due to tag to index translation

        output = {}

        for layer in self.layers:
            output[layer] = zeros_like(data[layer][0], device=self.device)

        for tag_left in self.tags_left:
            layer, index = self.descriptor.location(tag_left)
            output[layer][index] = self.direction_left

        for tag_right in self.tags_right:
            layer, index = self.descriptor.location(tag_right)
            output[layer][index] = self.direction_right

        for layer in self.layers:
            output[layer] = normalize(reshape(output[layer], [1, -1]), dim=1)

        return output


class RankedMonotonicityConstraint(MonotonicityConstraint):
    """Constraint that enforces a monotonic relationship between two tags.

    This constraint ensures that the activations of a prediction tag (`tag_prediction`)
    are monotonically ascending or descending with respect to a target tag (`tag_reference`).
    """

    def __init__(
        self,
        tag_prediction: str,
        tag_reference: str,
        rescale_factor_lower: float = 1.5,
        rescale_factor_upper: float = 1.75,
        stable: bool = True,
        direction: Literal["ascending", "descending"] = "ascending",
        name: str = None,
        enforce: bool = True,
    ):
        """Constraint that enforces monotonicity on a predicted output.

        This constraint ensures that the activations of a prediction tag (`tag_prediction`)
        are monotonically ascending or descending with respect to a target tag (`tag_reference`).

        Args:
            tag_prediction (str): Name of the tag whose activations should follow the monotonic relationship.
            tag_reference (str): Name of the tag that acts as the monotonic reference.
            rescale_factor_lower (float, optional): Lower bound for rescaling rank differences. Defaults to 1.5.
            rescale_factor_upper (float, optional): Upper bound for rescaling rank differences. Defaults to 1.75.
            stable (bool, optional): Whether to use stable sorting when ranking. Defaults to True.
            direction (str, optional): Direction of monotonicity to enforce, either 'ascending' or 'descending'. Defaults to 'ascending'.
            name (str, optional): Custom name for the constraint. If None, a descriptive name is auto-generated.
            enforce (bool, optional): If False, the constraint is only monitored (not enforced). Defaults to True.
        """
        # Compose constraint name
        if name is None:
            name = f"{tag_prediction} monotonically (ranked) {direction} by {tag_reference}"

        # Init parent class
        super().__init__(
            tag_prediction=tag_prediction,
            tag_reference=tag_reference,
            rescale_factor_lower=rescale_factor_lower,
            rescale_factor_upper=rescale_factor_upper,
            stable=stable,
            direction=direction,
            name=name,
            enforce=enforce,
        )

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluate whether the monotonicity constraint is satisfied."""
        # Select relevant columns
        preds = self.descriptor.select(self.tag_prediction, data)
        targets = self.descriptor.select(self.tag_reference, data)

        # Utility: convert values -> ranks (0 ... num_features-1)
        def compute_ranks(x: Tensor, descending: bool) -> Tensor:
            return argsort(
                argsort(x, descending=descending, stable=self.stable, dim=0),
                descending=False,
                stable=self.stable,
                dim=0,
            )

        # Compute predicted and target ranks
        pred_ranks = compute_ranks(preds, descending=self.descending)
        target_ranks = compute_ranks(targets, descending=False)

        # Rank difference
        rank_diff = pred_ranks - target_ranks

        # Rescale differences into [rescale_factor_lower, rescale_factor_upper]
        batch_size = preds.shape[0]
        invert_direction = -1 if self.descending else 1
        self.compared_rankings = (
            (rank_diff / batch_size) * (self.rescale_factor_upper - self.rescale_factor_lower)
            + self.rescale_factor_lower * sign(rank_diff)
        ) * invert_direction

        # Calculate satisfaction
        incorrect_rankings = eq(self.compared_rankings, 0).float()

        return incorrect_rankings, ones_like(incorrect_rankings)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Calculates ranking adjustments for monotonicity enforcement."""
        layer, _ = self.descriptor.location(self.tag_prediction)
        return {layer: self.compared_rankings}


class PairwiseMonotonicityConstraint(MonotonicityConstraint):
    """Constraint that enforces a monotonic relationship between two tags.

    This constraint ensures that the activations of a prediction tag (`tag_prediction`)
    are monotonically ascending or descending with respect to a target tag (`tag_reference`).
    """

    def __init__(
        self,
        tag_prediction: str,
        tag_reference: str,
        rescale_factor_lower: float = 1.5,
        rescale_factor_upper: float = 1.75,
        stable: bool = True,
        direction: Literal["ascending", "descending"] = "ascending",
        name: str = None,
        enforce: bool = True,
    ):
        """Constraint that enforces monotonicity on a predicted output.

        This constraint ensures that the activations of a prediction tag (`tag_prediction`)
        are monotonically ascending or descending with respect to a target tag (`tag_reference`).

        Args:
            tag_prediction (str): Name of the tag whose activations should follow the monotonic relationship.
            tag_reference (str): Name of the tag that acts as the monotonic reference.
            rescale_factor_lower (float, optional): Lower bound for rescaling rank differences. Defaults to 1.5.
            rescale_factor_upper (float, optional): Upper bound for rescaling rank differences. Defaults to 1.75.
            stable (bool, optional): Whether to use stable sorting when ranking. Defaults to True.
            direction (str, optional): Direction of monotonicity to enforce, either 'ascending' or 'descending'. Defaults to 'ascending'.
            name (str, optional): Custom name for the constraint. If None, a descriptive name is auto-generated.
            enforce (bool, optional): If False, the constraint is only monitored (not enforced). Defaults to True.
        """
        # Compose constraint name
        if name is None:
            name = f"{tag_prediction} monotonically (pairwise) {direction} by {tag_reference}"

        # Init parent class
        super().__init__(
            tag_prediction=tag_prediction,
            tag_reference=tag_reference,
            rescale_factor_lower=rescale_factor_lower,
            rescale_factor_upper=rescale_factor_upper,
            stable=stable,
            direction=direction,
            name=name,
            enforce=enforce,
        )

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluate whether the monotonicity constraint is satisfied."""
        # Select relevant columns
        preds = self.descriptor.select(self.tag_prediction, data).squeeze(1)
        targets = self.descriptor.select(self.tag_reference, data).squeeze(1)

        # Sort targets back to original order
        sorted_targets, idx = sort(targets, stable=self.stable, dim=0, descending=self.descending)
        sorted_preds = preds[idx]

        # Pairwise differences
        preds_diff = sorted_preds.unsqueeze(1) - sorted_preds.unsqueeze(0)
        targets_diff = sorted_targets.unsqueeze(1) - sorted_targets.unsqueeze(0)

        # Consider only upper triangle to avoid duplicate comparisons
        batch_size = preds.shape[0]
        mask = triu(ones(batch_size, batch_size, dtype=torch.bool), diagonal=1)

        # Pairwise violations
        violations = (preds_diff * targets_diff < 0) & mask

        # Rank difference
        sorted_rank_diff = violations.float().sum(dim=1) - violations.float().sum(dim=0)

        # Undo sorting to match original order
        rank_diff = zeros_like(sorted_rank_diff)
        rank_diff[idx] = sorted_rank_diff
        rank_diff = rank_diff.unsqueeze(1)

        # Rescale differences into [rescale_factor_lower, rescale_factor_upper]
        invert_direction = -1 if self.descending else 1
        self.compared_rankings = (
            (rank_diff / batch_size) * (self.rescale_factor_upper - self.rescale_factor_lower)
            + self.rescale_factor_lower * sign(rank_diff)
        ) * invert_direction

        # Calculate satisfaction
        incorrect_rankings = eq(self.compared_rankings, 0).float()

        return incorrect_rankings, ones_like(incorrect_rankings)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Calculates ranking adjustments for monotonicity enforcement."""
        layer, _ = self.descriptor.location(self.tag_prediction)
        return {layer: self.compared_rankings}


class PerGroupMonotonicityConstraint(Constraint):
    """Applies a monotonicity constraint independently per group of samples.

    This class wraps an existing `MonotonicityConstraint` instance (`base`) and
    enforces it **separately** for each unique group defined by `tag_group`.

    Each group is treated as an independent mini-batch:
    - The base constraint is applied to the group's subset of data.
    - Violations and directions are computed per group and then reassembled
      into the original batch order.

    This is an explicit alternative to :class:`EncodedGroupedMonotonicityConstraint`,
    which enforces the same logic using vectorized interval encoding.
    """

    def __init__(
        self,
        base: MonotonicityConstraint,
        tag_group: str,
        name: str | None = None,
    ):
        """Initializes the per-group monotonicity constraint.

        Args:
            base (MonotonicityConstraint): A pre-configured monotonicity constraint to apply per group.
            tag_group (str): Column/tag name that identifies groups.
            name (str | None, optional): Custom name for the constraint. Defaults to
                f"{base.tag_prediction} for each {tag_group} ...".
        """
        # Compose constraint name
        if name is None:
            name = base.name.replace(
                base.tag_prediction, f"{base.tag_prediction} for each {tag_group}", 1
            )

        # Init parent class
        super().__init__(base.tags, name, base.enforce, base.rescale_factor)

        # Init variables
        self.base = base
        self.tag_group = tag_group

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluate whether the monotonicity constraint is satisfied per group."""
        # Select group identifiers and convert to unique list
        group_identifiers = self.descriptor.select(self.tag_group, data)
        unique_group_identifiers = unique(group_identifiers, sorted=False).tolist()

        # Initialize checks and directions
        checks = zeros_like(group_identifiers, device=self.device)
        self.directions = zeros_like(group_identifiers, device=self.device)

        # Get prediction and target keys
        preds_key, _ = self.descriptor.location(self.base.tag_prediction)
        targets_key, _ = self.descriptor.location(self.base.tag_reference)

        for group_identifier in unique_group_identifiers:
            # Create mask for the samples in this group
            group_mask = (group_identifiers == group_identifier).squeeze(1)

            # Create mini-batch for the group
            group_data = {
                preds_key: data[preds_key][group_mask],
                targets_key: data[targets_key][group_mask],
            }

            # Call super on the mini-batch
            checks[group_mask], _ = self.base.check_constraint(group_data)
            self.directions[group_mask] = self.base.compared_rankings

        return checks, ones_like(checks)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Calculates ranking adjustments for monotonicity enforcement."""
        layer, _ = self.descriptor.location(self.base.tag_prediction)
        return {layer: self.directions}


class EncodedGroupedMonotonicityConstraint(Constraint):
    """Applies a base monotonicity constraint to groups via interval encoding.

    This constraint enforces a monotonic relationship between a prediction tag
    (`base.tag_prediction`) and a reference tag (`base.tag_reference`) for
    each group defined by `tag_group`.

    Instead of looping over groups, each group's predictions and targets are
    shifted by a large offset to place them in **non-overlapping intervals**.
    This allows the base constraint to be applied in a single, vectorized
    operation.

    This is a vectorized alternative to :class:`PerGroupMonotonicityConstraint`,
    which enforces the same logic via explicit per-group iteration.
    """

    def __init__(
        self,
        base: MonotonicityConstraint,
        tag_group: str,
        name: str | None = None,
    ):
        """Initializes the encoded grouped monotonicity constraint.

        Args:
            base (MonotonicityConstraint): The base constraint to apply to each group.
            tag_group (str): Column/tag name identifying groups in the batch.
            name (str | None, optional): Optional custom name for this constraint.
                If None, generates a descriptive name based on `base.name`.
        """
        # Compose constraint name
        if name is None:
            name = base.name.replace(
                base.tag_prediction,
                f"{base.tag_prediction} for each {tag_group} (encoded)",
                1,
            )

        # Init parent class
        super().__init__(base.tags, name, base.enforce, base.rescale_factor)

        # Init variables
        self.base = base
        self.tag_group = tag_group

        # Initialize negation factor based on direction
        self.negation = 1 if self.base.direction == "ascending" else -1

    def check_constraint(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Evaluate whether the monotonicity constraint is satisfied by mapping each group on a non-overlapping interval."""
        # Get data and keys
        ids = self.descriptor.select(self.tag_group, data)
        preds = self.descriptor.select(self.base.tag_prediction, data)
        targets = self.descriptor.select(self.base.tag_reference, data)
        preds_key, _ = self.descriptor.location(self.base.tag_prediction)
        targets_key, _ = self.descriptor.location(self.base.tag_reference)

        new_preds = preds + ids * (preds.max() - preds.min() + 1)
        new_targets = targets + self.negation * ids * (targets.max() - targets.min() + 1)

        # Create new batch for child constraint
        new_data = {preds_key: new_preds, targets_key: new_targets}

        # Call super on the adjusted batch
        checks, _ = self.base.check_constraint(new_data)
        self.directions = self.base.compared_rankings

        return checks, ones_like(checks)

    def calculate_direction(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Calculates ranking adjustments for monotonicity enforcement."""
        layer, _ = self.descriptor.location(self.base.tag_prediction)
        return {layer: self.directions}


class ANDConstraint(Constraint):
    """A composite constraint that enforces the logical AND of multiple constraints.

    This class combines multiple sub-constraints and evaluates them jointly:

    * The satisfaction of the AND constraint is `True` only if all sub-constraints
    are satisfied (elementwise logical AND).
    * The corrective direction is computed by weighting each sub-constraint's
    direction with its satisfaction mask and summing across all sub-constraints.
    """

    def __init__(
        self,
        *constraints: Constraint,
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """A composite constraint that enforces the logical AND of multiple constraints.

        This class combines multiple sub-constraints and evaluates them jointly:

        * The satisfaction of the AND constraint is `True` only if all sub-constraints
        are satisfied (elementwise logical AND).
        * The corrective direction is computed by weighting each sub-constraint's
        direction with its satisfaction mask and summing across all sub-constraints.

        Args:
            *constraints (Constraint): One or more `Constraint` instances to be combined.
            name (str, optional): A custom name for this constraint. If not provided,
                the name will be composed from the sub-constraint names joined with
                " AND ".
            monitor_only (bool, optional): If True, the constraint will be monitored
                but not enforced. Defaults to False.
            rescale_factor (Number, optional): A scaling factor applied when rescaling
                corrections. Defaults to 1.5.

        Attributes:
            constraints (tuple[Constraint, ...]): The sub-constraints being combined.
            neurons (set): The union of neurons referenced by the sub-constraints.
            name (str): The name of the constraint (composed or custom).
        """
        # Type checking
        validate_iterable("constraints", constraints, Constraint)

        # Compose constraint name
        if not name:
            name = " AND ".join([constraint.name for constraint in constraints])

        # Init parent class
        super().__init__(
            set().union(*(constraint.tags for constraint in constraints)),
            name,
            monitor_only,
            rescale_factor,
        )

        # Init variables
        self.constraints = constraints

    def check_constraint(self, data: dict[str, Tensor]):
        """Evaluate whether all sub-constraints are satisfied.

        Args:
            data: Model predictions and associated batch/context information.

        Returns:
            tuple[Tensor, Tensor]: A tuple `(total_satisfaction, mask)` where:
                * `total_satisfaction`: A boolean or numeric tensor indicating
                elementwise whether all constraints are satisfied
                (logical AND).
                * `mask`: A tensor of ones with the same shape as
                `total_satisfaction`. Typically used as a weighting mask
                in downstream processing.
        """
        total_satisfaction: Tensor = None
        total_mask: Tensor = None

        # TODO vectorize this loop
        for constraint in self.constraints:
            satisfaction, mask = constraint.check_constraint(data)
            if total_satisfaction is None:
                total_satisfaction = satisfaction
                total_mask = mask
            else:
                total_satisfaction = logical_and(total_satisfaction, satisfaction)
                total_mask = logical_or(total_mask, mask)

        return total_satisfaction.float(), total_mask.float()

    def calculate_direction(self, data: dict[str, Tensor]):
        """Compute the corrective direction by aggregating sub-constraint directions.

        Each sub-constraint contributes its corrective direction, weighted
        by its satisfaction mask. The directions are summed across constraints
        for each affected layer.

        Args:
            data: Model predictions and associated batch/context information.

        Returns:
            dict[str, Tensor]: A mapping from layer identifiers to correction
            tensors. Each entry represents the aggregated correction to apply
            to that layer, based on the satisfaction-weighted sum of
            sub-constraint directions.
        """
        total_direction: dict[str, Tensor] = {}

        # TODO vectorize this loop
        for constraint in self.constraints:
            # TODO improve efficiency by avoiding double computation?
            satisfaction, _ = constraint.check_constraint(data)
            direction = constraint.calculate_direction(data)

            for layer, dir in direction.items():
                if layer not in total_direction:
                    total_direction[layer] = satisfaction * dir
                else:
                    total_direction[layer] += satisfaction * dir

        return total_direction


class ORConstraint(Constraint):
    """A composite constraint that enforces the logical OR of multiple constraints.

    This class combines multiple sub-constraints and evaluates them jointly:

    * The satisfaction of the OR constraint is `True` if at least one sub-constraint
    is satisfied (elementwise logical OR).
    * The corrective direction is computed by weighting each sub-constraint's
    direction with its satisfaction mask and summing across all sub-constraints.
    """

    def __init__(
        self,
        *constraints: Constraint,
        name: str = None,
        monitor_only: bool = False,
        rescale_factor: Number = 1.5,
    ) -> None:
        """A composite constraint that enforces the logical OR of multiple constraints.

        This class combines multiple sub-constraints and evaluates them jointly:

        * The satisfaction of the OR constraint is `True` if at least one sub-constraint
        is satisfied (elementwise logical OR).
        * The corrective direction is computed by weighting each sub-constraint's
        direction with its satisfaction mask and summing across all sub-constraints.

        Args:
            *constraints (Constraint): One or more `Constraint` instances to be combined.
            name (str, optional): A custom name for this constraint. If not provided,
                the name will be composed from the sub-constraint names joined with
                " OR ".
            monitor_only (bool, optional): If True, the constraint will be monitored
                but not enforced. Defaults to False.
            rescale_factor (Number, optional): A scaling factor applied when rescaling
                corrections. Defaults to 1.5.

        Attributes:
            constraints (tuple[Constraint, ...]): The sub-constraints being combined.
            neurons (set): The union of neurons referenced by the sub-constraints.
            name (str): The name of the constraint (composed or custom).
        """
        # Type checking
        validate_iterable("constraints", constraints, Constraint)

        # Compose constraint name
        if not name:
            name = " OR ".join([constraint.name for constraint in constraints])

        # Init parent class
        super().__init__(
            set().union(*(constraint.tags for constraint in constraints)),
            name,
            monitor_only,
            rescale_factor,
        )

        # Init variables
        self.constraints = constraints

    def check_constraint(self, data: dict[str, Tensor]):
        """Evaluate whether any sub-constraints are satisfied.

        Args:
            data: Model predictions and associated batch/context information.

        Returns:
            tuple[Tensor, Tensor]: A tuple `(total_satisfaction, mask)` where:
                * `total_satisfaction`: A boolean or numeric tensor indicating
                elementwise whether any constraints are satisfied
                (logical OR).
                * `mask`: A tensor of ones with the same shape as
                `total_satisfaction`. Typically used as a weighting mask
                in downstream processing.
        """
        total_satisfaction: Tensor = None
        total_mask: Tensor = None

        # TODO vectorize this loop
        for constraint in self.constraints:
            satisfaction, mask = constraint.check_constraint(data)
            if total_satisfaction is None:
                total_satisfaction = satisfaction
                total_mask = mask
            else:
                total_satisfaction = logical_or(total_satisfaction, satisfaction)
                total_mask = logical_or(total_mask, mask)

        return total_satisfaction.float(), total_mask.float()

    def calculate_direction(self, data: dict[str, Tensor]):
        """Compute the corrective direction by aggregating sub-constraint directions.

        Each sub-constraint contributes its corrective direction, weighted
        by its satisfaction mask. The directions are summed across constraints
        for each affected layer.

        Args:
            data: Model predictions and associated batch/context information.

        Returns:
            dict[str, Tensor]: A mapping from layer identifiers to correction
            tensors. Each entry represents the aggregated correction to apply
            to that layer, based on the satisfaction-weighted sum of
            sub-constraint directions.
        """
        total_direction: dict[str, Tensor] = {}

        # TODO vectorize this loop
        for constraint in self.constraints:
            # TODO improve efficiency by avoiding double computation?
            satisfaction, _ = constraint.check_constraint(data)
            direction = constraint.calculate_direction(data)

            for layer, dir in direction.items():
                if layer not in total_direction:
                    total_direction[layer] = satisfaction * dir
                else:
                    total_direction[layer] += satisfaction * dir

        return total_direction
