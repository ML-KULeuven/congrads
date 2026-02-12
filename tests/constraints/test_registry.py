import inspect

import pytest
import torch
from torch import Tensor

from congrads.constraints import registry
from congrads.descriptor import Descriptor


# --- Fixtures --- #
@pytest.fixture
def mock_data():
    return {
        "input": torch.tensor([[0.5, 1.5], [2.0, 0.5]]),
        "output": torch.tensor([[1.0, 0.2], [0.0, 2.0]]),
        "context": torch.tensor([[0], [1]]),
    }


@pytest.fixture(autouse=True)
def patch_descriptor():
    mock_descriptor = Descriptor()
    mock_descriptor.add_layer("input", constant=True)
    mock_descriptor.add_tag("a", "input", 0)
    mock_descriptor.add_tag("b", "input", 1)
    mock_descriptor.add_layer("output")
    mock_descriptor.add_tag("c", "output", 0)
    mock_descriptor.add_tag("d", "output", 1)
    mock_descriptor.add_layer("context", constant=True)
    mock_descriptor.add_tag("id", "context", 0)

    registry.Constraint.descriptor = mock_descriptor
    registry.Constraint.device = "cpu"


# --- Dynamic discovery of all constraint classes --- #
def get_all_constraint_classes():
    """Return all Constraint subclasses in the registry, except the base class."""
    return [
        cls
        for name, cls in inspect.getmembers(registry, inspect.isclass)
        if issubclass(cls, registry.Constraint) and cls is not registry.Constraint
    ]


# --- Default constructors for each constraint --- #
constraint_defaults = {
    "ScalarConstraint": lambda: registry.ScalarConstraint("a", "<", 1.0),
    "BinaryConstraint": lambda: registry.BinaryConstraint("a", ">", "b"),
    "ImplicationConstraint": lambda: registry.ImplicationConstraint(
        registry.ScalarConstraint("a", "<", 1.0),
        registry.BinaryConstraint("a", "<", "b"),
    ),
    "SumConstraint": lambda: registry.SumConstraint(
        ["a", "b"], "<=", ["c", "d"], weights_left=[1.0, 0.5], weights_right=[0.5, 1.0]
    ),
    "RankedMonotonicityConstraint": lambda: registry.RankedMonotonicityConstraint(
        "a", "b", direction="ascending"
    ),
    "PairwiseMonotonicityConstraint": lambda: registry.PairwiseMonotonicityConstraint(
        "a", "b", direction="ascending"
    ),
    "PerGroupMonotonicityConstraint": lambda: registry.PerGroupMonotonicityConstraint(
        registry.RankedMonotonicityConstraint("a", "b", direction="ascending"),
        tag_group="id",
    ),
    "EncodedGroupedMonotonicityConstraint": lambda: registry.EncodedGroupedMonotonicityConstraint(
        registry.RankedMonotonicityConstraint("a", "b", direction="ascending"),
        tag_group="id",
    ),
    "ORConstraint": lambda: registry.ORConstraint(
        registry.ScalarConstraint("a", "<=", 1.0),
        registry.BinaryConstraint("a", ">", "b"),
    ),
    "ANDConstraint": lambda: registry.ANDConstraint(
        registry.ScalarConstraint("a", "<", 1.0),
        registry.BinaryConstraint("a", ">=", "b"),
    ),
}


# --- Parameterized tests over all constraints --- #
@pytest.mark.parametrize("constraint_cls", get_all_constraint_classes())
def test_constraints_initialization(constraint_cls):
    """Test that all constraints initialize properly with expected attributes."""
    if constraint_cls.__name__ not in constraint_defaults:
        pytest.skip(f"No default constructor for {constraint_cls.__name__}, skipping test.")
    constraint = constraint_defaults[constraint_cls.__name__]()
    assert isinstance(constraint, registry.Constraint)
    assert isinstance(constraint.tags, set)
    assert isinstance(constraint.name, str)
    assert hasattr(constraint, "rescale_factor")


@pytest.mark.parametrize("constraint_cls", get_all_constraint_classes())
def test_constraints_check_constraint(constraint_cls, mock_data):
    """Test that check_constraint returns tensors of correct shape and type."""
    if constraint_cls.__name__ not in constraint_defaults:
        pytest.skip(f"No default constructor for {constraint_cls.__name__}, skipping test.")

    constraint = constraint_defaults[constraint_cls.__name__]()
    result, mask = constraint.check_constraint(mock_data)

    # Shape and type checks
    assert isinstance(result, Tensor)
    assert isinstance(mask, Tensor)
    batch_size = next(iter(mock_data.values())).shape[0]
    assert result.shape[0] == batch_size
    assert mask.shape[0] == batch_size

    assert torch.all((result == 0) | (result == 1)), (
        f"{constraint_cls.__name__} returned non-binary values"
    )


@pytest.mark.parametrize("constraint_cls", get_all_constraint_classes())
def test_constraints_calculate_direction(constraint_cls, mock_data):
    """Test that calculate_direction returns direction tensors correctly."""
    if constraint_cls.__name__ not in constraint_defaults:
        pytest.skip(f"No default constructor for {constraint_cls.__name__}, skipping test.")

    constraint = constraint_defaults[constraint_cls.__name__]()
    result, _ = constraint.check_constraint(mock_data)  # ensure directions are computed
    directions = constraint.calculate_direction(mock_data)

    assert isinstance(directions, dict)
    for layer, dir_tensor in directions.items():
        # shape check: directions are 2D
        assert dir_tensor.ndim == 2, f"{layer} direction tensor not 2D"

        # layer key exists in descriptor
        descriptor_layers = getattr(constraint.descriptor, "variable_layers", set()) | getattr(
            constraint.descriptor, "constant_layers", set()
        )
        assert layer in descriptor_layers, f"{layer} not in descriptor layers"


# --- Specific logic tests --- #
def test_implication_constraint_logic(mock_data):
    scalar = registry.ScalarConstraint("a", "<", 1.0)
    binary = registry.BinaryConstraint("a", "<", "b")
    implication = registry.ImplicationConstraint(scalar, binary)
    result, head_satisfaction = implication.check_constraint(mock_data)
    assert torch.all((result == 1) | (result == 0))


def test_sum_constraint_weight_mismatch():
    # Should raise ValueError if weights don't match number of tags
    with pytest.raises(ValueError):
        registry.SumConstraint(["a", "b"], "<", ["c", "d"], weights_left=[1.0])
