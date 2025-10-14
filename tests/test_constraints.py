import pytest
import torch

from congrads.constraints import (
    BinaryConstraint,
    Constraint,
    ImplicationConstraint,
    MonotonicityConstraint,
    ScalarConstraint,
    SumConstraint,
)
from congrads.descriptor import Descriptor


@pytest.fixture
def mock_data():
    return {
        "layer1": torch.tensor([[0.5, 1.5], [2.0, 0.5]]),
        "layer2": torch.tensor([[1.0, 0.2], [0.0, 2.0]]),
    }


@pytest.fixture(autouse=True)
def patch_descriptor():
    mock_descriptor = Descriptor()
    mock_descriptor.add("layer1", "a", 0)
    mock_descriptor.add("layer1", "b", 1)
    mock_descriptor.add("layer2", "c", 0)
    mock_descriptor.add("layer2", "d", 1)

    Constraint.descriptor = mock_descriptor
    Constraint.device = "cpu"


# --- Constraint instances fixtures --- #
@pytest.fixture
def scalar_constraint():
    return ScalarConstraint("a", torch.lt, 1.0)


@pytest.fixture
def binary_constraint():
    return BinaryConstraint("a", torch.lt, "b")


@pytest.fixture
def implication_constraint(scalar_constraint, binary_constraint):
    return ImplicationConstraint(scalar_constraint, binary_constraint)


@pytest.fixture
def sum_constraint():
    return SumConstraint(
        ["a", "b"], torch.lt, ["c", "d"], weights_left=[1.0, 0.5], weights_right=[0.5, 1.0]
    )


@pytest.fixture
def monotonicity_constraint():
    return MonotonicityConstraint("a", "b", direction="ascending")


# --- Parameterized tests --- #
@pytest.mark.parametrize(
    "constraint_fixture",
    [
        "scalar_constraint",
        "binary_constraint",
        "implication_constraint",
        "sum_constraint",
        "monotonicity_constraint",
    ],
)
def test_constraints_initialization(constraint_fixture, request):
    constraint = request.getfixturevalue(constraint_fixture)
    assert isinstance(constraint, Constraint)
    assert isinstance(constraint.tags, set)
    assert isinstance(constraint.name, str)
    assert hasattr(constraint, "rescale_factor")


@pytest.mark.parametrize(
    "constraint_fixture",
    [
        "scalar_constraint",
        "binary_constraint",
        "implication_constraint",
        "sum_constraint",
        "monotonicity_constraint",
    ],
)
def test_constraints_check_constraint(constraint_fixture, request, mock_data):
    constraint = request.getfixturevalue(constraint_fixture)
    result, mask = constraint.check_constraint(mock_data)
    assert isinstance(result, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    batch_size = next(iter(mock_data.values())).shape[0]
    assert result.shape[0] == batch_size
    assert mask.shape[0] == batch_size
    # Only for constraints returning binary satisfaction
    if isinstance(
        constraint, (ScalarConstraint, BinaryConstraint, ImplicationConstraint, SumConstraint)
    ):
        assert torch.all((result == 0) | (result == 1))


@pytest.mark.parametrize(
    "constraint_fixture",
    [
        "scalar_constraint",
        "binary_constraint",
        "implication_constraint",
        "sum_constraint",
        "monotonicity_constraint",
    ],
)
def test_constraints_calculate_direction(constraint_fixture, request, mock_data):
    constraint = request.getfixturevalue(constraint_fixture)
    constraint.check_constraint(mock_data)  # In case direction state depends on satisfaction
    directions = constraint.calculate_direction(mock_data)
    assert isinstance(directions, dict)
    for layer, tensor_val in directions.items():
        assert layer in constraint.descriptor.variable_keys.union(
            constraint.descriptor.constant_keys
        )
        assert tensor_val.ndim == 2
        # Normalization test for constraints where directions are normalized
        if isinstance(
            constraint, (ScalarConstraint, BinaryConstraint, ImplicationConstraint, SumConstraint)
        ):
            norms = tensor_val.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms))


def test_implication_constraint_logic(mock_data):
    scalar = ScalarConstraint("a", torch.lt, 1.0)
    binary = BinaryConstraint("a", torch.lt, "b")
    implication = ImplicationConstraint(scalar, binary)
    result, head_satisfaction = implication.check_constraint(mock_data)
    assert torch.all((result == 1) | (result == 0))


def test_sum_constraint_weight_mismatch():
    # Should raise ValueError if weights don't match number of tags
    with pytest.raises(ValueError):
        SumConstraint(["a", "b"], torch.lt, ["c", "d"], weights_left=[1.0])
