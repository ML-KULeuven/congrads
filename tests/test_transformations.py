import pytest
import torch
from torch import Tensor

from congrads.transformations import (
    ApplyOperator,
    DenormalizeMinMax,
    IdentityTransformation,
    Transformation,
)


# --- Fixtures --- #
@pytest.fixture
def sample_tensor():
    return torch.tensor([[0.0, 0.5], [1.0, 0.25]])


# --- Base class tests --- #
def test_transformation_requires_tag_type():
    # Should raise TypeError if tag is not a string

    with pytest.raises(TypeError):
        Transformation(123)  # type: ignore


# --- IdentityTransformation tests --- #
def test_identity_transformation_returns_input(sample_tensor):
    identity = IdentityTransformation("a")
    output = identity(sample_tensor)
    assert isinstance(output, Tensor)
    # Should be exactly equal
    assert torch.allclose(output, sample_tensor)


# --- DenormalizeMinMax tests --- #
def test_denormalize_min_max_initialization(sample_tensor):
    denorm = DenormalizeMinMax("a", min=2.0, max=4.0)
    assert denorm.tag == "a"
    assert denorm.min == 2.0
    assert denorm.max == 4.0


def test_denormalize_min_max_type_checks():
    # min and max must be numbers
    with pytest.raises(TypeError):
        DenormalizeMinMax("a", min="0", max=1)
    with pytest.raises(TypeError):
        DenormalizeMinMax("a", min=0, max="1")


def test_denormalize_min_max_transformation(sample_tensor):
    denorm = DenormalizeMinMax("a", min=2.0, max=4.0)
    output = denorm(sample_tensor)
    expected = sample_tensor * (denorm.max - denorm.min) + denorm.min
    assert torch.allclose(output, expected)


# --- ApplyOperator tests --- #
def test_apply_operator_initialization(sample_tensor):
    op = ApplyOperator("a", torch.add, 5)
    assert op.tag == "a"
    assert op.operator == torch.add
    assert op.value == 5


def test_apply_operator_type_checks():
    # operator must be callable
    with pytest.raises(TypeError):
        ApplyOperator("a", operator=123, value=1)
    # value must be a number
    with pytest.raises(TypeError):
        ApplyOperator("a", torch.add, value="x")


def test_apply_operator_transformation(sample_tensor):
    add_op = ApplyOperator("a", torch.add, 3.0)
    output = add_op(sample_tensor)
    expected = torch.add(sample_tensor, 3.0)
    assert torch.allclose(output, expected)

    mul_op = ApplyOperator("b", torch.mul, 2.0)
    output_mul = mul_op(sample_tensor)
    expected_mul = torch.mul(sample_tensor, 2.0)
    assert torch.allclose(output_mul, expected_mul)


# --- Combined functionality tests --- #
def test_sequence_of_transformations(sample_tensor):
    identity = IdentityTransformation("a")
    denorm = DenormalizeMinMax("b", min=0, max=10)
    add_op = ApplyOperator("c", torch.add, 1)

    x = sample_tensor.clone()
    x = identity(x)
    x = denorm(x)
    x = add_op(x)

    expected = torch.add(sample_tensor * (10 - 0) + 0, 1)
    assert torch.allclose(x, expected)
