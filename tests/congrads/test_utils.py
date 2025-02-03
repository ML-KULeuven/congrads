import pytest
import torch

from congrads.utils import (
    validate_callable,
    validate_comparator_pytorch,
    validate_iterable,
    validate_type,
)


# Test for validate_type function
def test_validate_type_valid():
    validate_type("test_int", 10, int)
    validate_type("test_str", "hello", str)
    validate_type("test_tuple", (1, 2, 3), tuple)


def test_validate_type_invalid():
    with pytest.raises(TypeError):
        validate_type("test_int", "hello", int)

    with pytest.raises(TypeError):
        validate_type("test_list", [1, 2, 3], tuple)


# Test for validate_iterable function
def test_validate_iterable_valid():
    validate_iterable("test_list", [1, 2, 3], int)
    validate_iterable("test_set", {1, 2, 3}, int)


def test_validate_iterable_invalid_type():
    with pytest.raises(TypeError):
        validate_iterable("test_list", "not_a_list", int)


def test_validate_iterable_invalid_elements():
    with pytest.raises(TypeError):
        validate_iterable("test_list", [1, "not_int"], int)


def test_validate_iterable_none():
    validate_iterable("test_none", None, int, allow_none=True)

    with pytest.raises(TypeError):
        validate_iterable("test_none", None, int, allow_none=False)


# Test for validate_comparator_pytorch function
def test_validate_comparator_pytorch_valid():
    validate_comparator_pytorch("test_comparator", torch.gt)
    validate_comparator_pytorch("test_comparator", torch.lt)


def test_validate_comparator_pytorch_invalid_callable():
    with pytest.raises(TypeError):
        validate_comparator_pytorch("test_comparator", "not_a_function")


def test_validate_comparator_pytorch_invalid_comparator():
    with pytest.raises(TypeError):
        validate_comparator_pytorch("test_comparator", torch.add)


# Test for validate_callable function
def test_validate_callable_valid():
    def sample_function():
        pass

    validate_callable("test_callable", sample_function)
    validate_callable("test_callable_none", None, allow_none=True)


def test_validate_callable_invalid():
    with pytest.raises(TypeError):
        validate_callable("test_callable", "not_a_function")

    with pytest.raises(TypeError):
        validate_callable("test_callable_none", None, allow_none=False)
