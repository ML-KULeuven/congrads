"""Validation utilities for type checking and argument validation.

This module provides utility functions for validating function arguments,
including type validation, callable validation, and PyTorch-specific
validation functions.
"""

import torch
from torch.utils.data import DataLoader


def validate_type(name, value, expected_types, allow_none=False):
    """Validate that a value is of the specified type(s).

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.
        expected_types (type or tuple of types): Expected type(s) for the value.
        allow_none (bool): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not of the expected type(s).
    """
    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if not isinstance(value, expected_types):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            f"Only values of type {str(expected_types)} are allowed."
        )


def validate_iterable(
    name,
    value,
    expected_element_types,
    allowed_iterables=(list, set, tuple),
    allow_empty=False,
    allow_none=False,
):
    """Validate that a value is an iterable (e.g., list, set) with elements of the specified type(s).

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.
        expected_element_types (type or tuple of types): Expected type(s)
            for the elements.
        allowed_iterables (tuple of types): Iterable types that are
            allowed (default: list and set).
        allow_empty (bool): Whether to allow empty iterables. Defaults to False.
        allow_none (bool): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not an allowed iterable type or if
            any element is not of the expected type(s).
    """
    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if len(value) == 0:
        if not allow_empty:
            raise TypeError(f"Argument {name} cannot be an empty iterable.")
        return

    if not isinstance(value, allowed_iterables):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            f"Only values of type {str(allowed_iterables)} are allowed."
        )
    if not all(isinstance(element, expected_element_types) for element in value):
        raise TypeError(
            f"Invalid elements in {name} '{str(value)}'. "
            f"Only elements of type {str(expected_element_types)} are allowed."
        )


def validate_comparator_pytorch(name, value):
    """Validate that a value is a callable PyTorch comparator function.

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.

    Raises:
        TypeError: If the value is not callable or not a PyTorch comparator.
    """
    # List of valid PyTorch comparator functions
    pytorch_comparators = {torch.gt, torch.lt, torch.ge, torch.le}

    # Check if value is callable and if it's one of
    # the PyTorch comparator functions
    if not callable(value):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. Only callable functions are allowed."
        )

    if value not in pytorch_comparators:
        raise TypeError(
            f"Argument {name} '{str(value)}' is not a valid PyTorch comparator "
            "function. Only PyTorch functions like torch.gt, torch.lt, "
            "torch.ge, torch.le are allowed."
        )


def validate_callable(name, value, allow_none=False):
    """Validate that a value is callable function.

    Args:
        name (str): Name of the argument for error messages.
        value: Value to validate.
        allow_none (bool): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not callable.
    """
    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if not callable(value):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. Only callable functions are allowed."
        )


def validate_callable_iterable(
    name,
    value,
    allowed_iterables=(list, set, tuple),
    allow_none=False,
):
    """Validate that a value is an iterable containing only callable elements.

    This function ensures that the given value is an iterable
    (e.g., list or set and that all its elements are callable functions.

    Args:
        name (str): Name of the argument for error messages.
        value: The value to validate.
        allowed_iterables (tuple of types, optional): Iterable types that are
            allowed. Defaults to (list, set).
        allow_none (bool, optional): Whether to allow the value to be None.
            Defaults to False.

    Raises:
        TypeError: If the value is not an allowed iterable type or if any
            element is not callable.
    """
    if value is None:
        if not allow_none:
            raise TypeError(f"Argument {name} cannot be None.")
        return

    if not isinstance(value, allowed_iterables):
        raise TypeError(
            f"Argument {name} '{str(value)}' is not supported. "
            f"Only values of type {str(allowed_iterables)} are allowed."
        )

    if not all(callable(element) for element in value):
        raise TypeError(
            f"Invalid elements in {name} '{str(value)}'. Only callable functions are allowed."
        )


def validate_loaders(name: str, loaders: tuple[DataLoader, DataLoader, DataLoader]):
    """Validates that `loaders` is a tuple of three DataLoader instances.

    Args:
        name (str): The name of the parameter being validated.
        loaders (tuple[DataLoader, DataLoader, DataLoader]): A tuple of
            three DataLoader instances.

    Raises:
        TypeError: If `loaders` is not a tuple of three DataLoader
            instances or contains invalid types.
    """
    if not isinstance(loaders, tuple) or len(loaders) != 3:
        raise TypeError(f"{name} must be a tuple of three DataLoader instances.")

    for i, loader in enumerate(loaders):
        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"{name}[{i}] must be an instance of DataLoader, got {type(loader).__name__}."
            )
