import pytest
from torch.utils.data import DataLoader, Dataset

from congrads.constraints.registry import COMPARATOR_MAP
from congrads.utils.validation import (
    validate_callable,
    validate_callable_iterable,
    validate_comparator,
    validate_iterable,
    validate_loaders,
    validate_type,
)


def test_validate_type():
    validate_type("x", 5, int)
    validate_type("x", "hello", str)
    validate_type("x", None, int, allow_none=True)
    with pytest.raises(TypeError):
        validate_type("x", 5, str)
    with pytest.raises(TypeError):
        validate_type("x", None, int, allow_none=False)


def test_validate_iterable():
    validate_iterable("x", [1, 2, 3], int)
    validate_iterable("x", {"a", "b"}, str)
    validate_iterable("x", None, int, allow_none=True)
    with pytest.raises(TypeError):
        validate_iterable("x", [1, "a"], int)
    with pytest.raises(TypeError):
        validate_iterable("x", {1, 2, 3}, str)
    with pytest.raises(TypeError):
        validate_iterable("x", None, int, allow_none=False)


def test_validate_comparator_pytorch():
    for comparator in COMPARATOR_MAP.keys():
        validate_comparator("cmp", comparator, COMPARATOR_MAP)
    with pytest.raises(TypeError):
        validate_comparator("cmp", lambda x, y: x > y, COMPARATOR_MAP)
    with pytest.raises(TypeError):
        validate_comparator("cmp", "not a function", COMPARATOR_MAP)


def test_validate_callable():
    validate_callable("func", lambda x: x)
    validate_callable("func", None, allow_none=True)
    with pytest.raises(TypeError):
        validate_callable("func", 5)
    with pytest.raises(TypeError):
        validate_callable("func", None, allow_none=False)


def test_validate_callable_iterable():
    validate_callable_iterable("funcs", [lambda x: x, lambda y: y * 2])
    validate_callable_iterable("funcs", {len, abs})
    validate_callable_iterable("funcs", None, allow_none=True)
    with pytest.raises(TypeError):
        validate_callable_iterable("funcs", [lambda x: x, "not callable"])
    with pytest.raises(TypeError):
        validate_callable_iterable("funcs", None, allow_none=False)


def test_validate_loaders():
    loader1 = DataLoader([])
    loader2 = DataLoader([])
    loader3 = DataLoader([])
    validate_loaders("loaders", (loader1, loader2, loader3))
    with pytest.raises(TypeError):
        validate_loaders("loaders", [loader1, loader2, loader3])
    with pytest.raises(TypeError):
        validate_loaders("loaders", (loader1, loader2))
    with pytest.raises(TypeError):
        validate_loaders("loaders", (loader1, loader2, "invalid"))


class DummySingleOutputDataset(Dataset):
    """A simple dataset returning only one value per sample."""

    def __init__(self):
        self.data = [1, 2, 3]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
