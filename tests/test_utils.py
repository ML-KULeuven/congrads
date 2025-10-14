import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from congrads.utils import (
    DictDatasetWrapper,
    validate_callable,
    validate_callable_iterable,
    validate_comparator_pytorch,
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
    validate_comparator_pytorch("cmp", torch.gt)
    validate_comparator_pytorch("cmp", torch.lt)
    with pytest.raises(TypeError):
        validate_comparator_pytorch("cmp", lambda x, y: x > y)
    with pytest.raises(TypeError):
        validate_comparator_pytorch("cmp", "not a function")


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


def test_default_field_names_with_tensor_dataset():
    X = torch.randn(5, 3)
    y = torch.randint(0, 2, (5,))
    base = TensorDataset(X, y)

    wrapped = DictDatasetWrapper(base)

    sample = wrapped[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"field0", "field1"}
    assert torch.equal(sample["field0"], X[0])
    assert torch.equal(sample["field1"], y[0])


def test_custom_field_names_correct_length():
    X = torch.randn(4, 2)
    y = torch.randint(0, 2, (4,))
    base = TensorDataset(X, y)

    wrapped = DictDatasetWrapper(base, field_names=["features", "label"])

    sample = wrapped[0]
    assert set(sample.keys()) == {"features", "label"}
    assert torch.equal(sample["features"], X[0])
    assert torch.equal(sample["label"], y[0])


def test_custom_field_names_too_short():
    X = torch.randn(3, 2)
    y = torch.randint(0, 2, (3,))
    z = torch.ones(3)
    base = TensorDataset(X, y, z)

    wrapped = DictDatasetWrapper(base, field_names=["a"])

    sample = wrapped[0]
    assert set(sample.keys()) == {"a", "field1", "field2"}


def test_custom_field_names_too_long():
    X = torch.randn(2, 2)
    y = torch.randint(0, 2, (2,))
    base = TensorDataset(X, y)

    wrapped = DictDatasetWrapper(base, field_names=["first", "second", "extra", "ignore_me"])

    sample = wrapped[0]
    # Should truncate to match actual dataset length (2 fields)
    assert set(sample.keys()) == {"first", "second"}


def test_single_output_dataset():
    base = DummySingleOutputDataset()
    wrapped = DictDatasetWrapper(base)

    sample = wrapped[0]
    assert set(sample.keys()) == {"field0"}
    assert torch.equal(sample["field0"], torch.tensor(1))


def test_scalar_conversion():
    class ScalarDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return (idx, float(idx))

    base = ScalarDataset()
    wrapped = DictDatasetWrapper(base, field_names=["int_val", "float_val"])

    sample = wrapped[1]
    assert isinstance(sample["int_val"], torch.Tensor)
    assert isinstance(sample["float_val"], torch.Tensor)
    assert sample["int_val"].item() == 1
    assert sample["float_val"].item() == 1.0


def test_len_matches_base_dataset():
    X = torch.randn(7, 2)
    y = torch.randint(0, 2, (7,))
    base = TensorDataset(X, y)

    wrapped = DictDatasetWrapper(base)

    assert len(wrapped) == len(base)
