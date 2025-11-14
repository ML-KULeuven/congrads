import pytest
import torch

from congrads.descriptor import Descriptor


@pytest.fixture
def descriptor():
    return Descriptor()


def test_initialization(descriptor):
    assert descriptor._tag_to_key == {}
    assert descriptor._tag_to_index == {}
    assert descriptor.constant_keys == set()
    assert descriptor.variable_keys == set()


def test_add_tag_variable_layer(descriptor):
    descriptor.add("layer1", "tag1", index=0, constant=False)
    assert descriptor._tag_to_key["tag1"] == "layer1"
    assert descriptor._tag_to_index["tag1"] == 0
    assert "layer1" in descriptor.variable_keys
    assert "layer1" not in descriptor.constant_keys


def test_add_tag_constant_layer(descriptor):
    descriptor.add("layer2", "tag2", index=1, constant=True)
    assert descriptor._tag_to_key["tag2"] == "layer2"
    assert descriptor._tag_to_index["tag2"] == 1
    assert "layer2" in descriptor.constant_keys
    assert "layer2" not in descriptor.variable_keys


def test_add_duplicate_tag_raises(descriptor):
    descriptor.add("layer1", "tag1", index=0)
    with pytest.raises(ValueError):
        descriptor.add("layer1", "tag1", index=1)


def test_add_duplicate_index_in_same_layer_raises(descriptor):
    descriptor.add("layer1", "tag1", index=0)
    with pytest.raises(ValueError):
        descriptor.add("layer1", "tag2", index=0)


def test_exists_method(descriptor):
    descriptor.add("layer1", "tag1", index=0)
    assert descriptor.exists("tag1") is True
    assert descriptor.exists("nonexistent") is False


def test_location_method(descriptor):
    descriptor.add("layer1", "tag1", index=0)
    layer, index = descriptor.location("tag1")
    assert layer == "layer1"
    assert index == 0

    with pytest.raises(ValueError, match="Tag 'tag2' is not registered"):
        descriptor.location("tag2")


def test_select_method(descriptor):
    descriptor.add("layer1", "tag1", index=0)
    batch_data = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = descriptor.select("tag1", batch_data)
    expected = torch.tensor([[1.0], [3.0]])
    assert torch.allclose(result, expected)


def test_add_tag_with_none_index(descriptor):
    descriptor.add("layer1", "tag_none", index=None, constant=False)

    assert descriptor._tag_to_key["tag_none"] == "layer1"
    assert descriptor._tag_to_index["tag_none"] is None

    assert "layer1" in descriptor.variable_keys
    assert "layer1" not in descriptor.constant_keys


def test_location_with_none_index(descriptor):
    descriptor.add("layer1", "tag_none", index=None)

    layer, index = descriptor.location("tag_none")
    assert layer == "layer1"
    assert index is None


def test_select_with_none_index(descriptor):
    descriptor.add("layer1", "tag_none", index=None)

    batch_data = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = descriptor.select("tag_none", batch_data)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)


def test_duplicate_none_index_in_same_layer_raises(descriptor):
    descriptor.add("layer1", "tag_a", index=None)
    with pytest.raises(ValueError):
        descriptor.add("layer1", "tag_b", index=None)
