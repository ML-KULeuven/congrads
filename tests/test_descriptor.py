import pytest
import torch

from congrads.descriptor import Descriptor


@pytest.fixture
def descriptor():
    return Descriptor()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialization(descriptor):
    assert descriptor._layers == {}
    assert descriptor._tags == {}
    assert descriptor.constant_layers == set()
    assert descriptor.variable_layers == set()
    assert descriptor.affects_loss_layers == set()


# ---------------------------------------------------------------------------
# Layer registration
# ---------------------------------------------------------------------------


def test_add_layer_variable(descriptor):
    descriptor.add_layer("layer1")

    assert descriptor.has_layer("layer1")
    assert "layer1" in descriptor.variable_layers
    assert "layer1" not in descriptor.constant_layers
    assert "layer1" in descriptor.affects_loss_layers


def test_add_layer_constant(descriptor):
    descriptor.add_layer("layer1", constant=True, affects_loss=False)

    assert "layer1" in descriptor.constant_layers
    assert "layer1" not in descriptor.variable_layers
    assert "layer1" not in descriptor.affects_loss_layers


def test_add_duplicate_layer_raises(descriptor):
    descriptor.add_layer("layer1")
    with pytest.raises(ValueError):
        descriptor.add_layer("layer1")


# ---------------------------------------------------------------------------
# Tag registration
# ---------------------------------------------------------------------------


def test_add_tag_single_index(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag1", "layer1", index=0)

    assert descriptor.has_tag("tag1")
    layer, index = descriptor.location("tag1")

    assert layer == "layer1"
    assert index == 0


def test_add_tag_none_index(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag_none", "layer1", index=None)

    layer, index = descriptor.location("tag_none")

    assert layer == "layer1"
    assert index is None


def test_add_tag_multiple_indices(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag_multi", "layer1", index=[0, 2])

    layer, index = descriptor.location("tag_multi")

    assert layer == "layer1"
    assert index == (0, 2)


def test_add_tag_unknown_layer_raises(descriptor):
    with pytest.raises(ValueError):
        descriptor.add_tag("tag1", "unknown_layer", index=0)


def test_add_duplicate_tag_raises(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag1", "layer1", index=0)

    with pytest.raises(ValueError):
        descriptor.add_tag("tag1", "layer1", index=1)


def test_duplicate_index_same_layer_warns(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag1", "layer1", index=0)

    with pytest.warns(UserWarning):
        descriptor.add_tag("tag2", "layer1", index=0)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


def test_location_unregistered_tag_raises(descriptor):
    with pytest.raises(ValueError, match="not registered"):
        descriptor.location("unknown")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_select_single_index(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag1", "layer1", index=0)

    batch_data = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = descriptor.select("tag1", batch_data)

    expected = torch.tensor([[1.0], [3.0]])
    assert torch.allclose(result, expected)


def test_select_multiple_indices(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag_multi", "layer1", index=[0, 1])

    batch_data = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = descriptor.select("tag_multi", batch_data)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)


def test_select_none_index_returns_full_tensor(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag_none", "layer1", index=None)

    batch_data = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    result = descriptor.select("tag_none", batch_data)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(result, expected)


def test_select_raises_for_1d_tensor_with_index(descriptor):
    descriptor.add_layer("layer1")
    descriptor.add_tag("tag1", "layer1", index=0)

    batch_data = {"layer1": torch.tensor([1.0, 2.0])}

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        descriptor.select("tag1", batch_data)
