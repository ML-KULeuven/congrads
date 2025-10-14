import pytest
import torch
from torch.nn import Linear, ReLU, Tanh

from congrads.networks import MLPNetwork


def test_default_initialization():
    """Test that MLPNetwork initializes with default parameters."""
    n_inputs = 10
    n_outputs = 3
    model = MLPNetwork(n_inputs, n_outputs)

    # Check object attributes
    assert model.n_inputs == n_inputs
    assert model.n_outputs == n_outputs
    assert model.n_hidden_layers == 3
    assert model.hidden_dim == 35
    assert isinstance(model.activation, ReLU)

    # Check network layer types
    layers = list(model.network)
    # Expected: input Linear + ReLU + (hidden Linear + ReLU)*2 + output Linear => 3*2 + 1 = 7 layers
    assert len(layers) == 7
    assert isinstance(layers[0], Linear)
    assert layers[0].in_features == n_inputs
    assert layers[0].out_features == 35
    assert isinstance(layers[-1], Linear)
    assert layers[-1].out_features == n_outputs


def test_custom_hidden_layers_and_dim():
    """Test initialization with custom hidden layers and dimension."""
    model = MLPNetwork(n_inputs=5, n_outputs=2, n_hidden_layers=4, hidden_dim=20)
    layers = list(model.network)
    # 4 hidden layers: input + 3 hidden + output = 4 Linear + 3 activation = 7 layers
    assert len(layers) == 4 * 2 + 1  # input + hidden layers activations + output
    # Check hidden layer dimensions
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            assert layer.out_features == 20


def test_custom_activation_function():
    """Test that a custom activation function is used correctly."""
    model = MLPNetwork(8, 2, activation=Tanh())
    layers = list(model.network)
    # Every second layer except the last should be Tanh
    for i in range(1, len(layers) - 1, 2):
        assert isinstance(layers[i], Tanh)


def test_forward_pass_output_shape():
    """Test that forward pass produces output of correct shape."""
    n_inputs = 6
    n_outputs = 4
    model = MLPNetwork(n_inputs, n_outputs)
    x = torch.randn(2, n_inputs)
    data = {"input": x}
    output_data = model(data)
    assert "output" in output_data
    out_tensor = output_data["output"]
    assert out_tensor.shape == (2, n_outputs)


def test_forward_pass_preserves_other_data():
    """Ensure that other keys in the input dict are preserved."""
    model = MLPNetwork(3, 1)
    data = {"input": torch.randn(1, 3), "extra": torch.tensor([1.0])}
    output_data = model(data)
    assert "extra" in output_data
    assert output_data["extra"].item() == 1.0


def test_network_computation_consistency():
    """Check that repeated forward passes produce consistent results in eval mode."""
    model = MLPNetwork(5, 2)
    model.eval()
    x = torch.randn(3, 5)
    data = {"input": x.clone()}
    out1 = model(data)["output"]
    out2 = model({"input": x.clone()})["output"]
    assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main()
