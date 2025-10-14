"""Module defining the network architectures and components."""

from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential


class MLPNetwork(Module):
    """A multi-layer perceptron (MLP) neural network with configurable hidden layers."""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden_layers=3,
        hidden_dim=35,
        activation=None,
    ):
        """Initialize the MLPNetwork.

        Args:
            n_inputs (int, optional): Number of input features. Defaults to 25.
            n_outputs (int, optional): Number of output features. Defaults to 2.
            n_hidden_layers (int, optional): Number of hidden layers. Defaults to 3.
            hidden_dim (int, optional): Dimensionality of hidden layers. Defaults to 35.
            activation (nn.Module, optional): Activation function module (e.g.,
                `ReLU()`, `Tanh()`, `LeakyReLU(0.1)`). Defaults to `ReLU()`.
        """
        super().__init__()

        # Init object variables
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

        # Default activation function
        if activation is None:
            activation = ReLU()
        self.activation = activation

        # Build network layers
        layers = []

        # Input layer with activation
        layers.append(Linear(n_inputs, hidden_dim))
        layers.append(self.activation)

        # Hidden layers (with activation after each)
        for _ in range(n_hidden_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)

        # Output layer (no activation by default)
        layers.append(Linear(hidden_dim, n_outputs))

        self.network = Sequential(*layers)

    def forward(self, data: dict[str, Tensor]):
        """Run a forward pass through the network.

        Args:
            data (dict[str, Tensor]): Input data to be processed by the network.

        Returns:
            dict: The original data tensor augmented with the network's output (having key "output").
        """
        data["output"] = self.network(data["input"])
        return data
