"""
This module defines the `MLPNetwork` class, which constructs and
operates a multi-layer perceptron (MLP) neural network model. The MLP
network consists of an input layer, multiple hidden layers, and an
output layer. It allows for configurable hyperparameters such as the
number of input features, output features, number of hidden layers,
and the dimensionality of the hidden layers.

Classes:

    - MLPNetwork: A neural network model that implements a multi-layer
      perceptron with customizable layers and dimensionalities.

Key Methods:

    - `__init__`: Initializes the MLP network with specified input size,
      output size, number of hidden layers, and hidden layer dimensionality.
    - `forward`: Performs a forward pass through the network, returning
      both the input and output of the model.
    - `linear`: Creates a basic linear block consisting of a Linear layer
      followed by a ReLU activation function.

The `MLPNetwork` class constructs a fully connected neural network with 
multiple hidden layers, providing flexibility in designing the network 
architecture. It can be used for regression, classification, or other 
machine learning tasks that require a feedforward neural network structure.
"""

from torch.nn import Linear, Module, ReLU, Sequential


class MLPNetwork(Module):
    """
    A multi-layer perceptron (MLP) neural network model consisting of
    an input layer, multiple hidden layers, and an output layer.

    This class constructs an MLP with configurable hyperparameters such as the
    number of input features, output features, number of hidden layers, and
    the dimensionality of hidden layers. It provides methods for both
    building the model and performing a forward pass through the network.

    Args:
        n_inputs (int, optional): The number of input features. Defaults to 25.
        n_outputs (int, optional): The number of output features. Defaults to 2.
        n_hidden_layers (int, optional): The number of hidden layers.
            Defaults to 2.
        hidden_dim (int, optional): The dimensionality of the hidden layers.
            Defaults to 35.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden_layers=3,
        hidden_dim=35,
    ):
        """
        Initializes the MLPNetwork.
        """

        super().__init__()

        # Init object variables
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

        # Set up the components of our model
        self.input = Linear(self.n_inputs, self.hidden_dim)
        self.hidden = Sequential(
            *(
                self.linear(self.hidden_dim, self.hidden_dim)
                for _ in range(n_hidden_layers)
            )
        )
        self.out = Linear(self.hidden_dim, self.n_outputs)

    def forward(self, data):
        """
        Performs a forward pass through the network.

        Args:
            data (Tensor): The input tensor to be passed through the network.

        Returns:
            dict: A dictionary containing the 'input' (original input) and
            'output' (predicted output) of the network.
        """

        output = self.out(self.hidden(self.input(data)))

        return {"input": data, "output": output}

    @staticmethod
    def linear(in_features, out_features):
        """
        Creates a basic linear block with a linear transformation followed
        by a ReLU activation function.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            nn.Module: A sequential module consisting of a Linear layer
            and ReLU activation.
        """

        return Sequential(
            Linear(in_features, out_features),
            ReLU(),
        )
