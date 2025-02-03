"""
This module defines the `Descriptor` class, which is designed to manage
the mapping between neuron names, their corresponding layers, and additional
properties such as constant or variable status. It provides a way to easily 
place constraints on parts of your network, by referencing the neuron names
instead of indices.

The `Descriptor` class allows for easy constraint definitions on parts of
your neural network. It supports registering neurons with associated layers, 
indices, and optional attributes, such as whether the layer is constant 
or variable.

Key Methods:

    - `__init__`: Initializes the `Descriptor` object with empty mappings
      and sets for managing neurons and layers.
    - `add`: Registers a neuron with its associated layer, index, and
      optional constant status.

"""

from .utils import validate_type


class Descriptor:
    """
    A class to manage the mapping between neuron names, their corresponding
    layers, and additional properties (such as min/max values, output,
    and constant variables).

    This class is designed to track the relationship between neurons and
    layers in a neural network. It allows for the assignment of properties
    (like minimum and maximum values, and whether a layer is an output,
    constant, or variable) to each neuron. The data is stored in
    dictionaries and sets for efficient lookups.

    Attributes:
        neuron_to_layer (dict): A dictionary mapping neuron names to
            their corresponding layer names.
        neuron_to_index (dict): A dictionary mapping neuron names to
            their corresponding indices in the layers.
        constant_layers (set): A set of layer names that represent
            constant layers.
        variable_layers (set): A set of layer names that represent
            variable layers.
    """

    def __init__(
        self,
    ):
        """
        Initializes the Descriptor object.
        """

        # Define dictionaries that will translate neuron
        # names to layer and index
        self.neuron_to_layer: dict[str, str] = {}
        self.neuron_to_index: dict[str, int] = {}

        # Define sets that will hold the layers based on which type
        self.constant_layers: set[str] = set()
        self.variable_layers: set[str] = set()

    def add(
        self,
        layer_name: str,
        index: int,
        neuron_name: str,
        constant: bool = False,
    ):
        """
        Adds a neuron to the descriptor with its associated layer,
        index, and properties.

        This method registers a neuron name and associates it with a
        layer, its index, and optional properties such as whether
        the layer is an output or constant layer.

        Args:
            layer_name (str): The name of the layer where the neuron is located.
            index (int): The index of the neuron within the layer.
            neuron_name (str): The name of the neuron.
            constant (bool, optional): Whether the layer is a constant layer.
                Defaults to False.

        Raises:
            TypeError: If a provided attribute has an incompatible type.
            ValueError: If a layer or index is already assigned for a neuron
                or a duplicate index is used within a layer.

        """

        # Type checking
        validate_type("layer_name", layer_name, str)
        validate_type("index", index, int)
        validate_type("neuron_name", neuron_name, str)
        validate_type("constant", constant, bool)

        # Other validations
        if neuron_name in self.neuron_to_layer:
            raise ValueError(
                "There already is a layer registered for the neuron with name "
                f"'{neuron_name}'. Please use a unique name for each neuron."
            )

        if neuron_name in self.neuron_to_index:
            raise ValueError(
                "There already is an index registered for the neuron with name "
                f"'{neuron_name}'. Please use a unique name for each neuron."
            )

        for existing_neuron, assigned_index in self.neuron_to_index.items():
            if (
                assigned_index == index
                and self.neuron_to_layer[existing_neuron] == layer_name
            ):
                raise ValueError(
                    f"The index {index} in layer {layer_name} is already "
                    "assigned. Every neuron must be assigned a different "
                    "index that matches the network's output."
                )

        # Add to dictionaries and sets
        if constant:
            self.constant_layers.add(layer_name)
        else:
            self.variable_layers.add(layer_name)

        self.neuron_to_layer[neuron_name] = layer_name
        self.neuron_to_index[neuron_name] = index
