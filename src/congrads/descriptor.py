"""This module defines the `Descriptor` class, which allows assigning tags to parts in the network.

It is designed to manage the mapping between tags, their corresponding data dictionary keys and indices,
and additional properties such as constant or variable status. It provides a way to easily
place constraints on parts of your network, by referencing the tags
instead of indices.

The `Descriptor` class allows for easy constraint definitions on parts of
your neural network. It supports registering tags with associated data dictionary keys,
indices, and optional attributes, such as whether the data is constant or variable.
"""

from torch import Tensor

from .utils.validation import validate_type


class Descriptor:
    """A class to manage the mapping between tags.

    It represents data locations in the data dictionary and holds the dictionary keys, indices,
    and additional properties (such as min/max values, output, and constant variables).

    This class is designed to manage the relationships between the assigned tags and the
    data dictionary keys in a neural network model. It allows for the assignment of properties
    (like minimum and maximum values, and whether data is an output, constant, or variable) to
    each tag. The data is stored in dictionaries and sets for efficient lookups.

    Attributes:
        constant_keys (set): A set of keys that represent constant data in the data dictionary.
        variable_keys (set): A set of keys that represent variable data in the data dictionary.
        affects_loss_keys (set): A set of keys that represent data affecting the loss computation.
    """

    def __init__(
        self,
    ):
        """Initializes the Descriptor object."""
        # Define dictionaries that will translate tags to keys and indices
        self._tag_to_key: dict[str, str] = {}
        self._tag_to_index: dict[str, int] = {}

        # Define sets that will hold the keys based on which type
        self.constant_keys: set[str] = set()
        self.variable_keys: set[str] = set()
        self.affects_loss_keys: set[str] = set()

    def add(
        self,
        key: str,
        tag: str,
        index: int = None,
        constant: bool = False,
        affects_loss: bool = True,
    ):
        """Adds a tag to the descriptor with its associated key, index, and properties.

        This method registers a tag name and associates it with a
        data dictionary key, its index, and optional properties such as whether
        the key hold output or constant data.

        Args:
            key (str): The key on which the tagged data is located in the data dictionary.
            tag (str): The identifier of the tag.
            index (int): The index were the data is present. Defaults to None.
            constant (bool, optional): Whether the data is constant and is not learned. Defaults to False.
            affects_loss (bool, optional): Whether the data affects the loss computation. Defaults to True.

        Raises:
            TypeError: If a provided attribute has an incompatible type.
            ValueError: If a key or index is already assigned for a tag or a duplicate index is used within a key.
        """
        # Type checking
        validate_type("key", key, str)
        validate_type("tag", tag, str)
        validate_type("index", index, int, allow_none=True)
        validate_type("constant", constant, bool)
        validate_type("affects_loss", affects_loss, bool)

        # Other validations
        if tag in self._tag_to_key:
            raise ValueError(
                f"There already is a key registered for the tag '{tag}'. "
                "Please use a unique key name for each tag."
            )

        if tag in self._tag_to_index:
            raise ValueError(
                f"There already is an index registered for the tag '{tag}'. "
                "Please use a unique name for each tag."
            )

        for existing_tag, assigned_index in self._tag_to_index.items():
            if assigned_index == index and self._tag_to_key[existing_tag] == key:
                raise ValueError(
                    f"The index {index} on key {key} is already "
                    "assigned. Every tag must be assigned a different "
                    "index that matches the network's output."
                )

        # Add to dictionaries and sets
        # TODO this now happens on key level, can this also be done on tag level?
        if constant:
            self.constant_keys.add(key)
        else:
            self.variable_keys.add(key)

        if affects_loss:
            self.affects_loss_keys.add(key)

        self._tag_to_key[tag] = key
        self._tag_to_index[tag] = index

    def exists(self, tag: str) -> bool:
        """Check if a tag is registered in the descriptor.

        Args:
            tag (str): The tag identifier to check.

        Returns:
            bool: True if the tag is registered, False otherwise.
        """
        return tag in self._tag_to_key and tag in self._tag_to_index

    def location(self, tag: str) -> tuple[str, int | None]:
        """Get the key and index for a given tag.

        Looks up the mapping for a registered tag and returns the associated
        dictionary key and the index.

        Args:
            tag (str): The tag identifier. Must be registered.

        Returns:
            tuple ((str, int | None)): A tuple containing:
                - The key in the data dictionary which holds the data (str).
                - The tensor index where the data is present or None (int | None).

        Raises:
            ValueError: If the tag is not registered in the descriptor.
        """
        key = self._tag_to_key.get(tag)
        index = self._tag_to_index.get(tag)
        if key is None:
            raise ValueError(f"Tag '{tag}' is not registered in descriptor.")
        return key, index

    def select(self, tag: str, data: dict[str, Tensor]) -> Tensor:
        """Extract prediction values for a specific tag.

        Retrieves the key and index associated with a tag and selects
        the corresponding slice from the given prediction tensor.
        Returns the full tensor if no index was specified when registering the tag.

        Args:
            tag (str): The tag identifier. Must be registered.
            data (dict[str, Tensor]): Dictionary that holds batch data, model predictions and context.

        Returns:
            Tensor: A tensor slice of shape ``(batch_size, 1)`` containing
            the predictions for the specified tag, or the full tensor if no index was specified when registering the tag.

        Raises:
            ValueError: If the tag is not registered in the descriptor.
        """
        key, index = self.location(tag)
        if index is None:
            return data[key]
        return data[key][:, index : index + 1]
