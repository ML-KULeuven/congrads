"""Descriptor utilities for mapping semantic tags to model layers such as inputs and outputs.

This module defines the `Descriptor` class, which manages a structured
mapping between:

- Layers (data dictionary keys representing model layers such as inputs, outputs, or intermediate features)
- Tags (named references to full layers or specific feature indices)

Layers describe properties of a model output tensor, such as whether it is
constant and whether it contributes to the loss. Tags provide semantic names
for selecting either:

- The full layer output (index=None)
- A single feature column (index=int)
- Multiple feature columns (index=Iterable[int])

This abstraction allows constraints and other logic to reference model
outputs by descriptive tag names instead of hard-coded tensor indices.
"""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from torch import Tensor

from .utils.validation import validate_iterable, validate_type

__all__ = ["Descriptor"]


@dataclass(frozen=True)
class Layer:
    key: str
    constant: bool
    affects_loss: bool


@dataclass(frozen=True)
class Tag:
    layer: str
    index: int | tuple[int, ...] | None


class Descriptor:
    """Registry for model output layers and semantic tags.

    A `Descriptor` maintains two mappings:

    - Layers: represent keys in the model's data dictionary and store
      metadata such as whether the layer is constant and whether it
      affects the loss.
    - Tags: map semantic names to a specific layer and an optional
      index (or indices) within that layer.

    Tags allow selecting either:
    - The full layer tensor (index=None)
    - A single feature column (index=int)
    - Multiple feature columns (index=tuple[int, ...])

    The descriptor does not validate tensor shapes at registration time.
    Selection logic assumes tensors follow the shape convention:
    [batch_size, features].
    """

    def __init__(
        self,
    ):
        """Initialize an empty descriptor.

        Creates internal registries for layers and tags.
        """
        self._layers: dict[str, Layer] = {}
        self._tags: dict[str, Tag] = {}

    def add_layer(self, key: str, constant: bool = False, affects_loss: bool = True):
        """Register a new layer.

        A layer corresponds to a key in the model's data dictionary and
        describes metadata about that output tensor.

        Args:
            key (str): Name of the layer (must match a key in the data dictionary).
            constant (bool, optional): Whether the layer represents constant data.
                Defaults to False.
            affects_loss (bool, optional): Whether this layer contributes to
                the loss computation. Defaults to True.

        Raises:
            TypeError: If arguments have incorrect types.
            ValueError: If a layer with the same key is already registered.
        """
        # Type checking
        validate_type("key", key, str)
        validate_type("constant", constant, bool)
        validate_type("affects_loss", affects_loss, bool)

        # Other validations
        if key in self._layers:
            raise ValueError(f"Layer '{key}' already exists.")

        # Store layer information
        self._layers[key] = Layer(key=key, constant=constant, affects_loss=affects_loss)

    def add_tag(
        self,
        tag: str,
        layer: str,
        index: int | Iterable[int] | None = None,
    ):
        """Register a semantic tag for a layer or part of a layer.

        A tag maps a descriptive name to a specific selection within
        a registered layer.

        Index behavior:
            - None  → select the full layer tensor
            - int   → select a single feature column
            - Iterable[int] → select multiple feature columns

        Args:
            tag (str): Unique name of the tag.
            layer (str): Name of a previously registered layer.
            index (int | Iterable[int] | None, optional):
                Feature index or indices within the layer tensor.

        Raises:
            TypeError: If arguments have incorrect types.
            ValueError:
                - If the layer is not registered.
                - If the tag already exists.
                - If the index contains duplicates.
        """
        # Type checking
        validate_type("tag", tag, str)
        validate_type("layer", layer, str)

        if isinstance(index, int):
            validate_type("index", index, int, allow_none=True)
        else:
            validate_iterable("index", index, int, allow_none=True)

        # Normalize index
        if not isinstance(index, (int, type(None))):
            index = tuple(index)

        # Other validations
        if layer not in self._layers:
            raise ValueError(f"Layer '{layer}' is not registered.")

        if tag in self._tags:
            raise ValueError(f"Tag '{tag}' already exists.")

        if index is not None and not isinstance(index, int):
            if len(set(index)) != len(index):
                raise ValueError(f"Duplicate indices in tag '{tag}'.")

        for existing_name, existing_tag in self._tags.items():
            if existing_tag.layer == layer and existing_tag.index == index:
                warnings.warn(
                    f"Index '{index}' for layer '{layer}' is already assigned to tag '{existing_name}'.",
                    UserWarning,
                    stacklevel=2,
                )

        # Store tag information
        self._tags[tag] = Tag(
            layer=layer,
            index=index,
        )

    def has_layer(self, key: str) -> bool:
        """Return whether a layer is registered."""
        return key in self._layers

    def has_tag(self, tag: str) -> bool:
        """Return whether a tag is registered."""
        return tag in self._tags

    def location(self, tag: str) -> tuple[str, int | tuple[int, ...] | None]:
        """Return the layer name and index associated with a tag.

        Args:
            tag (str): Registered tag name.

        Returns:
            tuple[str, int | tuple[int, ...] | None]:
                - Layer name
                - Index specification (None, int, or tuple[int, ...])

        Raises:
            ValueError: If the tag is not registered.
        """
        tag_info = self._tags.get(tag)
        if tag_info is None:
            raise ValueError(f"Tag '{tag}' is not registered in descriptor.")
        return tag_info.layer, tag_info.index

    def select(self, tag: str, data: dict[str, Tensor]) -> Tensor:
        """Select data from a layer using a registered tag.

        The tensor is retrieved from the provided data dictionary using
        the tag's associated layer key.

        Selection behavior:
            - index=None  → return full tensor
            - index=int   → return a single feature column with shape
                             [batch_size, 1]
            - index=tuple → return selected feature columns

        Args:
            tag (str): Registered tag name.
            data (dict[str, Tensor]): Dictionary containing model outputs.

        Returns:
            Tensor: Selected tensor slice.

        Raises:
            ValueError:
                - If the tag is not registered.
                - If indexed selection is requested but the tensor does
                  not have at least two dimensions.
        """
        key, index = self.location(tag)
        selection = data[key]

        if index is not None and selection.ndim < 2:
            raise ValueError(
                f"Data for key '{key}' must have at least 2 dimensions to select index '{index}'. "
                "Tensors are assumed to have shape [batch, features]."
            )

        if index is None:
            return selection
        if isinstance(index, int):
            return selection[:, index : index + 1]
        return selection[:, index]

    @property
    def constant_layers(self) -> set[str]:
        """Return the set of registered layer keys marked as constant."""
        return {key for key, layer in self._layers.items() if layer.constant}

    @property
    def variable_layers(self) -> set[str]:
        """Return the set of registered layer keys marked as variable."""
        return {key for key, layer in self._layers.items() if not layer.constant}

    @property
    def affects_loss_layers(self) -> set[str]:
        """Return the set of registered layer keys that affect the loss."""
        return {key for key, layer in self._layers.items() if layer.affects_loss}
