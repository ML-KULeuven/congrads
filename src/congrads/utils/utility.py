"""This module holds utility functions and classes for the congrads package."""

import inspect
import os
import random
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Generator, Tensor, argsort, cat, int32, unique
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset, random_split


class CSVLogger:
    """A utility class for logging key-value pairs to a CSV file, organized by epochs.

    Supports merging with existing logs or overwriting them.

    Args:
        file_path (str): The path to the CSV file for logging.
        overwrite (bool): If True, overwrites any existing file at the file_path.
        merge (bool): If True, merges new values with existing data in the file.

    Raises:
        ValueError: If both overwrite and merge are True.
        FileExistsError: If the file already exists and neither overwrite nor merge is True.
    """

    def __init__(self, file_path: str, overwrite: bool = False, merge: bool = True):
        """Initializes the CSVLogger.

        Supports merging with existing logs or overwriting them.

        Args:
            file_path (str): The path to the CSV file for logging.
            overwrite (optional, bool): If True, overwrites any existing file at the file_path. Defaults to False.
            merge (optional, bool): If True, merges new values with existing data in the file. Defaults to True.

        Raises:
            ValueError: If both overwrite and merge are True.
            FileExistsError: If the file already exists and neither overwrite nor merge is True.
        """
        self.file_path = file_path
        self.values: dict[tuple[int, str], float] = {}

        if merge and overwrite:
            raise ValueError(
                "The attributes overwrite and merge cannot be True at the "
                "same time. Either specify overwrite=True or merge=True."
            )

        if not os.path.exists(file_path):
            pass
        elif merge:
            self.load()
        elif overwrite:
            pass
        else:
            raise FileExistsError(
                f"A CSV file already exists at {file_path}. Specify "
                "CSVLogger(..., overwrite=True) to overwrite the file."
            )

    def add_value(self, name: str, value: float, epoch: int):
        """Adds a value to the logger for a specific epoch and name.

        Args:
            name (str): The name of the metric or value to log.
            value (float): The value to log.
            epoch (int): The epoch associated with the value.
        """
        self.values[epoch, name] = value

    def save(self):
        """Saves the logged values to the specified CSV file.

        If the file exists and merge is enabled, merges the current data
        with the existing file.
        """
        data = self.to_dataframe(self.values)
        data.to_csv(self.file_path, index=False)

    def load(self):
        """Loads data from the CSV file into the logger.

        Converts the CSV data into the internal dictionary format for
        further updates or operations.
        """
        df = pd.read_csv(self.file_path)
        self.values = self.to_dict(df)

    @staticmethod
    def to_dataframe(values: dict[tuple[int, str], float]) -> pd.DataFrame:
        """Converts a dictionary of values into a DataFrame.

        Args:
            values (dict[tuple[int, str], float]): A dictionary of values keyed by (epoch, name).

        Returns:
            pd.DataFrame: A DataFrame where epochs are rows, names are columns, and values are the cell data.
        """
        # Convert to a DataFrame
        df = pd.DataFrame.from_dict(values, orient="index", columns=["value"])

        # Reset the index to separate epoch and name into columns
        df.index = pd.MultiIndex.from_tuples(df.index, names=["epoch", "name"])
        df = df.reset_index()

        # Pivot the DataFrame so epochs are rows and names are columns
        result = df.pivot(index="epoch", columns="name", values="value")

        # Optional: Reset the column names for a cleaner look
        result = result.reset_index().rename_axis(columns=None)

        return result

    @staticmethod
    def to_dict(df: pd.DataFrame) -> dict[tuple[int, str], float]:
        """Converts a CSVLogger DataFrame to a dictionary the format {(epoch, name): value}."""
        # Set the epoch column as the index (if not already)
        df = df.set_index("epoch")

        # Stack the DataFrame to create a multi-index series
        stacked = df.stack()

        # Convert the multi-index series to a dictionary
        result = stacked.to_dict()

        return result


def split_data_loaders(
    data: Dataset,
    loader_args: dict = None,
    train_loader_args: dict = None,
    valid_loader_args: dict = None,
    test_loader_args: dict = None,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    split_generator: Generator = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Splits a dataset into training, validation, and test sets, and returns corresponding DataLoader objects.

    Args:
        data (Dataset): The dataset to be split.
        loader_args (dict, optional): Default DataLoader arguments, merges
            with loader-specific arguments, overlapping keys from
            loader-specific arguments are superseded.
        train_loader_args (dict, optional): Training DataLoader arguments,
            merges with `loader_args`, overriding overlapping keys.
        valid_loader_args (dict, optional): Validation DataLoader arguments,
            merges with `loader_args`, overriding overlapping keys.
        test_loader_args (dict, optional): Test DataLoader arguments,
            merges with `loader_args`, overriding overlapping keys.
        train_size (float, optional): Proportion of data to be used for
            training. Defaults to 0.8.
        valid_size (float, optional): Proportion of data to be used for
            validation. Defaults to 0.1.
        test_size (float, optional): Proportion of data to be used for
            testing. Defaults to 0.1.
        split_generator (Generator, optional): Optional random seed generator
            to control the splitting of the dataset.

    Returns:
        tuple: A tuple containing three DataLoader objects: one for the
        training, validation and test set.

    Raises:
        ValueError: If the train_size, valid_size, and test_size are not
            between 0 and 1, or if their sum does not equal 1.
    """
    # Validate split sizes
    if not (0 < train_size < 1 and 0 < valid_size < 1 and 0 < test_size < 1):
        raise ValueError("train_size, valid_size, and test_size must be between 0 and 1.")
    if not abs(train_size + valid_size + test_size - 1.0) < 1e-6:
        raise ValueError("train_size, valid_size, and test_size must sum to 1.")

    # Perform the splits
    train_val_data, test_data = random_split(
        data, [1 - test_size, test_size], generator=split_generator
    )
    train_data, valid_data = random_split(
        train_val_data,
        [
            train_size / (1 - test_size),
            valid_size / (1 - test_size),
        ],
        generator=split_generator,
    )

    # Set default arguments for each loader
    train_loader_args = dict(loader_args or {}, **(train_loader_args or {}))
    valid_loader_args = dict(loader_args or {}, **(valid_loader_args or {}))
    test_loader_args = dict(loader_args or {}, **(test_loader_args or {}))

    # Create the DataLoaders
    train_generator = DataLoader(train_data, **train_loader_args)
    valid_generator = DataLoader(valid_data, **valid_loader_args)
    test_generator = DataLoader(test_data, **test_loader_args)

    return train_generator, valid_generator, test_generator


class ZeroLoss(_Loss):
    """A loss function that always returns zero.

    This custom loss function ignores the input and target tensors
    and returns a constant zero loss, which can be useful for debugging
    or when no meaningful loss computation is required.

    Args:
        reduction (str, optional): Specifies the reduction to apply to
            the output. Defaults to "mean". Although specified, it has
            no effect as the loss is always zero.
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize ZeroLoss with a specified reduction method.

        Args:
            reduction (str): Specifies the reduction to apply to the output. Defaults to "mean".
        """
        super().__init__(reduction=reduction)

    def forward(self, predictions: Tensor, target: Tensor, **kwargs) -> torch.Tensor:
        """Return a dummy loss of zero regardless of input and target."""
        return (predictions * 0).sum()


class LossWrapper:
    """Wraps a loss function to optionally accept batch-level data.

    This adapter allows both standard PyTorch loss functions (e.g.
    ``nn.MSELoss``) and custom loss functions that accept an additional
    ``data`` keyword argument to be used interchangeably.

    The wrapped loss can always be called with the same signature:

        loss(output, target, data=batch)

    If the underlying loss function does not accept ``data``, the
    argument is silently ignored.
    """

    def __init__(self, loss_fn: Callable):
        """Initializes the LossWrapper.

        Args:
            loss_fn (Callable): The underlying loss function or callable
                (e.g. a ``torch.nn.Module`` or a custom function).
        """
        self.loss_fn = loss_fn
        self.accepts_data = self._accepts_data()

    def _accepts_data(self) -> bool:
        """Checks whether the wrapped loss function accepts a ``data`` argument.

        The check returns ``True`` if either:
        - The function explicitly defines a ``data`` parameter, or
        - The function accepts arbitrary keyword arguments (``**kwargs``).

        Returns:
            bool: ``True`` if the loss function can accept ``data``,
            ``False`` otherwise.
        """
        # For nn.Module, inspect forward(), not __call__()
        if isinstance(self.loss_fn, nn.Module):
            fn = self.loss_fn.forward
        else:
            fn = self.loss_fn

        sig = inspect.signature(fn)

        return "data" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    def __call__(self, output: Tensor, target: Tensor, *, data: dict | None = None) -> Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground-truth targets.
            data (dict, optional): Full batch data passed to custom loss
                functions that require additional context.

        Returns:
            torch.Tensor: Computed loss value.
        """
        if self.accepts_data:
            return self.loss_fn(output, target, data=data)
        return self.loss_fn(output, target)


def process_data_monotonicity_constraint(data: Tensor, ordering: Tensor, identifiers: Tensor):
    """Reorders input samples to support monotonicity checking.

    Reorders input samples such that:
    1. Samples from the same run are grouped together.
    2. Within each run, samples are sorted chronologically.

    Args:
        data (Tensor): The input data.
        ordering (Tensor): On what to order the data.
        identifiers (Tensor): Identifiers specifying different runs.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Sorted data, ordering, and
        identifiers.
    """
    # Step 1: Sort by run identifiers
    sorted_indices = argsort(identifiers, stable=True, dim=0).reshape(-1)
    data_sorted, ordering_sorted, identifiers_sorted = (
        data[sorted_indices],
        ordering[sorted_indices],
        identifiers[sorted_indices],
    )

    # Step 2: Get unique runs and their counts
    _, counts = unique(identifiers, sorted=False, return_counts=True)
    counts = counts.to(int32)  # Avoid repeated conversions

    sorted_data, sorted_ordering, sorted_identifiers = [], [], []
    index = 0  # Tracks the current batch element index

    # Step 3: Process each run independently
    for count in counts:
        end = index + count
        run_data, run_ordering, run_identifiers = (
            data_sorted[index:end],
            ordering_sorted[index:end],
            identifiers_sorted[index:end],
        )

        # Step 4: Sort within each run by time
        time_sorted_indices = argsort(run_ordering, stable=True, dim=0).reshape(-1)
        sorted_data.append(run_data[time_sorted_indices])
        sorted_ordering.append(run_ordering[time_sorted_indices])
        sorted_identifiers.append(run_identifiers[time_sorted_indices])

        index = end  # Move to next run

    # Step 5: Concatenate results and return
    return (
        cat(sorted_data, dim=0),
        cat(sorted_ordering, dim=0),
        cat(sorted_identifiers, dim=0),
    )


class DictDatasetWrapper(Dataset):
    """A wrapper for PyTorch datasets that converts each sample into a dictionary.

    This class takes any PyTorch dataset and returns its samples as dictionaries,
    where each element of the original sample is mapped to a key. This is useful
    for integration with the Congrads toolbox or other frameworks that expect
    dictionary-formatted data.

    Attributes:
        base_dataset (Dataset): The underlying PyTorch dataset being wrapped.
        field_names (list[str] | None): Names assigned to each field of a sample.
            If None, default names like 'field0', 'field1', ... are generated.

    Args:
        base_dataset (Dataset): The PyTorch dataset to wrap.
        field_names (list[str] | None, optional): Custom names for each field.
            If provided, the list is truncated or extended to match the number
            of elements in a sample. Defaults to None.

    Example:
        Wrapping a TensorDataset with custom field names:

        >>> from torch.utils.data import TensorDataset
        >>> import torch
        >>> dataset = TensorDataset(torch.randn(5, 3), torch.randint(0, 2, (5,)))
        >>> wrapped = DictDatasetWrapper(dataset, field_names=["features", "label"])
        >>> wrapped[0]
        {'features': tensor([...]), 'label': tensor(1)}

        Wrapping a built-in dataset like CIFAR10:

        >>> from torchvision.datasets import CIFAR10
        >>> from torchvision import transforms
        >>> cifar = CIFAR10(
        ...     root="./data", train=True, download=True, transform=transforms.ToTensor()
        ... )
        >>> wrapped_cifar = DictDatasetWrapper(cifar, field_names=["input", "output"])
        >>> wrapped_cifar[0]
        {'input': tensor([...]), 'output': tensor(6)}
    """

    def __init__(self, base_dataset: Dataset, field_names: list[str] | None = None):
        """Initialize the DictDatasetWrapper.

        Args:
            base_dataset (Dataset): The PyTorch dataset to wrap.
            field_names (list[str] | None, optional): Optional list of field names
                for the dictionary output. Defaults to None, in which case
                automatic names 'field0', 'field1', ... are generated.
        """
        self.base_dataset = base_dataset
        self.field_names = field_names

    def __getitem__(self, idx: int):
        """Retrieve a sample from the dataset as a dictionary.

        Each element in the original sample is mapped to a key in the dictionary.
        If the sample is not a tuple or list, it is converted into a single-element
        tuple. Numerical values (int or float) are automatically converted to tensors.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary mapping field names to sample values.
        """
        sample = self.base_dataset[idx]

        # Ensure sample is always a tuple
        if not isinstance(sample, (tuple, list)):
            sample = (sample,)

        n_fields = len(sample)

        # Generate default field names if none are provided
        if self.field_names is None:
            names = [f"field{i}" for i in range(n_fields)]
        else:
            names = list(self.field_names)
            if len(names) < n_fields:
                names.extend([f"field{i}" for i in range(len(names), n_fields)])
            names = names[:n_fields]  # truncate if too long

        # Build dictionary
        out = {}
        for name, value in zip(names, sample, strict=False):
            if isinstance(value, (int, float)):
                value = torch.tensor(value)
            out[name] = value

        return out

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Length of the underlying dataset.
        """
        return len(self.base_dataset)


class Seeder:
    """A deterministic seed manager for reproducible experiments.

    This class provides a way to consistently generate pseudo-random
    seeds derived from a fixed base seed. It ensures that different
    libraries (Python's `random`, NumPy, and PyTorch) are initialized
    with reproducible seeds, making experiments deterministic across runs.
    """

    def __init__(self, base_seed: int):
        """Initialize the Seeder with a base seed.

        Args:
            base_seed (int): The initial seed from which all subsequent
                pseudo-random seeds are deterministically derived.
        """
        self._rng = random.Random(base_seed)

    def roll_seed(self) -> int:
        """Generate a new deterministic pseudo-random seed.

        Each call returns an integer seed derived from the internal
        pseudo-random generator, which itself is initialized by the
        base seed.

        Returns:
            int: A pseudo-random integer seed in the range [0, 2**31 - 1].
        """
        return self._rng.randint(0, 2**31 - 1)

    def set_reproducible(self) -> None:
        """Configure global random states for reproducibility.

        Seeds the following libraries with deterministically generated
        seeds based on the base seed:
          - Python's built-in `random`
          - NumPy's random number generator
          - PyTorch (CPU and GPU)

        Also enforces deterministic behavior in PyTorch by:
          - Seeding all CUDA devices
          - Disabling CuDNN benchmarking
          - Enabling CuDNN deterministic mode
        """
        random.seed(self.roll_seed())
        np.random.seed(self.roll_seed())
        torch.manual_seed(self.roll_seed())
        torch.cuda.manual_seed_all(self.roll_seed())

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
