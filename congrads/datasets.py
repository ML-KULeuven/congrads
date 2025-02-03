"""
This module defines several PyTorch dataset classes for loading and 
working with various datasets. Each dataset class extends the 
`torch.utils.data.Dataset` class and provides functionality for
downloading, loading, and transforming specific datasets.

Classes:

    - BiasCorrection: A dataset class for the Bias Correction dataset 
      focused on temperature forecast data.
    - FamilyIncome: A dataset class for the Family Income and 
      Expenditure dataset.
    - NoisySines: A dataset class that generates noisy sine wave 
      samples with added Gaussian noise.

Each dataset class provides methods for downloading the data 
(if not already available), checking the integrity of the dataset, loading 
the data from CSV files or generating synthetic data, and applying 
transformations to the data.

Key Methods:

    - `__init__`: Initializes the dataset by specifying the root directory, 
      transformation function, and optional download flag.
    - `__getitem__`: Retrieves a specific data point given its index, 
      returning input-output pairs.
    - `__len__`: Returns the total number of examples in the dataset.
    - `download`: Downloads and extracts the dataset from 
       the specified mirrors.
    - `_load_data`: Loads the dataset from CSV files and 
      applies transformations.
    - `_check_exists`: Checks if the dataset is already 
      downloaded and verified.

Each dataset class allows the user to apply custom transformations to the 
dataset through the `transform` argument to allow pre-processing and offers 
the ability to download the dataset if it's not already present on 
the local disk.
"""

import os
from pathlib import Path
from typing import Callable, Union
from urllib.error import URLError

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class BiasCorrection(Dataset):
    """
    A dataset class for accessing the Bias Correction dataset.

    This class extends the `Dataset` class and provides functionality for
    downloading, loading, and transforming the Bias Correction dataset.
    The dataset is focused on temperature forecast data and is made available
    for use with PyTorch. If `download` is set to True, the dataset will be
    downloaded if it is not already available. The data is then loaded,
    and a transformation function is applied to it.

    Args:
        root (Union[str, Path]): The root directory where the dataset
            will be stored or loaded from.
        transform (Callable): A function to transform the dataset
            (e.g., preprocessing).
        download (bool, optional): Whether to download the dataset if it's
            not already present. Defaults to False.

    Raises:
        RuntimeError: If the dataset is not found and `download`
            is not set to True or if all mirrors fail to provide the dataset.
    """

    mirrors = [
        "https://archive.ics.uci.edu/static/public/514/",
    ]

    resources = [
        (
            # pylint: disable-next=line-too-long
            "bias+correction+of+numerical+prediction+model+temperature+forecast.zip",
            "3deee56d461a2686887c4ae38fe3ccf3",
        ),
    ]

    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable,
        download: bool = False,
    ) -> None:
        """
        Constructor method to initialize the dataset.
        """

        super().__init__()
        self.root = root
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data_input, self.data_output = self._load_data()

    def _load_data(self):
        """
        Loads the dataset from the CSV file and applies the transformation.

        The data is read from the `Bias_correction_ucl.csv` file, and the
        transformation function is applied to it.
        The input and output data are separated and returned as numpy arrays.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the input
                and output data as numpy arrays.
        """

        data: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_folder, "Bias_correction_ucl.csv")
        ).pipe(self.transform)

        data_input = data["Input"].to_numpy(dtype=np.float32)
        data_output = data["Output"].to_numpy(dtype=np.float32)

        return data_input, data_output

    def __len__(self):
        """
        Returns the number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset
                (i.e., the number of rows in the input data).
        """

        return self.data_input.shape[0]

    def __getitem__(self, idx):
        """
        Returns the input-output pair for a given index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input-output pair
                as PyTorch tensors.
        """

        example = self.data_input[idx, :]
        target = self.data_output[idx, :]
        example = torch.tensor(example)
        target = torch.tensor(target)
        return example, target

    @property
    def data_folder(self) -> str:
        """
        Returns the path to the folder where the dataset is stored.

        Returns:
            str: The path to the dataset folder.
        """

        return os.path.join(self.root, self.__class__.__name__)

    def _check_exists(self) -> bool:
        """
        Checks if the dataset is already downloaded and verified.

        This method checks that all required files exist and
        their integrity is validated via MD5 checksums.

        Returns:
            bool: True if all resources exist and their
                integrity is valid, False otherwise.
        """

        return all(
            check_integrity(os.path.join(self.data_folder, file_path), checksum)
            for file_path, checksum in self.resources
        )

    def download(self) -> None:
        """
        Downloads and extracts the dataset.

        This method attempts to download the dataset from the mirrors and
        extract it into the appropriate folder. If any error occurs during
        downloading, it will try each mirror in sequence.

        Raises:
            RuntimeError: If all mirrors fail to provide the dataset.
        """

        if self._check_exists():
            return

        os.makedirs(self.data_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    download_and_extract_archive(
                        url,
                        download_root=self.data_folder,
                        filename=filename,
                        md5=md5,
                    )
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)


class FamilyIncome(Dataset):
    """
    A dataset class for accessing the Family Income and Expenditure dataset.

    This class extends the `Dataset` class and provides functionality for
    downloading, loading, and transforming the Family Income and
    Expenditure dataset. The dataset is intended for use with
    PyTorch-based projects, offering a convenient interface for data handling.
    This class provides access to the Family Income and Expenditure dataset
    for use with PyTorch. If `download` is set to True, the dataset will be
    downloaded if it is not already available. The data is then loaded,
    and a user-defined transformation function is applied to it.

    Args:
        root (Union[str, Path]): The root directory where the dataset will
            be stored or loaded from.
        transform (Callable): A function to transform the dataset
            (e.g., preprocessing).
        download (bool, optional): Whether to download the dataset if it's
            not already present. Defaults to False.

    Raises:
        RuntimeError: If the dataset is not found and `download`
            is not set to True or if all mirrors fail to provide the dataset.
    """

    mirrors = [
        # pylint: disable-next=line-too-long
        "https://www.kaggle.com/api/v1/datasets/download/grosvenpaul/family-income-and-expenditure",
    ]

    resources = [
        (
            "archive.zip",
            "7d74bc7facc3d7c07c4df1c1c6ac563e",
        ),
    ]

    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable,
        download: bool = False,
    ) -> None:
        """
        Constructor method to initialize the dataset.
        """

        super().__init__()
        self.root = root
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        self.data_input, self.data_output = self._load_data()

    def _load_data(self):
        """
        Loads the Family Income and Expenditure dataset from the CSV file
        and applies the transformation.

        The data is read from the `Family Income and Expenditure.csv` file,
        and the transformation function is applied to it. The input and
        output data are separated and returned as numpy arrays.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the input
                and output data as numpy arrays.
        """

        data: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_folder, "Family Income and Expenditure.csv")
        ).pipe(self.transform)

        data_input = data["Input"].to_numpy(dtype=np.float32)
        data_output = data["Output"].to_numpy(dtype=np.float32)

        return data_input, data_output

    def __len__(self):
        """
        Returns the number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset
                (i.e., the number of rows in the input data).
        """

        return self.data_input.shape[0]

    def __getitem__(self, idx):
        """
        Returns the input-output pair for a given index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input-output pair
                as PyTorch tensors.
        """

        example = self.data_input[idx, :]
        target = self.data_output[idx, :]
        example = torch.tensor(example)
        target = torch.tensor(target)
        return example, target

    @property
    def data_folder(self) -> str:
        """
        Returns the path to the folder where the dataset is stored.

        Returns:
            str: The path to the dataset folder.
        """

        return os.path.join(self.root, self.__class__.__name__)

    def _check_exists(self) -> bool:
        """
        Checks if the dataset is already downloaded and verified.

        This method checks that all required files exist and
        their integrity is validated via MD5 checksums.

        Returns:
            bool: True if all resources exist and their
                integrity is valid, False otherwise.
        """

        return all(
            check_integrity(os.path.join(self.data_folder, file_path), checksum)
            for file_path, checksum in self.resources
        )

    def download(self) -> None:
        """
        Downloads and extracts the dataset.

        This method attempts to download the dataset from the mirrors
        and extract it into the appropriate folder. If any error occurs
        during downloading, it will try each mirror in sequence.

        Raises:
            RuntimeError: If all mirrors fail to provide the dataset.
        """

        if self._check_exists():
            return

        os.makedirs(self.data_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}"
                try:
                    download_and_extract_archive(
                        url,
                        download_root=self.data_folder,
                        filename=filename,
                        md5=md5,
                    )
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)


class NoisySines(Dataset):
    """
    A PyTorch dataset generating samples from a causal
    sine wave with added noise.

    Args:
        length (int): Number of data points in the dataset.
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave in Hz.
        noise_std (float): Standard deviation of the Gaussian noise.
        bias (float): Offset from zero.

    The sine wave is zero for times before t=0 and follows a
    standard sine wave after t=0, with Gaussian noise added to all points.
    """

    def __init__(
        self,
        length,
        amplitude=1,
        frequency=10.0,
        noise_std=0.05,
        bias=0,
        random_seed=42,
    ):
        """
        Initializes the NoisyCausalSine dataset.
        """
        self.length = length
        self.amplitude = amplitude
        self.frequency = frequency
        self.noise_std = noise_std
        self.bias = bias
        self.random_seed = random_seed

        np.random.seed(self.random_seed)
        self.time = np.linspace(0, 1, length)
        self.noise = np.random.normal(0, self.noise_std, length)

    def __getitem__(self, idx):
        """
        Returns the time and noisy sine wave value for a given index.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
                time value and the noisy sine wave value.
        """

        t = self.time[idx]
        if idx < self.length // 2:
            sine_value = self.bias
            cosine_value = self.bias
        else:
            sine_value = (
                self.amplitude * np.sin(2 * np.pi * self.frequency * t)
                + self.bias
            )
            cosine_value = (
                self.amplitude * np.cos(2 * np.pi * self.frequency * t)
                + self.bias
            )

        # Add noise to the signals
        noisy_sine = sine_value + self.noise[idx]
        noisy_cosine = cosine_value + self.noise[idx]

        # Convert to tensor
        example, target = torch.tensor([t], dtype=torch.float32), torch.tensor(
            [noisy_sine, noisy_cosine], dtype=torch.float32
        )
        return example, target

    def __len__(self):
        """
        Returns the total number of data points in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.length
