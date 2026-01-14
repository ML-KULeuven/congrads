"""This module defines several PyTorch dataset classes for loading and working with various datasets.

Each dataset class extends the `torch.utils.data.Dataset` class and provides functionality for
downloading, loading, and transforming specific datasets where applicable.

Classes:

    - SyntheticClusterDataset: A dataset class for generating synthetic clustered 2D data with labels.
    - BiasCorrection: A dataset class for the Bias Correction dataset focused on temperature forecast data.
    - FamilyIncome: A dataset class for the Family Income and Expenditure dataset.

Each dataset class provides methods for downloading the data
(if not already available or synthetic), checking the integrity of the dataset, loading
the data from CSV files or generating synthetic data, and applying
transformations to the data.
"""

import os
import random
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd
import torch
from torch.distributions import Dirichlet
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class BiasCorrection(Dataset):
    """A dataset class for accessing the Bias Correction dataset.

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

    mirrors = ["https://archive.ics.uci.edu/static/public/514/"]
    resources = [
        (
            "bias+correction+of+numerical+prediction+model+temperature+forecast.zip",
            "3deee56d461a2686887c4ae38fe3ccf3",
        )
    ]

    def __init__(self, root: str | Path, transform: Callable, download: bool = False) -> None:
        """Constructor method to initialize the dataset."""
        super().__init__()
        self.root = root
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data_input, self.data_output = self._load_data()

    def _load_data(self):
        """Loads the dataset from the CSV file and applies the transformation.

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
        """Returns the number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset
                (i.e., the number of rows in the input data).
        """
        return self.data_input.shape[0]

    def __getitem__(self, idx):
        """Returns the input-output pair for a given index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                - "input" (torch.Tensor): The input features for the example.
                - "target" (torch.Tensor): The target output for the example.
        """
        example = self.data_input[idx, :]
        target = self.data_output[idx, :]
        example = torch.tensor(example)
        target = torch.tensor(target)
        return {"input": example, "target": target}

    @property
    def data_folder(self) -> str:
        """Returns the path to the folder where the dataset is stored.

        Returns:
            str: The path to the dataset folder.
        """
        return os.path.join(self.root, self.__class__.__name__)

    def _check_exists(self) -> bool:
        """Checks if the dataset is already downloaded and verified.

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
        """Downloads and extracts the dataset.

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
                        url, download_root=self.data_folder, filename=filename, md5=md5
                    )
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors, strict=False):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)


class FamilyIncome(Dataset):
    """A dataset class for accessing the Family Income and Expenditure dataset.

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
        "https://www.kaggle.com/api/v1/datasets/download/grosvenpaul/family-income-and-expenditure"
    ]
    resources = [
        (
            "archive.zip",
            "7d74bc7facc3d7c07c4df1c1c6ac563e",
        )
    ]

    def __init__(self, root: str | Path, transform: Callable, download: bool = False) -> None:
        """Constructor method to initialize the dataset."""
        super().__init__()
        self.root = root
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.data_input, self.data_output = self._load_data()

    def _load_data(self):
        """Load and transform the Family Income and Expenditure dataset.

        Reads the data from the `Family Income and Expenditure.csv` file located
        in `self.data_folder` and applies the transformation function. The input
        and output columns are extracted and returned as NumPy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - input data as a NumPy array of type float32
                - output data as a NumPy array of type float32
        """
        data: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_folder, "Family Income and Expenditure.csv")
        ).pipe(self.transform)

        data_input = data["Input"].to_numpy(dtype=np.float32)
        data_output = data["Output"].to_numpy(dtype=np.float32)

        return data_input, data_output

    def __len__(self):
        """Returns the number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset
                (i.e., the number of rows in the input data).
        """
        return self.data_input.shape[0]

    def __getitem__(self, idx):
        """Returns the input-output pair for a given index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                - "input" (torch.Tensor): The input features for the example.
                - "target" (torch.Tensor): The target output for the example.
        """
        example = self.data_input[idx, :]
        target = self.data_output[idx, :]
        example = torch.tensor(example)
        target = torch.tensor(target)
        return {"input": example, "target": target}

    @property
    def data_folder(self) -> str:
        """Returns the path to the folder where the dataset is stored.

        Returns:
            str: The path to the dataset folder.
        """
        return os.path.join(self.root, self.__class__.__name__)

    def _check_exists(self) -> bool:
        """Checks if the dataset is already downloaded and verified.

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
        """Downloads and extracts the dataset.

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
                        url, download_root=self.data_folder, filename=filename, md5=md5
                    )
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors, strict=False):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)


class SectionedGaussians(Dataset):
    """Synthetic dataset generating smoothly varying Gaussian signals across multiple sections.

    Each section defines a subrange of x-values with its own Gaussian distribution
    (mean and standard deviation). Instead of abrupt transitions, the parameters
    are blended smoothly between sections using a sigmoid function.

    The resulting signal can represent a continuous process where statistical
    properties gradually evolve over time or position.

    Features:
        - Input: Gaussian signal samples (y-values)
        - Context: Concatenation of time (x) and normalized energy feature
        - Target: Exponential decay ground truth from 1 at x_min to 0 at x_max

    Attributes:
        sections (list[dict]): List of section definitions.
        n_samples (int): Total number of samples across all sections.
        n_runs (int): Number of random waveforms generated from base configuration.
        time (torch.Tensor): Sampled x-values, shape [n_samples, 1].
        signal (torch.Tensor): Generated Gaussian signal values, shape [n_samples, 1].
        energies (torch.Tensor): Normalized energy feature, shape [n_samples, 1].
        context (torch.Tensor): Concatenation of time and energy, shape [n_samples, 2].
        x_min (float): Minimum x-value across all sections.
        x_max (float): Maximum x-value across all sections.
        ground_truth_steepness (float): Exponential decay steepness for target output.
        blend_k (float): Sharpness parameter controlling how rapidly means and
            standard deviations transition between sections.
    """

    def __init__(
        self,
        sections: list[dict],
        n_samples: int = 1000,
        n_runs: int = 1,
        seed: int | None = None,
        device="cpu",
        ground_truth_steepness: float = 0.0,
        blend_k: float = 10.0,
    ):
        """Initializes the dataset and generates smoothly blended Gaussian signals.

        Args:
            sections (list[dict]): List of sections. Each section must define:
                - range (tuple[float, float]): Start and end of the x-interval.
                - mean (float): Mean of the Gaussian for this section.
                - std (float): Standard deviation of the Gaussian for this section.
                - max_splits (int): Max number of extra splits per section.
                - split_prob (float): Probability of splitting a section.
                - mean_var (float): How much to vary mean (fraction of original).
                - std_var (float): How much to vary std (fraction of original).
                - range_var (float): How much to vary start/end positions (fraction of section length).
            n_samples (int, optional): Total number of samples to generate. Defaults to 1000.
            n_runs (int, optional): Number of random waveforms to generate from the
                base configuration. Defaults to 1.
            seed (int or None, optional): Random seed for reproducibility. Defaults to None.
            device (str or torch.device, optional): Device on which tensors are allocated.
                Defaults to "cpu".
            ground_truth_steepness (float, optional): Controls how sharply the ground-truth
                exponential decay decreases from 1 to 0. Defaults to 0.0 (linear decay).
            blend_k (float, optional): Controls the sharpness of the sigmoid blending
                between sections. Higher values make the transition steeper; lower
                values make it smoother. Defaults to 10.0.
        """
        self.sections = sections
        self.n_samples = n_samples
        self.n_runs = n_runs
        self.device = device
        self.blend_k = blend_k
        self.ground_truth_steepness = torch.tensor(ground_truth_steepness, device=device)

        if seed is not None:
            torch.manual_seed(seed)

        time: list[torch.Tensor] = []
        signal: list[torch.Tensor] = []
        energy: list[torch.Tensor] = []
        identifier: list[torch.Tensor] = []

        # Compute global min/max for time
        self.t_min = min(s["range"][0] for s in sections)
        self.t_max = max(s["range"][1] for s in sections)

        # Generate waveforms from base configuration
        for run in range(self.n_runs):
            waveform = self._generate_waveform()
            time.append(waveform[0])
            signal.append(waveform[1])
            energy.append(waveform[2])
            identifier.append(torch.full_like(waveform[0], float(run)))

        # Concatenate runs into single tensors
        self.time = torch.cat(time, dim=0)
        self.signal = torch.cat(signal, dim=0)
        self.energy = torch.cat(energy, dim=0)
        self.identifier = torch.cat(identifier, dim=0)
        self.context = torch.hstack([self.time, self.energy, self.identifier])

        # Adjust n_samples in case of rounding mismatch
        self.n_samples = len(self.time)

    def _generate_waveform(
        self,
    ):
        sections = []

        prev_mean = 0

        for sec in self.sections:
            start, end = sec["range"]
            mean, std = sec["add_mean"], sec["std"]
            max_splits = sec["max_splits"]
            split_prob = sec["split_prob"]
            mean_scale = sec["mean_var"]
            std_scale = sec["std_var"]
            range_scale = sec["range_var"]
            section_len = end - start

            # Decide whether to split this section into multiple contiguous parts
            n_splits = 1
            if random.random() < split_prob:
                n_splits += random.randint(1, max_splits)

            # Randomly divide the section into contiguous subsections
            random_fracs = Dirichlet(torch.ones(n_splits) * 3).sample()
            sub_lengths = (random_fracs * section_len).tolist()

            sub_start = start
            for base_len in sub_lengths:
                # Compute unperturbed subsection boundaries
                base_start = sub_start
                base_end = base_start + base_len

                # Define smooth range variation parameters
                max_shift = range_scale * base_len / 2

                # Decide how much to increase the mean overall
                prev_mean += mean * random.uniform(0.0, 2 * mean_scale)

                # Apply small random shifts to start/end, relative to subsection size
                if range_scale > 0:
                    # Keep total ordering consistent (avoid overlaps or inversions)
                    local_shift = random.uniform(-max_shift / n_splits, max_shift / n_splits)
                    new_start = base_start + local_shift
                    new_end = base_end + local_shift
                else:
                    new_start, new_end = base_start, base_end

                # Clamp to valid time bounds
                new_start = max(self.t_min, min(new_start, end))
                new_end = max(new_start, min(new_end, end))

                # Randomize mean and std within allowed ranges
                mean_var = prev_mean * (1 + random.uniform(-mean_scale, mean_scale))
                std_var = std * (1 + random.uniform(-std_scale, std_scale))

                sections.append(
                    {
                        "range": (new_start, new_end),
                        "mean": mean_var,
                        "std": std_var,
                    }
                )

                # Prepare for next subsection
                sub_start = base_end
                prev_mean = mean_var

        # Ensure last section ends exactly at x_max
        sections[-1]["range"] = (sections[-1]["range"][0], self.t_max)

        # Precompute exact float counts
        section_lengths = [s["range"][1] - s["range"][0] for s in sections]
        total_length = sum(section_lengths)
        exact_counts = [length / total_length * self.n_samples for length in section_lengths]

        # Allocate integer samples with rounding, track residual
        n_section_samples_list = []
        residual = 0.0
        for exact in exact_counts[:-1]:
            n = int(round(exact + residual))
            n_section_samples_list.append(n)
            residual += exact - n  # carry over rounding error

        # Assign remaining samples to last section
        n_section_samples_list.append(self.n_samples - sum(n_section_samples_list))

        # Generate data for each section
        signal_segments: list[torch.Tensor] = []
        for i, section in enumerate(sections):
            start, end = section["range"]
            mean, std = section["mean"], section["std"]

            # Samples proportional to section length
            n_section_samples = n_section_samples_list[i]

            # Determine next section’s parameters for blending
            if i < len(sections) - 1:
                next_mean = sections[i + 1]["mean"]
                next_std = sections[i + 1]["std"]
            else:
                next_mean = mean
                next_std = std

            # Sigmoid-based blend curve from 0 → 1
            x = torch.linspace(
                -self.blend_k, self.blend_k, n_section_samples, device=self.device
            ).unsqueeze(1)
            fade = torch.sigmoid(x)
            fade = (fade - fade.min()) / (fade.max() - fade.min())

            # Interpolate mean and std within the section
            mean_curve = mean + (next_mean - mean) * fade
            std_curve = std + (next_std - std) * fade

            # Sample from a gradually changing Gaussian
            y = torch.normal(mean=mean_curve, std=std_curve)
            signal_segments.append(y)

        # Concatenate tensors
        time: torch.Tensor = torch.linspace(
            self.t_min, self.t_max, self.n_samples, device=self.device
        ).unsqueeze(1)
        signal: torch.Tensor = torch.cat(signal_segments, dim=0)

        # Compute and normalize energy feature
        energy = torch.linalg.vector_norm(
            signal - signal[0].reshape([1, -1]), ord=2, dim=1
        ).unsqueeze(1)
        min_e, max_e = energy.min(), energy.max()
        energy = 1 - (energy - min_e) / (max_e - min_e) * 3

        # Combine into context tensor
        return time, signal, energy

    def _compute_ground_truth(self, time: torch.Tensor) -> torch.Tensor:
        """Computes the ground-truth exponential decay at coordinate t.

        Args:
            time (torch.Tensor): Input coordinate (shape: [1]).

        Returns:
            torch.Tensor: Corresponding target value between 0 and 1.
        """
        time_norm = (time - self.t_min) / (self.t_max - self.t_min)
        k = self.ground_truth_steepness
        if k == 0:
            return 1 - time_norm
        return (torch.exp(-k * time_norm) - torch.exp(-k)) / (1 - torch.exp(-k))

    def __len__(self):
        """Returns the total number of generated samples."""
        return self.n_samples

    def __getitem__(self, idx):
        """Retrieves a single dataset sample.

        Args:
            idx (int): Sample index.

        Returns:
            dict:
                - "input":   Gaussian signal value (torch.Tensor),
                - "context: Concatenated features (time, energy, identifier),
                - "target":  Ground-truth exponential decay value

        """
        sigal = self.signal[idx]
        context = self.context[idx]
        time = self.time[idx]
        target = self._compute_ground_truth(time)

        return {
            "input": sigal,
            "context": context,
            "target": target,
        }


class SyntheticMonotonicity(Dataset):
    """Synthetic 1D dataset with monotone ground truth (log(1+x)), plus configurable structured noise.

    True function:
        y_true(x) = log(1 + x)

    Observed:
        y(x) = y_true(x) + heteroscedastic_noise(x) + local oscillatory perturbation

    Args:
        n_samples (int): number of data points (default 200)
        x_range (tuple): range of x values (default [0, 5])
        noise_base (float): baseline noise level (default 0.05)
        noise_scale (float): scale of heteroscedastic noise (default 0.15)
        noise_sharpness (float): steepness of heteroscedastic transition (default 4.0)
        noise_center (float): center point of heteroscedastic increase (default 2.5)
        osc_amplitude (float): amplitude of oscillatory perturbation (default 0.08)
        osc_frequency (float): frequency of oscillation (default 6.0)
        osc_prob (float): probability each sample receives oscillation (default 0.5)
        seed (int or None): random seed
    """

    def __init__(
        self,
        n_samples=200,
        x_range=(0.0, 5.0),
        noise_base=0.05,
        noise_scale=0.15,
        noise_sharpness=4.0,
        noise_center=2.5,
        osc_amplitude=0.08,
        osc_frequency=6.0,
        osc_prob=0.5,
        seed=None,
    ):
        """Synthetic 1D dataset with monotone ground truth (log(1+x)), plus configurable structured noise.

        True function:
        y_true(x) = log(1 + x)

        Observed:
        y(x) = y_true(x) + heteroscedastic_noise(x) + local oscillatory perturbation

        Args:
        n_samples (int): number of data points (default 200)
        x_range (tuple): range of x values (default [0, 5])
        noise_base (float): baseline noise level (default 0.05)
        noise_scale (float): scale of heteroscedastic noise (default 0.15)
        noise_sharpness (float): steepness of heteroscedastic transition (default 4.0)
        noise_center (float): center point of heteroscedastic increase (default 2.5)
        osc_amplitude (float): amplitude of oscillatory perturbation (default 0.08)
        osc_frequency (float): frequency of oscillation (default 6.0)
        osc_prob (float): probability each sample receives oscillation (default 0.5)
        seed (int or None): random seed
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.n_samples = n_samples
        self.x_range = x_range

        # Sample inputs
        self.x = np.random.rand(n_samples) * (x_range[1] - x_range[0]) + x_range[0]
        self.x = np.sort(self.x)

        # Heteroscedastic noise (logistic growth with x)
        noise_sigma = noise_base + noise_scale / (
            1 + np.exp(-noise_sharpness * (self.x - noise_center))
        )

        # Oscillatory perturbation (applied with probability osc_prob)
        mask = (np.random.rand(n_samples) < osc_prob).astype(float)
        osc = mask * (osc_amplitude * np.sin(osc_frequency * self.x))

        # Final observed target
        self.y = np.log1p(self.x) + osc + noise_sigma * np.random.randn(n_samples)

        # Convert to tensors
        self.inputs = torch.tensor(self.x, dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: The number of generated data points.
        """
        return self.n_samples

    def __getitem__(self, idx) -> dict:
        """Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                - "input" (torch.Tensor): The x value.
                - "target" (torch.Tensor): The corresponding y value.
        """
        return {"input": self.inputs[idx], "target": self.targets[idx]}


class SyntheticClusters(Dataset):
    """PyTorch dataset for generating synthetic clustered 2D data with labels.

    Each cluster is defined by its center, size, spread (standard deviation), and label.
    The dataset samples points from a Gaussian distribution centered at the cluster mean.

    Args:
        cluster_centers (list[tuple[float, float]]): Coordinates of each cluster center,
            e.g. [(x1, y1), (x2, y2), ...].
        cluster_sizes (list[int]): Number of points to generate in each cluster.
        cluster_std (list[float]): Standard deviation (spread) of each cluster.
        cluster_labels (list[int]): Class label for each cluster (e.g., 0 or 1).

    Raises:
        AssertionError: If the input lists do not all have the same length.

    Attributes:
        data (torch.Tensor): A concatenated tensor of all generated points with shape (N, 2).
        labels (torch.Tensor): A concatenated tensor of class labels with shape (N,),
            where N is the total number of generated points.
    """

    def __init__(self, cluster_centers, cluster_sizes, cluster_std, cluster_labels):
        """Initialize the ClusterDataset.

        Args:
            cluster_centers (list[tuple[float, float]]): Coordinates of each cluster center,
                e.g. [(x1, y1), (x2, y2), ...].
            cluster_sizes (list[int]): Number of points to generate in each cluster.
            cluster_std (list[float]): Standard deviation (spread) of each cluster.
            cluster_labels (list[int]): Class label for each cluster (e.g., 0 or 1).

        Raises:
            AssertionError: If the input lists do not all have the same length.
        """
        assert (
            len(cluster_centers) == len(cluster_sizes) == len(cluster_std) == len(cluster_labels)
        ), "All input lists must have the same length"

        self.data = []
        self.labels = []

        # Generate points for each cluster
        for center, size, std, label in zip(
            cluster_centers, cluster_sizes, cluster_std, cluster_labels, strict=False
        ):
            x = torch.normal(mean=center[0], std=std, size=(size, 1))
            y = torch.normal(mean=center[1], std=std, size=(size, 1))
            points = torch.cat([x, y], dim=1)

            self.data.append(points)
            self.labels.append(torch.full((size,), label, dtype=torch.long))

        # Concatenate all clusters
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: The number of generated data points.
        """
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        """Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with the following keys:
                - "input" (torch.Tensor): The 2D point at index `idx`.
                - "target" (torch.Tensor): The corresponding class label.
        """
        return {"input": self.data[idx], "target": self.labels[idx]}
