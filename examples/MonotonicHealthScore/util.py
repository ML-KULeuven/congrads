import random
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset, Subset


def split_dataset(
    dataset: Dataset,
    loader_args: dict = None,
    train_loader_args: dict = None,
    valid_loader_args: dict = None,
    test_loader_args: dict = None,
    seed: int = 42,
    train_valid_split: float = 0.9,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset into train, validation, and test DataLoaders.

    One waveform (run) is randomly selected for testing, based on a seed.
    Validation samples come from the same runs as training (not unseen).
    """
    # Set default arguments for each loader
    train_loader_args = dict(loader_args or {}, **(train_loader_args or {}))
    valid_loader_args = dict(loader_args or {}, **(valid_loader_args or {}))
    test_loader_args = dict(loader_args or {}, **(test_loader_args or {}))

    # Group indices by run_identifier
    run_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset):
        run_id = int(sample["context"][-1].item())
        run_to_indices[run_id].append(idx)

    all_runs = list(run_to_indices.keys())

    # Pick a random run for testing
    random.seed(seed)
    test_run = random.choice(all_runs)
    test_runs = [test_run]
    train_runs = [r for r in all_runs if r not in test_runs]

    # Collect indices
    train_indices = [i for r in train_runs for i in run_to_indices[r]]
    test_indices = [i for r in test_runs for i in run_to_indices[r]]

    # Split train_indices into train/validation
    # (same runs, just different random shuffle)
    random.shuffle(train_indices)
    split_point = int(train_valid_split * len(train_indices))
    valid_indices = train_indices[split_point:]
    train_indices = train_indices[:split_point]

    # Create subsets
    train_data = Subset(dataset, train_indices)
    valid_data = Subset(dataset, valid_indices)
    test_data = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_data, **train_loader_args)
    valid_loader = DataLoader(valid_data, **valid_loader_args)
    test_loader = DataLoader(test_data, **test_loader_args)

    return train_loader, valid_loader, test_loader
