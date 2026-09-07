import random

import numpy as np
import torch
from config import CGGDConfig
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
    random_split,
)

from congrads.utils.utility import DictDatasetWrapper, Seeder, lhs


class Loader:
    def __init__(self, config: CGGDConfig, base_seed: int = 42):
        self.config = config

        self.seeder = Seeder(base_seed=base_seed)
        self.seeder.set_reproducible()

        # spatial domain
        self.x_min, self.x_max = 0, self.config.L
        # diffusive time scale
        self.tau = self.config.L**2 / self.config.kappa
        # time domain
        self.t_min, self.t_max = 0, self.config.lambda_tau * self.tau

        # limits and ranges
        self.x_range = self.x_max - self.x_min
        self.t_range = self.t_max - self.t_min
        self.X_min = [self.x_min, self.t_min]
        self.X_max = [self.x_max, self.t_max]
        self.X_range = [self.x_range, self.t_range]

    def load(self):
        X_IC, u_IC = self._sample_IC(N=self.config.N_IC)
        X_BC, u_BC = self._sample_BC(N=self.config.N_BC)
        X_col, u_col = self._sample_domain(N=self.config.N_col)
        X_test, u_test = self._sample_domain(N=self.config.N_test)

        ctx_col = torch.full((len(X_col), 1), 0.0)  # collocation points
        ctx_IC = torch.full((len(X_IC), 1), 1.0)  # initial condition points
        ctx_BC = torch.full((len(X_BC), 1), 2.0)  # boundary condition points

        X_train = torch.cat([X_col, X_IC, X_BC], dim=0)
        u_train = torch.cat([u_col, u_IC, u_BC], dim=0)
        ctx_train = torch.cat([ctx_col, ctx_IC, ctx_BC], dim=0)

        trainval_dataset = TensorDataset(X_train, u_train, ctx_train)
        test_dataset = TensorDataset(X_test, u_test, torch.full((len(X_test), 1), 0.0))

        # Split the trainval data into training and validation sets
        generator = torch.Generator().manual_seed(self.seeder.roll_seed())
        train_dataset, valid_dataset = random_split(
            trainval_dataset, [0.8, 0.2], generator=generator
        )

        # Convert datasets to Congrads compatible dict-based ones
        train_dataset = DictDatasetWrapper(train_dataset, ["input", "target", "context"])
        valid_dataset = DictDatasetWrapper(valid_dataset, ["input", "target", "context"])
        test_dataset = DictDatasetWrapper(test_dataset, ["input", "target", "context"])

        # Datasets to loaders
        train_loader = self._dataset_to_loader(
            train_dataset, batch_size=self.config.batch_size_train, shuffle=True
        )
        valid_loader = self._dataset_to_loader(
            valid_dataset, batch_size=self.config.batch_size_valid, shuffle=False
        )
        test_loader = self._dataset_to_loader(
            test_dataset, batch_size=self.config.batch_size_test, shuffle=False
        )

        return train_loader, valid_loader, test_loader

    def _dataset_to_loader(
        self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4
    ) -> DataLoader:
        def seed_worker(worker_id):
            worker_seed = self.seeder.roll_seed()
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator().manual_seed(self.seeder.roll_seed())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=True,
        )

    def regular_grid(self, N=128):
        """Provides coordinates and solution on a regular grid"""
        # Coordinate ticks
        x = np.linspace(self.x_min, self.x_max, N)
        t = np.linspace(self.t_min, self.t_max, N)
        # Meshgrid
        xx, tt = np.meshgrid(x, t)
        X_np = np.vstack((xx.flatten(), tt.flatten())).T
        X = self._array2tensor(X_np, exp_dim=False)
        # solution
        u = self._array2tensor(self._analytical_solution(X), exp_dim=False)

        # ---- classification ----
        tol = 1e-12

        x_vals = X_np[:, 0]
        t_vals = X_np[:, 1]

        is_left = np.isclose(x_vals, self.x_min, atol=tol)
        is_right = np.isclose(x_vals, self.x_max, atol=tol)
        is_initial = np.isclose(t_vals, self.t_min, atol=tol)

        identifiers = np.zeros(X_np.shape[0])

        # initial condition first
        identifiers[is_initial] = 1

        # spatial boundary (but not IC)
        boundary = (is_left | is_right) & (~is_initial)
        identifiers[boundary] = 2

        identifiers = torch.as_tensor(identifiers).unsqueeze(1).float()

        return X, u, identifiers

    def _array2tensor(self, array, exp_dim=True):
        """Auxilary function to convert numpy-array to tf-tensor
        expands dimensions if necessary
        """
        if exp_dim:
            array = torch.unsqueeze(array, dim=1)

        return torch.as_tensor(array, dtype=torch.float32)

    def _analytical_solution(self, X):
        """Returns analytical solution for diffusion equation"""
        x, t = X[:, 0], X[:, 1]

        u = torch.sin(torch.pi * x / self.config.L) * torch.exp(-(torch.pi**2) * t / self.tau)
        return self._array2tensor(u, exp_dim=True)

    def _sample_IC(self, N=128):
        """Provides random samples of IC with N data points
        IC: u = sin(pi * x / L)
        """
        # IC coordinates (t=0)
        x_ticks = np.random.rand(N) * self.x_range
        X_IC = self._array2tensor([[x, 0] for x in x_ticks], exp_dim=False)
        # IC temperature
        u_IC = self._array2tensor(
            [[np.sin(np.pi * x / self.config.L)] for x in x_ticks], exp_dim=False
        )

        return X_IC, u_IC

    def _sample_BC(self, N=128):
        """Provides random samples of BC with N data points at each boundary (top and bottom)
        BC: u = 0
        """
        # top boundary coordinates (x=x_max)
        t_ticks = np.random.rand(N) * self.t_range
        X_top = [[self.x_max, t] for t in t_ticks]
        # bottom boundary coordinates (x=x_min)
        t_ticks = np.random.rand(N) * self.t_range
        X_bottom = [[self.x_min, t] for t in t_ticks]
        # add both boundaries and prodive BC temperature
        X_BC = self._array2tensor(X_top + X_bottom, exp_dim=False)
        u_BC = self._array2tensor([[0]] * X_BC.shape[0], exp_dim=False)

        return X_BC, u_BC

    def _sample_domain(self, N=1024):
        """LHS sampling of coordinates inside function domain"""
        # coordinates
        X = self._array2tensor(self.X_min + self.X_range * lhs(2, N), exp_dim=False)
        # solution
        u = self._array2tensor(self._analytical_solution(X), exp_dim=False)

        return X, u
