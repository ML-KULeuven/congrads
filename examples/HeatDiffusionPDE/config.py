from dataclasses import dataclass


@dataclass
class CGGDConfig:
    L: float = 1.0
    lambda_tau: float = 1.0
    kappa: float = 1.0

    N_IC: int = 128
    N_BC: int = 128
    N_col: int = 1024
    N_test: int = 1024

    batch_size_train: int = 64
    batch_size_valid: int = 512
    batch_size_test: int = 1024
