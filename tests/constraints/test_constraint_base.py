import pytest
import torch
from torch import Tensor

from congrads.constraints.base import Constraint
from congrads.descriptor import Descriptor


# --- Fixtures --- #
class MinimalConstraint(Constraint):
    def check_constraint(self, data: dict[str, Tensor]):
        checks = torch.ones(data["y"].shape[0], 1)
        return checks, torch.ones_like(checks)

    def calculate_direction(self, data: dict[str, Tensor]):
        return {"y": torch.ones_like(data["y"])}


@pytest.fixture
def constraint():
    descriptor = Descriptor()
    descriptor.add_layer("h")
    descriptor.add_layer("y", affects_loss=False, gradients_from="h")
    descriptor.add_tag("y", "y", 0)

    MinimalConstraint.descriptor = descriptor
    MinimalConstraint.device = "cpu"
    return MinimalConstraint(tags={"y"}, name="minimal", rescale_factor=2.0)


@pytest.fixture
def h():
    return torch.tensor([[1.0], [2.0]], requires_grad=True)


# --- correct_direction_norm --- #
def test_correct_direction_norm_value(constraint, h):
    data = {"h": h, "y": 2 * h}
    direction = torch.ones(2, 1)

    corrected = constraint.correct_direction_norm(
        direction, output_layer="y", gradient_layer="h", data=data
    )

    # d(sum(direction * 2h))/dh = 2 -> norm 2 -> corrected direction 0.5
    assert torch.allclose(corrected, torch.full((2, 1), 0.5))


def test_correct_direction_norm_per_sample(constraint):
    h = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)
    weights = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
    data = {"h": h, "y": (h * weights).sum(dim=1, keepdim=True)}
    direction = torch.ones(2, 1)

    corrected = constraint.correct_direction_norm(
        direction, output_layer="y", gradient_layer="h", data=data
    )

    # Per-sample gradient w.r.t. h equals that sample's weights: norms 5 and 1
    assert torch.allclose(corrected, torch.tensor([[0.2], [1.0]]))


def test_correct_direction_norm_zero_gradient_uses_epsilon(constraint, h):
    data = {"h": h, "y": h * 0.0}
    direction = torch.ones(2, 1)

    corrected = constraint.correct_direction_norm(
        direction, output_layer="y", gradient_layer="h", data=data, epsilon=1e-4
    )

    assert torch.isfinite(corrected).all()
    assert torch.allclose(corrected, torch.full((2, 1), 1e4))


def test_correct_direction_norm_retains_graph(constraint, h):
    data = {"h": h, "y": 2 * h}
    direction = torch.ones(2, 1)

    constraint.correct_direction_norm(direction, output_layer="y", gradient_layer="h", data=data)

    # The graph must stay intact so the loss can still be backpropagated afterwards
    data["y"].sum().backward()
    assert torch.allclose(h.grad, torch.full((2, 1), 2.0))
