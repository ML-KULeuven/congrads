from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from congrads.core.constraint_engine import ConstraintEngine

# -------------------------
# Dummy / Fake Components
# -------------------------


class DummyConstraint:
    def __init__(
        self,
        name="C1",
        enforce=True,
        rescale_factor=1.0,
        layers=None,
    ):
        self.name = name
        self.enforce = enforce
        self.rescale_factor = rescale_factor
        self.layers = layers or {"x"}

    def check_constraint(self, data: dict[str, Tensor]):
        """
        Returns:
          checks: 1 if satisfied, 0 if violated
          mask:   1 if active
        """
        # batch size = data["x"].shape[0]
        checks = torch.tensor([1.0, 0.0], device=data["x"].device)
        mask = torch.tensor([1.0, 1.0], device=data["x"].device)
        return checks, mask

    def calculate_direction(self, data: dict[str, Tensor]):
        # Direction per layer
        return {
            "x": torch.ones_like(data["x"]),
        }


class DummyDescriptor:
    def __init__(self):
        self.variable_keys = {"x"}
        self.affects_loss_keys = {"x"}


class DummyMetricManager:
    def __init__(self):
        self.accumulate = MagicMock()


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def descriptor():
    return DummyDescriptor()


@pytest.fixture
def metric_manager():
    return DummyMetricManager()


@pytest.fixture
def aggregator():
    # Simple sum aggregator
    return lambda x: x.sum()


@pytest.fixture
def constraint():
    return DummyConstraint()


@pytest.fixture
def engine(
    constraint,
    descriptor,
    metric_manager,
    device,
    aggregator,
):
    return ConstraintEngine(
        constraints=[constraint],
        descriptor=descriptor,
        metric_manager=metric_manager,
        device=device,
        epsilon=1e-6,
        aggregator=aggregator,
        enforce_all=True,
    )


@pytest.fixture
def data():
    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    return {"x": x}


@pytest.fixture
def loss(data):
    # Simple loss so gradients exist
    return (data["x"] ** 2).mean()


# -------------------------
# Core Behavior Tests
# -------------------------


def test_train_applies_constraints(engine, data, loss):
    adjusted_loss = engine.train(data, loss)

    assert torch.is_tensor(adjusted_loss)
    assert adjusted_loss.requires_grad
    assert adjusted_loss.item() != pytest.approx(loss.item())


def test_validate_does_not_modify_loss(engine, data, loss):
    out = engine.validate(data, loss)
    assert torch.is_tensor(out)
    assert torch.isclose(out, loss)


def test_test_does_not_modify_loss(engine, data, loss):
    out = engine.test(data, loss)
    assert torch.is_tensor(out)
    assert torch.isclose(out, loss)


def test_csr_logged_for_each_phase(engine, data, loss, metric_manager):
    engine.train(data, loss)
    engine.validate(data, loss)
    engine.test(data, loss)

    calls = metric_manager.accumulate.call_args_list

    # Expect CSR logs for each phase
    keys = [args[0][0] for args in calls]

    assert "C1/train" in keys
    assert "CSR/train" in keys
    assert "C1/valid" in keys
    assert "CSR/valid" in keys
    assert "C1/test" in keys
    assert "CSR/test" in keys


def test_csr_value_correct(engine, data, loss, metric_manager):
    engine.train(data, loss)

    # checks = [1, 0], mask = [1, 1] => CSR = 0.5
    csr_call = metric_manager.accumulate.call_args_list[0]
    csr_value = csr_call[0][1]

    assert torch.is_tensor(csr_value)
    assert csr_value.item() == pytest.approx(0.5)


# -------------------------
# Enforcement Logic
# -------------------------


def test_constraint_not_enforced_when_constraint_flag_false(
    constraint,
    descriptor,
    metric_manager,
    device,
    aggregator,
    data,
    loss,
):
    constraint.enforce = False

    engine = ConstraintEngine(
        constraints=[constraint],
        descriptor=descriptor,
        metric_manager=metric_manager,
        device=device,
        epsilon=1e-6,
        aggregator=aggregator,
        enforce_all=True,
    )

    out = engine.train(data, loss)
    assert torch.is_tensor(out)
    assert torch.isclose(out, loss)


def test_constraint_not_enforced_when_enforce_all_false(
    constraint,
    descriptor,
    metric_manager,
    device,
    aggregator,
    data,
    loss,
):
    engine = ConstraintEngine(
        constraints=[constraint],
        descriptor=descriptor,
        metric_manager=metric_manager,
        device=device,
        epsilon=1e-6,
        aggregator=aggregator,
        enforce_all=False,
    )

    out = engine.train(data, loss)
    assert torch.is_tensor(out)
    assert torch.isclose(out, loss)


def test_constraint_not_enforced_in_validation(engine, data, loss):
    out = engine.validate(data, loss)
    assert torch.is_tensor(out)
    assert torch.isclose(out, loss)


# -------------------------
# Gradient Handling
# -------------------------


def test_loss_gradients_computed(engine, data, loss):
    grads = engine._calculate_loss_gradients(loss, data)

    assert "x" in grads
    assert torch.is_tensor(grads["x"])
    assert grads["x"].shape[0] == data["x"].shape[0]


def test_gradient_norm_clamped(engine, data):
    zero_loss = data["x"].sum() * 0.0
    grads = engine._calculate_loss_gradients(zero_loss, data)

    assert torch.all(grads["x"] >= engine.epsilon)


def test_error_if_variable_does_not_require_grad(engine, loss):
    bad_data = {"x": torch.tensor([[1.0], [2.0]], requires_grad=False)}

    with pytest.raises(RuntimeError, match="does not require gradients"):
        engine._calculate_loss_gradients(loss, bad_data)


def test_error_if_loss_not_dependent_on_variable(engine, data):
    unrelated_loss = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(RuntimeError, match="Unable to compute loss gradients"):
        engine._calculate_loss_gradients(unrelated_loss, data)


# -------------------------
# Aggregator Usage
# -------------------------


def test_aggregator_called(engine, data, loss, monkeypatch):
    spy = MagicMock(side_effect=engine.aggregator)
    monkeypatch.setattr(engine, "aggregator", spy)

    engine.train(data, loss)

    spy.assert_called()
