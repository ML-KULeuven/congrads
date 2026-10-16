from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from congrads.constraints.base import Constraint
from congrads.core.constraint_engine import ConstraintEngine
from congrads.descriptor import Descriptor

# -------------------------
# Dummy / Fake Components
# -------------------------


class DummyConstraint(Constraint):
    def __init__(
        self,
        tag="x",
        name="C1",
        enforce=True,
        rescale_factor=1.0,
    ):
        super().__init__(tags={tag}, name=name, enforce=enforce, rescale_factor=rescale_factor)

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
    descriptor = Descriptor()
    descriptor.add_layer("x")
    descriptor.add_tag("x", "x", 0)
    return descriptor


@pytest.fixture
def metric_manager():
    return DummyMetricManager()


@pytest.fixture
def aggregator():
    # Simple sum aggregator
    return lambda x: x.sum()


@pytest.fixture
def constraint(descriptor, device):
    DummyConstraint.descriptor = descriptor
    DummyConstraint.device = device
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


def test_error_if_variable_does_not_require_grad(engine, loss):
    bad_data = {"x": torch.tensor([[1.0], [2.0]], requires_grad=False)}

    with pytest.raises(RuntimeError, match="does not require gradients"):
        engine._calculate_loss_gradients(loss, bad_data)


def test_error_if_loss_not_dependent_on_variable(engine, data):
    unrelated_loss = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(RuntimeError, match="Unable to compute"):
        engine._calculate_loss_gradients(unrelated_loss, data)


# -------------------------
# Aggregator Usage
# -------------------------


def test_aggregator_called(engine, data, loss, monkeypatch):
    spy = MagicMock(side_effect=engine.aggregator)
    monkeypatch.setattr(engine, "aggregator", spy)

    engine.train(data, loss)

    spy.assert_called()


# -------------------------
# Gradient Tracking & Lazy Evaluation
# -------------------------


def test_check_constraint_runs_without_grad(engine, constraint, data, loss, monkeypatch):
    grad_enabled = []
    original = constraint.check_constraint

    def recording_check(d):
        grad_enabled.append(torch.is_grad_enabled())
        return original(d)

    monkeypatch.setattr(constraint, "check_constraint", recording_check)

    engine.train(data, loss)
    engine.validate(data, loss)

    assert grad_enabled == [False, False]


def test_directions_and_rescale_only_computed_when_enforcing(
    engine, constraint, data, loss, monkeypatch
):
    direction_spy = MagicMock(side_effect=constraint.calculate_direction)
    rescale_spy = MagicMock(side_effect=constraint.compute_rescale_factor)
    monkeypatch.setattr(constraint, "calculate_direction", direction_spy)
    monkeypatch.setattr(constraint, "compute_rescale_factor", rescale_spy)

    engine.validate(data, loss)
    engine.test(data, loss)
    direction_spy.assert_not_called()
    rescale_spy.assert_not_called()

    constraint.enforce = False
    engine.train(data, loss)
    direction_spy.assert_not_called()
    rescale_spy.assert_not_called()

    constraint.enforce = True
    engine.train(data, loss)
    direction_spy.assert_called_once()
    rescale_spy.assert_called_once()


def test_loss_gradients_only_computed_when_enforcing(engine, data, loss, monkeypatch):
    spy = MagicMock(side_effect=engine._calculate_loss_gradients)
    monkeypatch.setattr(engine, "_calculate_loss_gradients", spy)

    engine.validate(data, loss)
    engine.test(data, loss)
    spy.assert_not_called()

    engine.enforce_all = False
    engine.train(data, loss)
    spy.assert_not_called()

    engine.enforce_all = True
    engine.train(data, loss)
    spy.assert_called_once()


# -------------------------
# Multi-layer Constraints
# -------------------------


class MultiLayerConstraint(DummyConstraint):
    def __init__(self, tags, name="ML", rescale_factor=2.0):
        Constraint.__init__(self, tags=tags, name=name, enforce=True, rescale_factor=rescale_factor)

    def calculate_direction(self, data):
        return {layer: torch.ones_like(data[layer]) for layer in self.layers}


@pytest.fixture
def two_layer_descriptor():
    descriptor = Descriptor()
    descriptor.add_layer("x")
    descriptor.add_layer("y")
    descriptor.add_tag("x", "x", 0)
    descriptor.add_tag("y", "y", 0)
    return descriptor


def make_engine(constraints, descriptor, device, aggregator):
    return ConstraintEngine(
        constraints=constraints,
        descriptor=descriptor,
        metric_manager=None,
        device=device,
        epsilon=1e-6,
        aggregator=aggregator,
        enforce_all=True,
    )


def test_rescale_factor_computed_once_per_constraint(
    two_layer_descriptor, device, aggregator, monkeypatch
):
    MultiLayerConstraint.descriptor = two_layer_descriptor
    MultiLayerConstraint.device = device
    constraint = MultiLayerConstraint(tags={"x", "y"})
    assert constraint.layers == {"x", "y"}

    spy = MagicMock(side_effect=constraint.compute_rescale_factor)
    monkeypatch.setattr(constraint, "compute_rescale_factor", spy)

    engine = make_engine([constraint], two_layer_descriptor, device, aggregator)

    x = torch.tensor([[1.0], [2.0]], requires_grad=True)
    y = torch.tensor([[3.0], [4.0]], requires_grad=True)
    loss = (x**2).mean() + (y**2).mean()

    out = engine.train({"x": x, "y": y}, loss)

    spy.assert_called_once()
    assert out.item() != pytest.approx(loss.item())


def test_multidim_layer_broadcasts_loss_gradient_norm(device, aggregator):
    descriptor = Descriptor()
    descriptor.add_layer("img")
    descriptor.add_tag("img", "img")

    class ImageConstraint(DummyConstraint):
        def __init__(self):
            Constraint.__init__(self, tags={"img"}, name="img", rescale_factor=2.0)

        def check_constraint(self, data):
            checks = torch.tensor([1.0, 0.0]).view(2, 1, 1)
            return checks, torch.ones_like(checks)

        def calculate_direction(self, data):
            return {"img": torch.ones_like(data["img"])}

    ImageConstraint.descriptor = descriptor
    ImageConstraint.device = device
    engine = make_engine([ImageConstraint()], descriptor, device, aggregator)

    img = torch.arange(24, dtype=torch.float32).view(2, 3, 4).requires_grad_()
    loss = (img**2).mean()

    out = engine.train({"img": img}, loss)

    # Only the violated second sample contributes: sum(img[1]) * rescale * ||dL/dimg[1]||
    grad_norm = (2 * img[1] / img.numel()).norm()
    expected = loss + img[1].sum() * 2.0 * grad_norm
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


# -------------------------
# gradients_from Handling
# -------------------------


class ColumnConstraint(DummyConstraint):
    """Constraint on a single [B, 1] layer: first sample satisfied, second violated."""

    def __init__(self, tag):
        Constraint.__init__(self, tags={tag}, name=tag, rescale_factor=2.0)
        self.tag = tag

    def check_constraint(self, data):
        checks = torch.tensor([[1.0], [0.0]])
        return checks, torch.ones_like(checks)

    def calculate_direction(self, data):
        return {self.tag: torch.ones_like(data[self.tag])}


@pytest.fixture
def gradients_from_descriptor():
    descriptor = Descriptor()
    descriptor.add_layer("h")
    descriptor.add_layer("y", affects_loss=False, gradients_from="h")
    descriptor.add_tag("h", "h", 0)
    descriptor.add_tag("y", "y", 0)
    return descriptor


@pytest.fixture
def gradients_from_data():
    h = torch.tensor([[1.0], [2.0]], requires_grad=True)
    return {"h": h, "y": 2 * h}


def test_gradients_from_corrects_direction_norm(
    gradients_from_descriptor, gradients_from_data, device, aggregator, monkeypatch
):
    ColumnConstraint.descriptor = gradients_from_descriptor
    ColumnConstraint.device = device
    constraint = ColumnConstraint("y")

    spy = MagicMock(side_effect=constraint.correct_direction_norm)
    monkeypatch.setattr(constraint, "correct_direction_norm", spy)

    engine = make_engine([constraint], gradients_from_descriptor, device, aggregator)

    data = gradients_from_data
    loss = (data["h"] ** 2).mean()
    out = engine.train(data, loss)

    spy.assert_called_once()
    assert spy.call_args.kwargs["output_layer"] == "y"
    assert spy.call_args.kwargs["gradient_layer"] == "h"
    assert out.requires_grad

    # Second sample is violated: y=4, corrected direction=1/2, rescale=2, ||dL/dh||=|h|=2
    assert out.item() == pytest.approx(loss.item() + 4.0 * 0.5 * 2.0 * 2.0)


def test_gradients_from_uses_constraint_override(
    gradients_from_descriptor, gradients_from_data, device, aggregator
):
    class ScaledCorrectionConstraint(ColumnConstraint):
        def correct_direction_norm(self, direction, output_layer, gradient_layer, data):
            return direction * 3.0

    ScaledCorrectionConstraint.descriptor = gradients_from_descriptor
    ScaledCorrectionConstraint.device = device
    constraint = ScaledCorrectionConstraint("y")

    engine = make_engine([constraint], gradients_from_descriptor, device, aggregator)

    data = gradients_from_data
    loss = (data["h"] ** 2).mean()
    out = engine.train(data, loss)

    # Second sample is violated: y=4, overridden direction=3, rescale=2, ||dL/dh||=|h|=2
    assert out.item() == pytest.approx(loss.item() + 4.0 * 3.0 * 2.0 * 2.0)


def test_direction_not_corrected_without_gradients_from(
    engine, constraint, data, loss, monkeypatch
):
    spy = MagicMock(side_effect=constraint.correct_direction_norm)
    monkeypatch.setattr(constraint, "correct_direction_norm", spy)

    engine.train(data, loss)

    spy.assert_not_called()


def test_error_if_no_loss_gradients_for_layer(device, aggregator):
    descriptor = Descriptor()
    descriptor.add_layer("h")
    descriptor.add_layer("y", affects_loss=False)
    descriptor.add_tag("y", "y", 0)

    ColumnConstraint.descriptor = descriptor
    ColumnConstraint.device = device
    constraint = ColumnConstraint("y")

    engine = make_engine([constraint], descriptor, device, aggregator)

    h = torch.tensor([[1.0], [2.0]], requires_grad=True)
    data = {"h": h, "y": 2 * h}
    loss = (h**2).mean()

    with pytest.raises(RuntimeError, match="Cannot get gradients for layer 'y'"):
        engine.train(data, loss)
