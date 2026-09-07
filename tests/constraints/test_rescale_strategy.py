import pytest
import torch

from congrads.constraints.rescale_strategy import (
    ImplicationBalancedStrategy,
    StaticStrategy,
)


# Sample dummy batch data
@pytest.fixture
def dummy_batch():
    data = {"x": torch.randn(4, 3), "y": torch.randn(4, 3)}
    checks = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
    directions = {"x": torch.ones_like(data["x"])}
    loss = torch.tensor(1.0)
    return data, checks, mask, directions, loss


def test_static_strategy_value(dummy_batch):
    data, checks, mask, directions, loss = dummy_batch
    value = 2.5
    strat = StaticStrategy(value)

    # compute() should return the fixed value
    result = strat.compute(data, checks, mask, directions, loss)
    assert result == value

    # compute_modifier() should always return 1
    modifier = strat.compute_modifier(data, checks, mask, directions, loss)
    assert modifier == 1.0


def test_implication_balanced_modifier(dummy_batch):
    data, checks, mask, directions, loss = dummy_batch
    strat = ImplicationBalancedStrategy(1.0)

    modifier = strat.compute_modifier(data, checks, mask, directions, loss)

    # Expected: total elements / sum of mask
    expected = mask.numel() / mask.sum().clamp_min(1.0)
    assert modifier == expected


def test_chaining_static_and_implication(dummy_batch):
    data, checks, mask, directions, loss = dummy_batch

    base = StaticStrategy(2.0)
    strat = ImplicationBalancedStrategy(base)

    result = strat.compute(data, checks, mask, directions, loss)

    expected_modifier = mask.numel() / mask.sum().clamp_min(1.0)
    expected_result = 2.0 * expected_modifier
    assert result == expected_result


def test_nested_chaining(dummy_batch):
    """Test multiple chained strategies."""
    data, checks, mask, directions, loss = dummy_batch

    base = StaticStrategy(1.5)
    chain = ImplicationBalancedStrategy(ImplicationBalancedStrategy(base))

    result = chain.compute(data, checks, mask, directions, loss)

    # Compute expected manually
    first_modifier = mask.numel() / mask.sum().clamp_min(1.0)
    expected = 1.5 * first_modifier * first_modifier
    assert result == expected


def test_invalid_base_type():
    """Passing a non-number/non-strategy as base should raise ValueError."""
    with pytest.raises(ValueError):
        ImplicationBalancedStrategy(base="invalid")


def test_tensor_compatibility(dummy_batch):
    """Ensure compute returns a tensor-like result if loss is tensor."""
    data, checks, mask, directions, loss = dummy_batch

    base = StaticStrategy(2.0)
    strat = ImplicationBalancedStrategy(base)

    result = strat.compute(data, checks, mask, directions, loss)

    # Result should be numeric type compatible with PyTorch
    assert isinstance(result, (int, float, torch.Tensor))


def test_edge_case_zero_mask(dummy_batch):
    """Test behavior when mask.sum() == 0."""
    data, checks, mask, directions, loss = dummy_batch
    mask = torch.zeros_like(mask)

    strat = ImplicationBalancedStrategy(StaticStrategy(1.0))
    modifier = strat.compute_modifier(data, checks, mask, directions, loss)

    # Should not divide by zero, minimum clamp ensures 1.0
    assert modifier == mask.numel() / 1.0
