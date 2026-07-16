import pytest

from src.llm_compare.token_utils import estimate_cost

def test_estimate_cost_returns_correct_dollar_amount() -> None:
    """estimate_cost() should return the exact sum of input + output cost for known token counts."""
    actual_cost = estimate_cost(1_000_000, 1_000_000, "claude")
    expected_cost = 6.0
    assert actual_cost == expected_cost

def test_estimate_cost_unknown_provider_raises_key_error() -> None:
    """estimate_cost() should raise KeyError for a provider not in PROVIDER_PRICING."""
    with pytest.raises(KeyError):
        estimate_cost(1_000_000, 1_000_000, "cloud-haiku")
