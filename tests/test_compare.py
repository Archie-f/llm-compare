from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.llm_compare.providers.base import LLMProvider, LLMResult
from src.llm_compare.compare import run_comparison, ComparisonResult, save_to_json, load_from_json


def generate_test_providers_list() -> list[LLMProvider]:
    providers: list[LLMProvider] = [MagicMock(spec=LLMProvider) for _ in range(3)]
    tokens: list[int] = [10, 20, 30]
    latencies: list[float] = [500.0, 200.0, 800.0]
    for i, (provider, token, latency) in enumerate(zip(providers, tokens, latencies), start=1):
        provider.ask.return_value = LLMResult(
            provider=f"test_provider-{i}",
            model="test_model",
            text=f"test_provider-{i} - test_text.",
            tokens_in=token,
            tokens_out=token,
            latency_ms=latency,
        )
    return providers

def test_run_comparison_collects_all_results() -> None:
    """Tests run_comparison_collects_all_results() using 3 fake providers."""
    providers = generate_test_providers_list()
    comparison_results = run_comparison(prompt='test_prompt', providers=providers)
    assert len(comparison_results.results) == 3

def test_best_cost_returns_cheapest() -> None:
    """Tests best_cost() returns the provider with the lowest cost_usd()."""
    providers = generate_test_providers_list()
    comparison_results = run_comparison(prompt='test_prompt', providers=providers)
    assert comparison_results.best_cost().provider == 'test_provider-1'

def test_save_to_json_creates_file() -> None:
    """Tests save_to_json_creates_file() using 3 fake providers."""
    providers: list[LLMProvider] = generate_test_providers_list()
    comparison_results: ComparisonResult = run_comparison(prompt='test_prompt', providers=providers)
    path: Path = save_to_json(comparison_results)
    assert path.exists() and path.is_file()

    c_results: ComparisonResult = load_from_json(path)
    assert c_results.prompt == comparison_results.prompt
    assert c_results.results == comparison_results.results
    assert c_results.time_stamp == comparison_results.time_stamp

def test_fastest_returns_lowest_latency() -> None:
    """Tests fastest_returns_lowest_latency()."""
    providers = generate_test_providers_list()
    comparison_results = run_comparison(prompt='test_prompt', providers=providers)
    assert comparison_results.fastest().latency_ms == 200.0

def test_best_cost_raises_on_empty() -> None:
    """Tests best_cost() raising value error when results list passed is empty."""
    comparison_results = ComparisonResult(prompt='test_prompt', results=[])
    with pytest.raises(ValueError):
        comparison_results.best_cost()
