import dataclasses
import datetime
import json
from pathlib import Path
from typing import Optional

from .types import EvalCase, EvalResult
from .harness import run_eval

from ..providers.base import LLMProvider, PROVIDER_REGISTRY, Provider

AVG_OUTPUT_TOKENS = 200
RESULTS_DIR = Path(__file__).parent / "results"

def estimate_cost(dataset: list[EvalCase], providers: list[str]) -> float:
    """Rough cost estimate in USD before running the batch.

        Assumes ~200 output tokens per call and uses pricing for each provider.
    """
    total = 0.0

    for case in dataset:
        input_tokens = len(case.prompt.split()) * 1.3

        for provider in providers:
            if provider not in PROVIDER_REGISTRY:
                raise ValueError(f"Unknown provider: {provider!r}. Add it to PROVIDER_REGISTRY.")

            provider = Provider(provider)
            config = PROVIDER_REGISTRY[provider]
            input_price = input_tokens * config.input_price_per_million / 1_000_000
            output_price = AVG_OUTPUT_TOKENS * config.output_price_per_million / 1_000_000
            total += input_price + output_price

    return total

def run_batch(
    dataset: list[EvalCase],
    providers: dict[str, LLMProvider],
    judge: Optional[LLMProvider] = None,
    persist: bool = True,
    system_prompt: str = ""
) -> dict[str, list[EvalResult]]:
    """Run all cases against all providers. Returns results keyed by provider name.

    Args:
        dataset:   List of EvalCase objects to evaluate.
        providers: Dict mapping provider name → LLMProvider instance.
        judge:     Optional LLMProvider to use for LLM-as-judge scoring.
        persist:   If True, write results to results/batch_YYYY-MM-DD_HH-MM.json.
        system_prompt: System prompt passed to all providers (default: none).
    """
    all_results: dict[str, list[EvalResult]] = {}
    for provider_name, provider in providers.items():
        results = run_eval(
            cases=dataset,
            provider=provider,
            judge=judge,
            system_prompt=system_prompt,
        )
        all_results[provider_name] = results

    if persist:
        _persist(all_results)

    return all_results


def _persist(all_results: dict[str, list[EvalResult]]) -> None:
    """Write all_results to results/batch_YYYY-MM-DD_HH-MM.json as plain dicts."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = RESULTS_DIR / f"batch_{timestamp}.json"

    # Convert dataclasses to plain dicts for JSON serialization
    serializable = {
        name: [dataclasses.asdict(r) for r in results]
        for name, results in all_results.items()
    }
    path.write_text(json.dumps(serializable, indent=2))
    print(f"Results saved to {path}")
