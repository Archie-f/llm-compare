import dataclasses
import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from .providers.base import LLMProvider, LLMResult, ProviderError
from .eval.judge import score_with_llm
from .eval.types import EvalCase
from .cost_dashboard import log_run
from .retry_backoff import retry_with_backoff
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider
from .providers.groq_provider import GroqProvider

TRUNCATE_AT: int = 60
RESULTS_DIR: Path = Path('results')

@dataclass
class ComparisonResult:
    """Comparison of the results returned by each LLM provider."""
    prompt: str
    results: list[LLMResult] = field(default_factory=list)
    time_stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def best_cost(self) -> LLMResult:
        """Get the result that costs least."""
        if not self.results:
            raise ValueError("No results to compare.")
        return min(self.results, key=lambda x: x.cost_usd())

    def fastest(self) -> LLMResult:
        """Get the result that returns fastest."""
        return min(self.results, key=lambda x: x.latency_ms)

    def winner(self) -> LLMResult:
        """Get the result that's both the cheapest and fastest."""
        max_cost = max(r.cost_usd() for r in self.results) or 1.0
        max_latency = max(r.latency_ms for r in self.results) or 1.0

        return min(self.results, key=lambda r: (r.cost_usd() / max_cost) + (r.latency_ms / max_latency))

def run_comparison(
        prompt: str,
        providers: list[LLMProvider],
        system_prompt: str = '',
        judge: LLMProvider | None = None,
) -> ComparisonResult:
    """Run prompt across all providers and return a ComparisonResult."""
    comparison_results = ComparisonResult(prompt=prompt)
    for provider in providers:
        try:
            result = retry_with_backoff(lambda: provider.ask(user_input=prompt, system_prompt=system_prompt))
        except ProviderError as e:
            result = LLMResult(
                provider=e.provider_name,
                model='unknown',
                text='',
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                judge_score=None,
                judge_reason=f"ERROR: {e}",
            )

        if judge is not None:
            task_description: str = "Evaluate the quality of the given response to the given prompt."
            case = EvalCase(prompt=prompt, expected='', task_description=task_description)
            try:
                eval_result = score_with_llm(
                    case=case,
                    actual=result.text,
                    judge=judge,
                )
                result.judge_score = eval_result.score
                result.judge_reason = eval_result.reason
            except Exception as e:
                result.judge_score, result.judge_reason = None, f'Error: {e}'

        log_run(result)
        comparison_results.results.append(result)

    return comparison_results

def truncate(text: str, length: int) -> str:
    """Truncate a string to the given length."""
    return f"{text[:length]}.." if len(text) > length else text

def print_table(comparison: ComparisonResult) -> None:
    """Print a formatted comparison table to stdout."""
    has_judge_score: bool = any(result.judge_score is not None for result in comparison.results)
    header: str = f"{'Provider':<20} | {'Response':<65} | {'In':<5} | {'Out':<5} | {'Cost ($)':<10} | {'Latency (ms)':<13}"
    if has_judge_score:
        header += f" | {'Judge Score':<13} | {'Judge Reason':<65}"
    
    divider: str = '-' * len(header)
    winner: LLMResult = comparison.winner()

    print(f"Prompt: {comparison.prompt}")
    print(divider)
    print(header)
    print(divider)

    for result in comparison.results:
        row: str = (f"{result.provider:<20} | {truncate(result.text, TRUNCATE_AT):<65} | {result.tokens_in:<5} | "
              f"{result.tokens_out:<5} | {result.cost_usd():<10.6f} | {result.latency_ms:<13}")
        if has_judge_score:
            row += f" | {'None':<13} | {truncate(result.judge_reason, TRUNCATE_AT):<65}" if result.judge_score is None \
                else f" | {result.judge_score:<13.2f} | {truncate(result.judge_reason, TRUNCATE_AT):<65}"
        print(row)
    print(divider)

    fastest: LLMResult = comparison.fastest()
    cheapest: LLMResult = comparison.best_cost()
    print(f"⚡ Fastest: {fastest.provider} ({fastest.latency_ms} ms), 💰 Cheapest: {cheapest.provider} (${cheapest.cost_usd():.6f})")
    print(divider)
    print(f"WINNER: {winner.provider} with Latency of {winner.latency_ms} ms and Cost of ${winner.cost_usd():.6f}")
    print(divider)

def save_to_json(comparison: ComparisonResult) -> Path:
    """Save a ComparisonResult to a timestamped JSON file. Returns the file path."""
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = RESULTS_DIR / f"{timestamp}.json"
    data = dataclasses.asdict(comparison)

    filename.write_text(json.dumps(data, indent=2))
    return filename

def load_from_json(path: Path) -> ComparisonResult:
    """Load a ComparisonResult back from a JSON file saved by save_to_json()."""
    data = json.loads(path.read_text())
    data['results'] = [LLMResult(**res) for res in data['results']]
    return ComparisonResult(**data)

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    inp: str = "Explain what an API is in one sentence."
    system_message: str = 'You are a helpful assistant.'
    providers_list: list[LLMProvider] = [
        AnthropicProvider(),
        OpenAIProvider(),
        GroqProvider(),
        OllamaProvider(),
    ]

    results: ComparisonResult = run_comparison(inp, providers_list, system_message, GroqProvider())
    print_table(results)
    print()

    path_to_json_file = save_to_json(results)
    comparison_result: ComparisonResult = load_from_json(path_to_json_file)
    print(comparison_result)
