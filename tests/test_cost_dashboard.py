import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm_compare.cost_dashboard import log_run, summarize
from src.llm_compare.providers.base import LLMResult


results = [
        LLMResult(provider="open_ai", model="gpt-4o-mini", text="Hello! How can I help you today?", tokens_in=8,
                  tokens_out=9, latency_ms=1207),
        LLMResult(provider="open_ai", model="gpt-4o-mini", text="The weather is sunny.", tokens_in=6, tokens_out=5,
                  latency_ms=1235),
        LLMResult(provider="open_ai", model="gpt-4o-mini", text="Python is a programming language.", tokens_in=7,
                  tokens_out=6, latency_ms=2451),
        LLMResult(provider="claude", model="claude-haiku-4-5", text="Sure, here is the recipe for pancakes.",
                  tokens_in=12, tokens_out=45, latency_ms=1093),
        LLMResult(provider="claude", model="claude-haiku-4-5", text="I cannot fulfill this request.", tokens_in=10,
                  tokens_out=6, latency_ms=2213),
        LLMResult(provider="ollama", model="llama3", text="That is correct.", tokens_in=4, tokens_out=3,
                  latency_ms=3934),
        LLMResult(provider="ollama", model="llama3", text="Paris is the capital of France.", tokens_in=6, tokens_out=7,
                  latency_ms=6362),
        LLMResult(provider="ollama", model="llama3", text="2 + 2 equals 4.", tokens_in=5, tokens_out=5,
                  latency_ms=2387),
        LLMResult(provider="ollama", model="llama3", text="Goodbye!", tokens_in=2, tokens_out=2, latency_ms=1126),
    ]

def test_log_run_appends_one_valid_json_line_per_call(tmp_path: Path) -> None:
    """log_run() should append one independently-parseable JSON line per call,
    with the LLMResult's fields (provider, tokens_in, tokens_out, cost,
    latency_ms) plus a timestamp — and never overwrite previous lines.
    """
    file_path = tmp_path / "test_log_run.jsonl"
    for result in results:
        log_run(result, file_path)

    with file_path.open() as f:
        entries = [json.loads(line) for line in f]

    assert len(entries) == len(results)

def test_summarize_returns_correct_totals_per_provider(tmp_path: Path) -> None:
    """summarize() should group logged calls by provider and return the
    correct calls / total_cost / avg_latency_ms for each — computed from
    known, hand-calculable LLMResult values, not live API output.
    """
    file_path = tmp_path / "test_log_run.jsonl"
    for result in results:
        log_run(result, file_path)

    entries = summarize(file_path)
    assert len(entries) == 3

    open_ai_provider = entries["open_ai"]
    assert open_ai_provider["calls"] == 3
    assert open_ai_provider["total_cost"] == 1.5e-05
    assert open_ai_provider["avg_latency_ms"] == 1631.0

    claude_provider = entries["claude"]
    assert claude_provider["calls"] == 2
    assert claude_provider["total_cost"] == 0.000277
    assert claude_provider["avg_latency_ms"] == 1653.0

    ollama_provider = entries["ollama"]
    assert ollama_provider["calls"] == 4
    assert ollama_provider["total_cost"] == 0.0
    assert ollama_provider["avg_latency_ms"] == 3452.2
