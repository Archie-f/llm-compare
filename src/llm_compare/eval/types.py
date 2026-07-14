from dataclasses import dataclass

@dataclass
class EvalCase:
    """One row in the eval dataset."""
    prompt: str
    expected: str
    category: str = ''
    task_description: str = ''

@dataclass
class EvalResult:
    """The outcome of scoring one case."""
    case: EvalCase
    actual_output: str
    score: float | None
    passed: bool
    reason: str = ''
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    latency_ms: float | None = None
