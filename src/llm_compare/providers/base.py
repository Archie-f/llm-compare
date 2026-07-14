from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, Any


@dataclass
class LLMResult:
    """Unified result returned by every LLM provider."""
    provider: str  # 'claude', 'open_ai', 'ollama'
    model: str  # exact model string used
    text: str  # response text
    tokens_in: int  # input / prompt tokens
    tokens_out: int  # output / completion tokens
    latency_ms: float  # wall-clock time in milliseconds
    cost: float = field(default=0.0) # cost in usd
    judge_score: float | None = None
    judge_reason: str = ""

    def __post_init__(self) -> None:
        self.cost = self.cost_usd()

    def cost_usd(self) -> float:
        """Calculate cost in USD based on provider pricing.

        Returns: Calculated cost in USD for the provider"""
        cost_rates = {
            "claude": (1.00, 5.00),
            "open_ai": (0.15, 0.60),
            "ollama": (0.00, 0.00),
            "groq": (0.00, 0.00)
        }
        in_rate, out_rate = cost_rates.get(self.provider, (0, 0))
        return (self.tokens_in * in_rate + self.tokens_out * out_rate) / 1_000_000

class LLMProvider(ABC):
    """Abstract base class for LLM provider."""

    @abstractmethod
    def ask(self, user_input: str, system_prompt: str = '') -> LLMResult:
        """Send prompt to the LLM and return a unified LLMResult."""

    @abstractmethod
    def ask_stream(self, user_input: str, system_prompt: str = '') -> Generator[str, Any, LLMResult]:
        """Yield response text chunks as they arrive from the provider.

        Once the underlying stream is exhausted, the generator should make
        the equivalent LLMResult available to the caller (via the generator's
        return value, a filled-in mutable object, or another approach of your
        choosing) so streaming callers don't lose token/cost/latency tracking.
        """

class ProviderError(Exception):
    def __init__(
        self,
        provider_name: str,
        original_error: Exception,
        retryable: bool = False,
    ) -> None:
        self.provider_name = provider_name
        self.original_error = original_error
        self.retryable = retryable
        super().__init__(
            f"[{provider_name}] {type(original_error).__name__}: {original_error}"
        )