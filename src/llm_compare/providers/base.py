import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generator, Any, NoReturn
import dotenv
dotenv.load_dotenv()


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
        try:
            provider = Provider(self.provider)
        except KeyError:
            return 0.0
        config = PROVIDER_REGISTRY[provider]
        return (self.tokens_in * config.input_price_per_million
                + self.tokens_out * config.output_price_per_million) / 1_000_000

class LLMProvider(ABC):
    """Abstract base class for LLM provider."""

    @abstractmethod
    def ask(self, user_input: str, system_prompt: str = '') -> LLMResult:
        """Send prompt to the LLM and return a unified LLMResult."""

    @abstractmethod
    def ask_stream(self, user_input: str, system_prompt: str = '') -> Generator[str, Any, LLMResult]:
        """Yield response text chunks, then return the final LLMResult
        (via the generator's return value) once the stream ends."""

class Provider(StrEnum):
    """Canonical provider identifiers, used anywhere a provider name is
    looked up, stored, or compared."""
    claude = "claude"
    open_ai = "open_ai"
    ollama = "ollama"
    groq = "groq"

    @classmethod
    def _missing_(cls, value: object) -> NoReturn:
        """Raise KeyError (instead of the default ValueError) for an unknown value."""
        raise KeyError(f"{value!r} is not a valid Provider")

@dataclass(frozen=True)
class ProviderConfig:
    """Metadata for one provider: identity, default model, pricing, and
    display info. Single source of truth, keyed by Provider in PROVIDER_REGISTRY."""
    name: Provider
    default_model: str
    input_price_per_million: float
    output_price_per_million: float
    label: str
    color: str

claude_model = os.getenv("ANTHROPIC_MODEL_NAME") or "claude-sonnet-4-6"
openai_model = os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
groq_model = os.getenv("GROQ_MODEL_NAME") or "llama-3.1-8b-instant"
ollama_model = os.getenv("OLLAMA_MODEL_NAME") or "llama3"

PROVIDER_REGISTRY: dict[Provider, ProviderConfig] = {
    Provider.claude: ProviderConfig(
        name=Provider.claude,
        default_model=claude_model,
        input_price_per_million=1.00,
        output_price_per_million=5.00,
        label=f"Claude-{claude_model}",
        color="#2E7FE0",
    ),
    Provider.open_ai: ProviderConfig(
        name=Provider.open_ai,
        default_model=openai_model,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        label=f"Open-Ai-{openai_model}",
        color="#1DB876",
    ),
    Provider.groq: ProviderConfig(
        name=Provider.groq,
        default_model=groq_model,
        input_price_per_million=0.00,
        output_price_per_million=0.00,
        label=f"Groq-{groq_model}",
        color="#F5A623",
    ),
    Provider.ollama: ProviderConfig(
        name=Provider.ollama,
        default_model=ollama_model,
        input_price_per_million=0.00,
        output_price_per_million=0.00,
        label=f"Ollama-{ollama_model}",
        color="#6C5CE7",
    )
}

def provider_label(provider: str) -> str:
    """Display label for a provider; falls back to the raw string if unknown."""
    try:
        return PROVIDER_REGISTRY[Provider(provider)].label
    except KeyError:
        return provider

def provider_color(provider: str) -> str:
    """Display color for a provider; falls back to neutral grey if unknown."""
    try:
        return PROVIDER_REGISTRY[Provider(provider)].color
    except KeyError:
        return "#888888"

class ProviderError(Exception):
    """Raised when a provider call fails; carries the provider name, the
    original exception, and whether the failure is worth retrying."""

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