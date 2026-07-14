from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float

PROVIDER_PRICING: dict[str, TokenPricing] = {
    "claude-haiku": TokenPricing(1.00, 5.00),
    "claude-sonnet": TokenPricing(3.00, 15.00),
    "claude-opus": TokenPricing(5.00, 25.00),
    "gpt-4o-mini": TokenPricing(0.15, 0.60),
    "ollama": TokenPricing(0.00, 0.00),
}

def estimate_cost(input_tokens: int, output_tokens: int, provider: str) -> float:
    """Return the estimated USD cost of a call, given token counts."""
    price = PROVIDER_PRICING[provider]
    input_cost = input_tokens * price.input_per_million / 1_000_000
    output_cost = output_tokens * price.output_per_million / 1_000_000
    return round(input_cost + output_cost, 6)

def estimate_batch_cost(
        num_cases: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        providers: list[str],
) -> dict[str, float]:
    """Estimate total cost of running num_cases through each provider.

    Returns a mapping of provider -> estimated total USD cost.
    """
    return {
        provider: estimate_cost(
            input_tokens=avg_input_tokens * num_cases,
            output_tokens=avg_output_tokens * num_cases,
            provider=provider,
        ) for provider in providers
    }

def count_tokens(text: str, provider: str = "anthropic") -> int:
    """Estimate the token count of a string for a given provider.

    Uses a character-based heuristic (~4 chars/token for English)
    as a fast local approximation. For exact counts, providers
    expose real tokenizers (see Quick Reference Card).
    """
    #TODO Update the function
    chars_per_token = 4
    return max(1, len(text) // chars_per_token)

def build_batch_estimate(texts: list[str], provider: str) -> int:
    """Calculates the total estimated token count across all texts in the list"""
    return sum(count_tokens(text, provider) for text in texts)

