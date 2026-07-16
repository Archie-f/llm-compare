from .providers.base import PROVIDER_REGISTRY, Provider


def estimate_cost(input_tokens: int, output_tokens: int, provider: str) -> float:
    """Return the estimated USD cost of a call, given token counts."""
    provider = Provider(provider)
    config = PROVIDER_REGISTRY[provider]
    input_cost = input_tokens * config.input_price_per_million / 1_000_000
    output_cost = output_tokens * config.output_price_per_million / 1_000_000
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

def count_tokens(text: str) -> int:
    """Estimate the token count of a string.

    Uses a character-based heuristic (~4 chars/token for English)
    as a fast local approximation.
    """
    chars_per_token = 4
    return max(1, len(text) // chars_per_token)

def build_batch_estimate(texts: list[str]) -> int:
    """Calculates the total estimated token count across all texts in the list"""
    return sum(count_tokens(text) for text in texts)

