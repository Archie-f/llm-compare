import os
import time
from typing import Generator, Any

import anthropic
from anthropic.types import MessageParam, TextBlock
from dotenv import load_dotenv

from .base import LLMProvider, LLMResult, Provider, ProviderError

load_dotenv()

class AnthropicProvider(LLMProvider):
    """Anthropic Claude via the Anthropic SDK."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize Anthropic client and store model name."""
        self.client = anthropic.Anthropic()
        # model defaults resolved here, not in the signature, so a later
        # load_dotenv() call still takes effect (signature defaults are
        # bound once, at import time, and would freeze an unset env var).
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-sonnet-4-6"

    def ask(self, user_input: str, system_prompt: str = '', temperature: float = 0.7) -> LLMResult:
        """Call Anthropic messages.create and return a unified LLMResult.

        Args:
            user_input: The prompt to send to the model.
            system_prompt: Optional system instructions for the model.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            LLMResult with the response text and usage metrics.
        """
        prompt: list[MessageParam] = [
            {"role": "user", "content": user_input}
        ]
        start_time: float = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=system_prompt,
                messages=prompt,
                temperature=temperature,
            )
            elapsed_time = (time.perf_counter() - start_time) * 1000
            text_block = next(b for b in response.content if isinstance(b, TextBlock))
            return LLMResult(
                provider=Provider.claude,
                model=self.model,
                text=text_block.text,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                latency_ms=round(elapsed_time),
            )
        except anthropic.RateLimitError as e:
            raise ProviderError(
                provider_name=Provider.claude,
                original_error=e,
                retryable=True,
            ) from e
        except anthropic.APITimeoutError as e:
            raise ProviderError(
                provider_name=Provider.claude,
                original_error=e,
                retryable=True,
            ) from e
        except (KeyError, AttributeError, StopIteration) as e:
            raise ProviderError(
                provider_name=Provider.claude,
                original_error=e,
                retryable=False,
            ) from e

    def ask_stream(self, user_input: str, system_prompt: str = '') -> Generator[str, Any, LLMResult]:
        """Yield response text chunks as they arrive from the provider."""
        prompt: list[MessageParam] = [
            {"role": "user", "content": user_input}
        ]
        response: str = ""
        start_time: float = time.perf_counter()
        try:
            with self.client.messages.stream(
                    model=self.model,
                    max_tokens=256,
                    system=system_prompt,
                    messages=prompt,
            ) as stream:
                for text in stream.text_stream:
                    response += text
                    yield text
                final_message = stream.get_final_message()
            elapsed_time = (time.perf_counter() - start_time) * 1000
            return LLMResult(
                provider=Provider.claude,
                model=self.model,
                text=response,
                tokens_in=final_message.usage.input_tokens,
                tokens_out=final_message.usage.output_tokens,
                latency_ms=round(elapsed_time),
            )
        except anthropic.RateLimitError as e:
            raise ProviderError(
                provider_name=Provider.claude,
                original_error=e,
                retryable=True,
            ) from e
        except anthropic.APITimeoutError as e:
            raise ProviderError(
                provider_name=Provider.claude,
                original_error=e,
                retryable=True,
            ) from e
        except (KeyError, AttributeError, StopIteration) as e:
            raise ProviderError(
                provider_name=Provider.claude,
                original_error=e,
                retryable=False,
            ) from e