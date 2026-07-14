import time
from typing import Generator, Any

import openai
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam, ChatCompletionStreamOptionsParam
)

from .base import LLMProvider, LLMResult, ProviderError


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = 'gpt-4o-mini') -> None:
        """Initialize OpenAI client and store model name."""
        self.client = OpenAI()
        self.model = model

    def ask(self, user_input: str, system_prompt: str = '') -> LLMResult:
        """Call OpenAI chat completions and return unified LLMResult.

            Args:
                user_input: Input to ask user to enter chat.
                system_prompt: Input to ask user to enter chat.

            Returns:
                LLMResult: Result of asking user to enter chat.
        """
        system_turn: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt
        }
        user_turn: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_input
        }
        prompt: list[ChatCompletionMessageParam] = [system_turn, user_turn]
        start_time: float = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=prompt,
            )
            elapsed_time: float = (time.perf_counter() - start_time) * 1000
            assert response.choices[0].message.content is not None
            assert response.usage is not None
            return LLMResult(
                provider='open_ai',
                model=self.model,
                text=response.choices[0].message.content,
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                latency_ms=round(elapsed_time),
            )
        except openai.RateLimitError as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=True
            ) from e
        except openai.APITimeoutError as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=True
            ) from e
        except (KeyError, AttributeError) as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=False
            ) from e

    def ask_stream(self, user_input: str, system_prompt: str = '') -> Generator[str, Any, LLMResult]:
        """Yield response text chunks as they arrive from OpenAI's streaming API."""
        system_turn: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt
        }
        user_turn: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_input
        }
        prompt: list[ChatCompletionMessageParam] = [system_turn, user_turn]
        response_text: str = ""
        tokens_in: int = 0
        tokens_out: int = 0
        start_time: float = time.perf_counter()
        stream_options: ChatCompletionStreamOptionsParam = {"include_usage": True}
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=prompt,
                stream=True,
                stream_options=stream_options,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    response_text += text
                    yield text
                if chunk.usage is not None:
                    tokens_in = chunk.usage.prompt_tokens
                    tokens_out = chunk.usage.completion_tokens
            elapsed_time: float = (time.perf_counter() - start_time) * 1000
            return LLMResult(
                provider='open_ai',
                model=self.model,
                text=response_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=round(elapsed_time),
            )
        except openai.RateLimitError as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=True
            ) from e
        except openai.APITimeoutError as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=True
            ) from e
        except (KeyError, AttributeError) as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=False
            ) from e
