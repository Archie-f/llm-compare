from typing import Any, Generator, cast

from .providers.base import LLMResult

def print_stream_and_collect_result(generator: Generator[str, Any, LLMResult]) -> LLMResult:
    try:
        while True:
            print(next(generator), end="", flush=True)
    except StopIteration as end:
        result = cast(LLMResult, end.value)
        print("\n----------------------------------------")
        print(f" * Response: \n{result.text}", flush=True)
        print(f" * Tokens In: {result.tokens_in}", flush=True)
        print(f" * Tokens Out: {result.tokens_out}", flush=True)
        print(f" * Elapsed Time: {result.latency_ms}", flush=True)
        return result
