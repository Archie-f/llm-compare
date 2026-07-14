import json

from .types import EvalCase, EvalResult

from ..providers.base import LLMProvider, LLMResult
from ..providers.anthropic_provider import AnthropicProvider

JUDGE_SYSTEM_PROMPT = '''
You are an impartial evaluator. You will be given:
1. A task description
2. A model output to evaluate

Score the output on a scale of 0 to 3:
  0 = Completely wrong or harmful
  1 = Partially correct but missing key information
  2 = Correct but could be clearer or more complete
  3 = Excellent: accurate, clear, and appropriately concise
If the respond includes any Error, return the score value as None.

Respond with ONLY a JSON object in this exact format:
  {"score": <0-3> or <None>, "reason": "<one sentence>"}
Do not add any other text.
'''

def build_judge_prompt(task: str, output: str, task_description: str = '') -> str:
    """Build the prompt sent to the judge model for scoring."""
    context = f"Task description: {task_description}\n\n" if task_description else ""
    return f"""{context}Input: {task}

Output to evaluate:
{output}

Score this output."""

def normalize_score(score: float, scale: int) -> float | None:
    """Normalize given score from a given scale to the 0.0–1.0 range."""
    if scale == 0:
        raise ValueError('Scale cannot be zero')

    if score is None:
        return None

    return score / scale

def clean_result(result: LLMResult) -> str:
    """Strip LLM response text before JSON parsing."""
    text_res = result.text.strip()
    if text_res.startswith("```"):
        text_res = text_res.replace("```", "")
        text_res = text_res.replace("json\n", "", 1)
    return text_res

def score_with_llm(
    case: EvalCase,
    actual: str,
    judge: LLMProvider,
) -> EvalResult:
    """Score an output using a second LLM as judge."""
    judge_prompt = build_judge_prompt(case.prompt, actual, case.task_description)
    raw_result = judge.ask(judge_prompt, JUDGE_SYSTEM_PROMPT, temperature=0.0) if isinstance(judge, AnthropicProvider) \
        else judge.ask(judge_prompt, JUDGE_SYSTEM_PROMPT)
    judge_result = clean_result(raw_result)

    try:
        judge_response = json.loads(judge_result.strip())
        score = judge_response["score"]
        reason = judge_response["reason"]
    except (json.JSONDecodeError, KeyError, ValueError):
        score, reason = None, f'Judge returned invalid JSON: {judge_result!r}'

    score_normalized = normalize_score(score, scale=3)
    return EvalResult(
        case=case,
        actual_output=actual,
        score=score_normalized,
        passed=score >= 2 if score is not None else False,
        reason=reason,
    )
