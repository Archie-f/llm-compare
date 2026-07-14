from .types import EvalCase, EvalResult

def normalize(text: str) -> str:
    """Normalize a string by cleaning noise symbols and changing to lowercase."""
    noise_symbols = " .!?'\"`\n$"
    return text.strip(noise_symbols).lower()

def score_exact(case: EvalCase, actual: str) -> EvalResult:
    """Score by exact string equality (case-insensitive, stripped)."""
    cleaned_expected = normalize(case.expected)
    cleaned_actual = normalize(actual)
    is_match = cleaned_expected == cleaned_actual

    return EvalResult(
        case=case,
        actual_output=cleaned_actual,
        score=1.0 if is_match else 0.0,
        passed=is_match,
        reason='Exact match' if is_match else f'Expected: {cleaned_expected!r}, got {cleaned_actual!r}',
    )
