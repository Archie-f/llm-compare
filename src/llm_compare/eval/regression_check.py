import json


def load_baseline(path: str) -> dict[str, float]:
    """Load a baseline dict[str, float] of previously recorded pass rates."""
    with open(path) as f:
        data: dict[str, float] = json.load(f)
        return data

def check_regression(
    baseline: dict[str, float],
    current: dict[str, float],
    tolerance: float = 0.05,
) -> list[str]:
    """Return keys whose pass rate dropped by more than `tolerance`
    relative to baseline. Empty list means no regressions."""
    failed: list[str] = []
    for key in baseline:
        if key in current:
            drop = baseline[key] - current[key]
            if drop > tolerance:
                failed.append(key)
    return failed


