import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm_compare.eval.types import EvalCase, EvalResult
from src.llm_compare.eval.dataset import Category
from src.llm_compare.eval.report import get_pass_rates_per_category, get_pass_rates_per_provider
from src.llm_compare.eval.regression_check import check_regression


def make_result(provider_category: str, passed: bool) -> EvalResult:
    """Build a minimal EvalResult for a given category, ignoring provider (tests group by category/provider separately)."""
    case = EvalCase(prompt="p", expected="e", category=provider_category)
    return EvalResult(case=case, actual_output="a", score=1.0 if passed else 0.0, passed=passed)


def test_pass_rates_per_category_and_provider():
    all_results: dict[str, list[EvalResult]] = {
        "claude": [
            make_result(Category.factual, True),
            make_result(Category.factual, False),
            make_result(Category.sentiment, True),
        ],
        "groq": [
            make_result(Category.factual, True),
            make_result(Category.sentiment, False),
        ],
    }

    category_rates = get_pass_rates_per_category(all_results)
    provider_rates = get_pass_rates_per_provider(all_results)

    # factual: 2 passed / 3 total = 0.67, sentiment: 1 passed / 2 total = 0.5
    assert category_rates["factual"] == 0.67
    assert category_rates["sentiment"] == 0.5
    assert category_rates["summarization"] == 0.0  # no cases this run, should not crash

    # claude: 2 passed / 3 total = 0.67, groq: 1 passed / 2 total = 0.5
    assert provider_rates["claude"] == 0.67
    assert provider_rates["groq"] == 0.5


def test_check_regression_flags_drop_and_skips_missing_key():
    baseline = {"factual": 0.90, "summarization": 0.80, "sentiment": 0.95, "legacy_category": 0.70}
    current = {"factual": 0.88, "summarization": 0.62, "sentiment": 0.95}

    regressed = check_regression(baseline, current, tolerance=0.05)

    assert regressed == ["summarization"]  # 0.18 drop, over tolerance
    # "factual" (0.02 drop) and "sentiment" (0.0 drop) are within tolerance, not flagged
    # "legacy_category" only exists in baseline, not current, and must be skipped, not crash
