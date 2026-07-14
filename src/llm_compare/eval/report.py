import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from .dataset import Category
from .types import EvalCase, EvalResult
from .regression_check import load_baseline, check_regression


def load_results(path: str) -> dict[str, list[EvalResult]]:
    raw: dict[str, list[dict[str, Any]]] = json.loads(Path(path).read_text())

    data: dict[str, list[EvalResult]] = {}
    for llm in raw:
        for r in raw[llm]:
            r["case"] = EvalCase(**r["case"])
        data[llm] = [EvalResult(**r) for r in raw[llm]]

    return data

def get_categorized_result_numbers(all_results: dict[str, list[EvalResult]]) -> dict[str, dict[str, int]]:
    """Categorize passed results by category."""
    category_classified: dict[str, dict[str, int]] = {
        Category.factual:       {"passed": 0, "total": 0},
        Category.summarization: {"passed": 0, "total": 0},
        Category.sentiment:     {"passed": 0, "total": 0},
    }

    for results in all_results.values():
        for eval_result in results:
            cat = eval_result.case.category
            category_classified[cat]["total"] += 1
            if eval_result.passed:
                category_classified[cat]["passed"] += 1

    return category_classified

def get_pass_rates_per_provider(all_results: dict[str, list[EvalResult]]) -> dict[str, float]:
    rates_per_provider = {}
    for provider_name, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        rate = passed / total if total > 0 else 0.0
        rates_per_provider[provider_name] = round(rate, 2)
    return rates_per_provider

def get_pass_rates_per_category(all_results: dict[str, list[EvalResult]]) -> dict[str, float]:
    rates_per_category = {}
    categorized_results = get_categorized_result_numbers(all_results)
    for cat in categorized_results.keys():
        passed = categorized_results[cat]["passed"]
        total = categorized_results[cat]["total"]
        rate = passed / total if total > 0 else 0.0
        rates_per_category[str(cat)] = round(rate, 2)
    return rates_per_category

def get_pass_rates(base: str, all_results: dict[str, list[EvalResult]]) -> dict[str, float]:
    if not base:
        raise ValueError(f"{base} is not a valid baseline. Should be 'provider' or 'category'.")

    if base == "provider":
        return get_pass_rates_per_provider(all_results)
    elif base == "category":
        return get_pass_rates_per_category(all_results)
    else:
        raise ValueError(f"{base!r} is not a valid baseline. Should be 'provider' or 'category'.")

def generate_report(all_results: dict[str, list[EvalResult]]) -> None:
    """Generate a report of the results per-provider and per-category pass-rate tables, plus failure-only detail"""

    total_results = sum(len(r) for r in all_results.values())
    passed_results = sum(1 for r in all_results.values() for x in r if x.passed)
    print(f"\nOverall Pass Rate: {passed_results}/{total_results} ({passed_results/total_results:.0%})")

    print("\n----------- Pass Rates - Per Provider -----------")
    print(f"  {'Provider Name':<16} {'Pass Rate':<13} {'Pass Percentage'}")
    print("-" * 49)
    provider_rates = get_pass_rates_per_provider(all_results)
    for provider_name, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        rate = provider_rates[provider_name]
        print(f"  {provider_name:<15}  {passed}/{total:<10}  ({rate:.0%})")

    print("\n-------------- Categorized Results --------------")
    print(f"  {'Category':<16} {'Pass Rate':<13} {'Pass Percentage'}")
    print("-" * 49)
    categorized_results = get_categorized_result_numbers(all_results)
    category_rates = get_pass_rates_per_category(all_results)

    for cat, counts in categorized_results.items():
        passed, total = counts["passed"], counts["total"]
        rate = category_rates[cat]
        print(f"  {cat:<15}  {passed}/{total:<10}  ({rate:.0%})")

    print("Failure Details: ")
    for provider_name, results in all_results.items():
        for result in results:
            if not result.passed:
                prompt = f"{result.case.prompt[:20]}.."
                reason = f"{result.reason[:20]}..."
                score = round(result.score, 1) if result.score is not None else None
                print(
                    f" * Provider: {provider_name:<10} Prompt: {prompt:<25} Category: {result.case.category.capitalize():<15}  "
                    f"Score: {str(score):<5} Reason: {reason}"
                )

def build_report_markdown(all_results: dict[str, list[EvalResult]]) -> str:
    """Build the report as a Markdown string (pass/fail + cost/latency)."""
    lines = []
    total = sum(len(r) for r in all_results.values())
    passed = sum(1 for r in all_results.values() for x in r if x.passed)
    lines.append("# Eval Report")
    lines.append(f"**Overall Pass Rate:** {passed}/{total} ({passed / total:.0%})")

    lines.append("\n## Pass Rate, Cost & Latency — Per Provider\n")
    lines.append("| Provider | Pass Rate | Total Cost (USD) | Avg Latency (ms) |")
    lines.append("|---|---|---|---|")
    for provider_name, results in all_results.items():
        p = sum(1 for r in results if r.passed)
        t = len(results)
        cost = sum(r.cost or 0 for r in results)
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        lines.append(
            f"| {provider_name} | {p}/{t} ({p / t:.0%}) | "
            f"${cost:.4f} | {avg_latency:.0f} |"
        )

    lines.append("\n## Failure Details\n")
    for provider_name, results in all_results.items():
        for r in results:
            if not r.passed:
                lines.append(f"- **{provider_name}** — {r.case.prompt[:40]}... — {r.reason[:60]}")

    return "\n".join(lines)

def save_report(all_results: dict[str, list[EvalResult]], results_dir: Path) -> Path:
    """Write the Markdown report to results/report_<timestamp>.md. Returns the path."""
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = results_dir / f"report_{timestamp}.md"
    path.write_text(build_report_markdown(all_results))
    print(f"Report saved to {path}")
    return path



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate an eval report from a batch results file.")
    parser.add_argument(
        "results_path",
        nargs="?",
        default=None,
        help="Path to a batch_*.json file. Defaults to the most recent file in results/.",
    )
    parser.add_argument(
        "--check-regression",
        metavar="BASELINE_PATH",
        default=None,
        help="Path to a baseline JSON file (dict[str, float]) to check current category pass rates against.",
    )
    args = parser.parse_args()

    RESULTS_DIR = Path(__file__).parent / "results"
    if args.results_path:
        results_path = args.results_path
    else:
        results_path = str(max(RESULTS_DIR.glob("batch_*.json"), key=lambda p: p.stat().st_mtime))

    all_eval_results = load_results(results_path)
    generate_report(all_eval_results)
    save_report(all_eval_results, RESULTS_DIR)

    if args.check_regression:
        baseline = load_baseline(args.check_regression)
        current = get_pass_rates("category", all_eval_results)
        regressed = check_regression(baseline, current, tolerance=0.05)

        print("\n----------------- Regression Check -----------------")
        if regressed:
            print(f"REGRESSION DETECTED in: {', '.join(regressed)}")
        else:
            print("No regressions detected.")