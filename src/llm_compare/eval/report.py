import argparse
import base64
import datetime
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .dataset import Category
from .types import EvalCase, EvalResult
from .regression_check import load_baseline, check_regression
from ..cost_dashboard import STYLE_BLOCK
from ..providers.base import provider_label, provider_color


def load_results(path: str) -> dict[str, list[EvalResult]]:
    """Load a batch_*.json file back into {provider: [EvalResult]}."""
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
    """Return {provider: pass_rate} (0.0-1.0), rounded to 2 decimals."""
    rates_per_provider = {}
    for provider_name, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        rate = passed / total if total > 0 else 0.0
        rates_per_provider[provider_name] = round(rate, 2)
    return rates_per_provider

def get_pass_rates_per_category(all_results: dict[str, list[EvalResult]]) -> dict[str, float]:
    """Return {category: pass_rate} (0.0-1.0), rounded to 2 decimals."""
    rates_per_category = {}
    categorized_results = get_categorized_result_numbers(all_results)
    for cat in categorized_results.keys():
        passed = categorized_results[cat]["passed"]
        total = categorized_results[cat]["total"]
        rate = passed / total if total > 0 else 0.0
        rates_per_category[str(cat)] = round(rate, 2)
    return rates_per_category

def get_pass_rates(base: str, all_results: dict[str, list[EvalResult]]) -> dict[str, float]:
    """Return pass rates grouped by 'provider' or 'category'."""
    if not base:
        raise ValueError(f"{base} is not a valid baseline. Should be 'provider' or 'category'.")

    if base == "provider":
        return get_pass_rates_per_provider(all_results)
    elif base == "category":
        return get_pass_rates_per_category(all_results)
    else:
        raise ValueError(f"{base!r} is not a valid baseline. Should be 'provider' or 'category'.")

def generate_report(all_results: dict[str, list[EvalResult]]) -> None:
    """Print per-provider and per-category pass-rate tables, plus failure details."""

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

def _pass_rate_chart_to_base64(rates: dict[str, float], use_provider_meta: bool = False, title: str = "Pass rate") -> str:
    """Render a 0-100% pass-rate bar chart for a {label: rate} mapping,
    return it as a base64 PNG data URI. No file is written to disk.

    If use_provider_meta is True, bar colors and x-axis labels come from
    PROVIDER_REGISTRY (via provider_label()/provider_color()) so provider
    bars match cost_dashboard.py's chart; otherwise labels are just
    capitalized as-is (e.g. for the per-category chart).
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = list(rates.keys())

    for i, label in enumerate(labels):
        color = provider_color(label) if use_provider_meta else "#2E7FE0"
        pct = rates[label] * 100
        ax.bar(i, pct, color=color)
        ax.text(i, pct + 1.5, f"{pct:.0f}%", ha="center")

    ax.set_title(title)
    ax.set_ylim(0, 115)
    ax.set_xticks(range(len(labels)))
    display_labels = [provider_label(lbl) if use_provider_meta else str(lbl).capitalize() for lbl in labels]
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Pass rate (%)")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_report_html(all_results: dict[str, list[EvalResult]], out_path: Path | None = None) -> Path:
    """Write a self-contained HTML eval report: a card + chart per provider,
    a per-category pass-rate chart, and a failure-details table. Returns
    the file path."""
    if out_path is None:
        out_path = Path(__file__).parent / "results"

    total = sum(len(r) for r in all_results.values())
    passed = sum(1 for r in all_results.values() for x in r if x.passed)
    overall_rate = passed / total if total else 0.0

    provider_rates = get_pass_rates_per_provider(all_results)
    category_rates = get_pass_rates_per_category(all_results)

    provider_chart = _pass_rate_chart_to_base64(provider_rates, use_provider_meta=True, title="Pass rate per provider")
    category_chart = _pass_rate_chart_to_base64(category_rates, title="Pass rate per category")

    n = len(all_results)
    wide = n >= 3
    grid_attrs = f'class="grid wide" style="grid-template-columns: repeat({n}, 1fr);"' if wide else 'class="grid"'
    chart_row_class = "chart-row wide" if wide else "chart-row"

    cards = []
    for provider_name, results in all_results.items():
        p = sum(1 for r in results if r.passed)
        t = len(results)
        label = provider_label(provider_name)
        color = provider_color(provider_name)
        cards.append(f"""
        <div class="card">
            <h2><span style="color:{color};">●</span> {label}</h2>
            <p class="cost">{p}/{t}</p>
            <p class="meta">{provider_rates[provider_name]:.0%} pass rate</p>
        </div>
        """)

    failure_rows = []
    for provider_name, results in all_results.items():
        for r in results:
            if not r.passed:
                score = f"{r.score:.1f}" if r.score is not None else "None"
                failure_rows.append(
                    f"<tr><td>{provider_name}</td><td>{r.case.category}</td>"
                    f"<td>{r.case.prompt[:60]}</td><td>{score}</td><td>{r.reason[:80]}</td></tr>"
                )

    failures_block = ""
    if failure_rows:
        failures_block = f"""
        <h2 style="margin-top:2rem;">Failure Details</h2>
        <table style="width:100%; border-collapse:collapse; font-size:14px;">
            <tr style="background:#1F3864; color:white;">
                <th style="padding:6px; text-align:left;">Provider</th>
                <th style="padding:6px; text-align:left;">Category</th>
                <th style="padding:6px; text-align:left;">Prompt</th>
                <th style="padding:6px; text-align:left;">Score</th>
                <th style="padding:6px; text-align:left;">Reason</th>
            </tr>
            {''.join(failure_rows)}
        </table>
        """

    html_string = f"""
        <html>
            <head><meta charset="UTF-8">{STYLE_BLOCK}</head>
            <body>
                <div style="padding: 1.5rem 0 0.5rem;">
                    <h1 style="margin: 0 0 4px;">llm-compare — Eval Report</h1>
                    <p style="margin: 0 0 1.5rem; font-size: 14px; color: #6b7280;">
                        Overall pass rate: {passed}/{total} ({overall_rate:.0%})
                    </p>
                </div>
                <div {grid_attrs}>
                    {''.join(cards)}
                </div>
                <div class="{chart_row_class}">
                    <div><img src="{provider_chart}"></div>
                    <div><img src="{category_chart}"></div>
                </div>
                {failures_block}
            </body>
        </html>
    """

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"eval_report_{timestamp}.html"
    path.write_text(html_string, encoding="utf-8")

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
    generate_report_html(all_eval_results, RESULTS_DIR)

    if args.check_regression:
        baseline = load_baseline(args.check_regression)
        current = get_pass_rates("category", all_eval_results)
        regressed = check_regression(baseline, current, tolerance=0.05)

        print("\n----------------- Regression Check -----------------")
        if regressed:
            print(f"REGRESSION DETECTED in: {', '.join(regressed)}")
        else:
            print("No regressions detected.")