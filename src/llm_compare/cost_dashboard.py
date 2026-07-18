import base64
import csv
import dataclasses
import json
from io import BytesIO

import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .providers.base import LLMResult, LLMProvider, provider_label, provider_color

LOG_PATH: Path = Path(__file__).parent / "results" / "cost_log.jsonl"
CSV_PATH: Path = Path(__file__).parent / "results" / "cost_summary.csv"
RESULTS_PATH: Path = Path(__file__).parent / "results"

STYLE_BLOCK = """
<style>
    body {
        font-family: -apple-system, "Segoe UI", sans-serif;
        background: #f4f5f7;
        margin: 0;
        padding: 2rem;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 16px;
        max-width: 1000px;
        margin-bottom: 1.5rem;
    }
    .chart-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        max-width: 1000px;
        margin-bottom: 1.5rem;
    }
    .chart-row > div {
        flex: 1 1 320px;
        max-width: 492px;
    }
    .grid.wide {
        max-width: none;
    }
    .chart-row.wide {
        margin-left: auto;
        margin-right: auto;
    }
    .card {
        background: #ffffff;
        border: 0.5px solid #d8dbe0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .card.offline { color: #999999; }
    .card h2 {
        font-size: 18px;
        margin: 0 0 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card .cost {
        font-size: 26px;
        font-weight: 600;
        margin: 0 0 4px;
    }
    .card .meta {
        font-size: 14px;
        color: #6b7280;
        margin: 0;
    }
    img { max-width: 100%; border-radius: 8px; }
</style>
"""

def log_run(result: LLMResult, log_path: Path | None = None) -> None:
    """Append one LLMResult as a single JSON line to the cost log."""
    if log_path is None: log_path = LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = dataclasses.asdict(result)
    entry["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with log_path.open(mode="a") as f:
        f.write(json.dumps(entry) + "\n")

def tracked_call(
    provider: LLMProvider, user_input: str, system_prompt: str = ""
) -> LLMResult:
    """Call provider.ask() and log the resulting metrics before returning it."""
    result = provider.ask(user_input=user_input, system_prompt=system_prompt)
    log_run(result)
    return result


def summarize(log_path: Path | None = None, since: str | None = None) -> dict[str, dict[str, Any]]:
    """Aggregate a cost log into per-provider totals and averages.

    Returns:
        {provider: {"calls": int, "total_cost": float, "avg_latency_ms": float}}
    """
    if log_path is None: log_path = LOG_PATH
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with log_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if since is not None and  entry["timestamp"] < since:
                continue
            grouped[entry["provider"]].append(entry)

    summary: dict[str, dict[str, Any]] = {}
    for provider, entries in grouped.items():
        calls = len(entries)
        total_cost = round(sum(entry["cost"] for entry in entries), 6)
        avg_latency_ms = round((sum(entry["latency_ms"] for entry in entries) / calls), 1)
        scored = [entry["judge_score"] for entry in entries if entry["judge_score"] is not None]
        quality = round(sum(scored) / len(scored), 1) if scored else None
        summary[provider] = {
            "calls": calls,
            "total_cost": total_cost,
            "avg_latency_ms": avg_latency_ms,
            "quality": quality
        }

    return summary


def generate_cost_summary_csv(summary: dict[str, dict[str, Any]], out_path: Path | None = None) -> None:
    """Write summarize()'s output as a CSV file.

    Columns: provider, calls, total_cost, avg_latency_ms — one row per
    provider in summary. out_path is the destination CSV file itself.
    """
    if out_path is None: out_path = RESULTS_PATH

    headers = ["provider", "calls", "total_cost", "avg_latency_ms", "quality"]
    rows = []
    for provider in summary.keys():
        stats = summary[provider]
        rows.append([
            provider,
            stats['calls'],
            stats['total_cost'],
            stats['avg_latency_ms'],
            stats['quality']
        ])

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"cost_summary_table_{timestamp}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def generate_cost_summary_markdown(summary: dict[str, dict[str, Any]], out_path: Path | None = None) -> Path:
    """Write summarize()'s output as a Markdown table. Returns the file path."""
    if out_path is None: out_path = RESULTS_PATH

    lines = ["# Summary Report", "\n## Number of calls, Cost, Latency and Quality — Per Provider\n",
             "| Provider | Calls | Total Cost (USD) | Avg Latency (ms) | Quality |", "|---|---|---|---|---|"]

    for provider in summary.keys():
        stats = summary[provider]
        lines.append(f"| {provider} | {stats['calls']} | {stats['total_cost']} | {stats['avg_latency_ms']} | {stats['quality']} |")

    report_text = "\n".join(lines)

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"cost_summary_report_{timestamp}.md"
    path.write_text(report_text)
    print(f"Report saved to {path}")

    return path

def _chart_to_base64(summary: dict[str, dict[str, Any]]) -> str:
    """Render avg latency per provider as a base64 PNG data URI (no file
    written). Providers with zero calls get an "offline" label instead
    of a bar. Colors/labels come from PROVIDER_REGISTRY."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    providers = list(summary.keys())

    for i, provider in enumerate(providers):
        stats = summary[provider]
        if stats["calls"] == 0:
            ax.text(i, 0.05, "offline", ha="center", style="italic", color="#999999")
            continue
        seconds = stats["avg_latency_ms"] / 1000
        ax.bar(i, seconds, color=provider_color(provider))
        ax.text(i, seconds + 0.05, f"{seconds:.2f}s", ha="center")

    ax.set_title("Average latency per provider")
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels([provider_label(p) for p in providers])
    ax.set_ylabel("Latency (s)")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def _quality_chart_to_base64(summary: dict[str, dict[str, Any]]) -> str:
    """Render judge-score quality (0-1) per provider as a base64 PNG data
    URI (no file written). Providers with zero calls get an "offline"
    label; providers with calls but no judge score get an "N/A" label.
    Colors/labels come from PROVIDER_REGISTRY."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    providers = list(summary.keys())

    for i, provider in enumerate(providers):
        stats = summary[provider]
        if stats["calls"] == 0:
            ax.text(i, 0.05, "offline", ha="center", style="italic", color="#999999")
            continue
        if stats["quality"] is None:
            ax.text(i, 0.05, "N/A", ha="center", style="italic", color="#999999")
            continue
        quality = stats["quality"]
        ax.bar(i, quality, color=provider_color(provider))
        ax.text(i, quality + 0.02, f"{quality:.1f}", ha="center")

    ax.set_title("Quality per provider")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels([provider_label(p) for p in providers])
    ax.set_ylabel("Quality (0-1)")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_cost_summary_html(
    summary: dict[str, dict[str, Any]], out_path: Path | None = None
) -> Path:
    """Write summarize()'s output as a self-contained HTML report: one
    cost-only card per provider, plus side-by-side latency and quality
    charts from _chart_to_base64()/_quality_chart_to_base64()."""
    if out_path is None: out_path = RESULTS_PATH

    n = len(summary)
    wide = n >= 3
    grid_attrs = f'class="grid wide" style="grid-template-columns: repeat({n}, 1fr);"' if wide else 'class="grid"'
    chart_row_class = "chart-row wide" if wide else "chart-row"

    html_string = f"""
            <html>
                <head><meta charset="UTF-8">{STYLE_BLOCK}</head>
                <body>
                    <div style="padding: 1.5rem 0 0.5rem;">
                        <h1 style="margin: 0 0 4px;">llm-compare — Cost Dashboard Report</h1>
                        <p style="margin: 0 0 1.5rem; font-size: 14px; color: #6b7280;">Cost, latency & quality — 5 real prompts across 4 providers</p>
                    </div>
                    <div {grid_attrs}>
        """

    latency_bar_chart = _chart_to_base64(summary)
    quality_bar_chart = _quality_chart_to_base64(summary)
    charts_row = f"""
        <div class="{chart_row_class}">
            <div><img src="{latency_bar_chart}"></div>
            <div><img src="{quality_bar_chart}"></div>
        </div>
        """

    closure = """
                </body>
            </html>
        """

    cards = []
    for provider in summary.keys():
        stats = summary[provider]
        label = provider_label(provider)
        color = provider_color(provider)

        if stats["calls"] == 0:
            card_string = f"""
            <div class="card offline">
                <h2><span style="color:{color};">●</span> {label}</h2>
                <p class="meta">offline this run</p>
            </div>
            """
        else:
            card_string = f"""
            <div class="card">
                <h2><span style="color:{color};">●</span> {label}</h2>
                <p class="cost">${stats['total_cost']}</p>
            </div>
            """
        cards.append(card_string)
    cards_text = "\n".join(cards)
    report_text = html_string + "\n" + cards_text + "\n</div>\n" + charts_row + "\n" + closure

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"cost_summary_report_{timestamp}.html"
    path.write_text(report_text, encoding="utf-8")

    print(f"Report saved to {path}")
    return path

def load_prompts(path: Path) -> list[str]:
    """Load prompts from a txt file."""
    prompts = []
    with open(path, "r") as f:
        raw_prompts = f.read().splitlines()
    for line in raw_prompts:
        if not line or line.startswith("#"):
            continue
        prompts.append(line.strip())
    return prompts

def run_comparison_batch(path: Path, providers: list[LLMProvider], judge: LLMProvider) -> None:
    """Run every prompt through run_comparison(), scored by judge, then
    print and write out a per-provider cost/latency/quality summary
    (CSV, Markdown, HTML) scoped to just this batch via a `since` timestamp.

    Args:
        path: Path to the txt file that includes the prompts to run through every provider.
        providers: The LLMProvider instances to compare.
        judge: The LLMProvider used for LLM-as-judge scoring.
    """
    from .compare import run_comparison

    since = datetime.now().isoformat(timespec="seconds")
    prompts = load_prompts(path)
    for prompt in prompts:
        run_comparison(prompt, providers, judge=judge)

    summary = summarize(since=since)
    print(f"Summary: {summary}")
    generate_cost_summary_csv(summary)
    generate_cost_summary_markdown(summary)
    generate_cost_summary_html(summary)


if __name__ == "__main__":
    print(summarize(LOG_PATH))
