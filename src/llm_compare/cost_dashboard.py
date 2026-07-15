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

from .providers.base import LLMResult, LLMProvider

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
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        max-width: 700px;
        margin-bottom: 1.5rem;
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

# Display name + color per provider key, shared between the cards and the
# chart so a provider always looks the same in both places. Adjust these to
# match your actual provider keys/models.
PROVIDER_META: dict[str, dict[str, str]] = {
    "claude": {"label": "Claude", "color": "#2E7FE0"},
    "open_ai": {"label": "GPT-4o-mini", "color": "#1DB876"},
    "groq": {"label": "Groq", "color": "#F5A623"},
    "ollama": {"label": "Ollama", "color": "#6C5CE7"},
}

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
    """Render avg latency (in seconds) per provider as a bar chart, return
    it as a base64 PNG data URI. No file is written to disk.

    Providers with zero calls (e.g. offline this run) get an italic
    "offline" label instead of a zero-height bar. Bar colors and x-axis
    labels come from PROVIDER_META so the chart matches the report cards.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    providers = list(summary.keys())

    for i, provider in enumerate(providers):
        stats = summary[provider]
        meta = PROVIDER_META.get(provider, {"label": provider, "color": "#888888"})
        if stats["calls"] == 0:
            ax.text(i, 0.05, "offline", ha="center", style="italic", color="#999999")
            continue
        seconds = stats["avg_latency_ms"] / 1000
        ax.bar(i, seconds, color=meta["color"])
        ax.text(i, seconds + 0.05, f"{seconds:.2f}s", ha="center")

    ax.set_title("Average latency per provider")
    ax.set_xticks(range(len(providers)))
    ax.set_xticklabels([PROVIDER_META.get(p, {"label": p})["label"] for p in providers])
    ax.set_ylabel("Latency (s)")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def generate_cost_summary_html(
    summary: dict[str, dict[str, Any]], out_path: Path | None = None
) -> Path:
    """Write summarize()'s output as a self-contained HTML report:
    one card per provider (calls, total_cost, avg_latency_ms,
    quality) plus the latency chart from _chart_to_base64()
    embedded inline."""
    if out_path is None: out_path = RESULTS_PATH

    html_string = f"""
            <html>
                <head><meta charset="UTF-8">{STYLE_BLOCK}</head>
                <body>
                    <div style="padding: 1.5rem 0 0.5rem;">
                        <h1 style="margin: 0 0 4px;">llm-compare v0.7.0</h1>
                        <p style="margin: 0 0 1.5rem; font-size: 14px; color: #6b7280;">Cost, latency & quality — 5 real prompts across 4 providers</p>
                    </div>
                    <div class="grid">
        """

    latency_bar_chart = _chart_to_base64(summary)
    chart_image_str = f'<img src="{latency_bar_chart}">'

    closure = """
                </body>
            </html>
        """

    cards = []
    for provider in summary.keys():
        stats = summary[provider]
        meta = PROVIDER_META.get(provider, {"label": provider, "color": "#888888"})

        if stats["calls"] == 0:
            card_string = f"""
            <div class="card offline">
                <h2><span style="color:{meta['color']};">●</span> {meta['label']}</h2>
                <p class="meta">offline this run</p>
            </div>
            """
        else:
            quality_text = stats['quality'] if stats['quality'] is not None else "—"
            card_string = f"""
            <div class="card">
                <h2><span style="color:{meta['color']};">●</span> {meta['label']}</h2>
                <p class="cost">${stats['total_cost']}</p>
                <p class="meta">{stats['avg_latency_ms'] / 1000:.2f}s avg &middot; quality {quality_text}</p>
            </div>
            """
        cards.append(card_string)
    cards_text = "\n".join(cards)
    report_text = html_string + "\n" + cards_text + "\n</div>\n" + chart_image_str + "\n" + closure

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"cost_summary_report_{timestamp}.html"
    path.write_text(report_text, encoding="utf-8")

    print(f"Report saved to {path}")
    return path

def run_comparison_batch(prompts: list[str], providers: list[LLMProvider], judge: LLMProvider) -> None:
    """Run a batch of prompts through run_comparison(), then report cost,
    latency, and quality for just this batch.

    Each prompt is run across all given providers via run_comparison(),
    scored by an OpenAI judge, and logged to the cost log as it goes
    (run_comparison() already calls log_run() internally, after judging).
    Once the whole batch is done, summarize() is scoped to only this run's
    entries via `since`, so older log data (e.g. from a previous session)
    isn't mixed into the results. The combined per-provider summary
    (calls, total_cost, avg_latency_ms, quality) is printed and also
    written out as a CSV via generate_cost_summary_csv().

    Args:
        prompts: The prompts to run through every provider.
        providers: The LLMProvider instances to compare.
        judge: The LLMJudge instance as judge.
    """
    from .compare import run_comparison

    since = datetime.now().isoformat(timespec="seconds")
    for prompt in prompts:
        run_comparison(prompt, providers, judge=judge)

    summary = summarize(since=since)
    print(f"Summary: {summary}")
    generate_cost_summary_csv(summary)
    generate_cost_summary_markdown(summary)
    generate_cost_summary_html(summary)


if __name__ == "__main__":
    print(summarize(LOG_PATH))
