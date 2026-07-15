# llm-compare

![CI](https://github.com/<your-username>/llm-compare/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)

**Stop picking LLMs based on "vibes." Run a single prompt across the providers Claude, GPT-4o-mini, Groq, and Ollama to see who actually wins on price, speed, and output quality.**

## What & Why

Choosing an LLM usually comes down to blind guesswork or a vague hunch like, *"Claude just feels better for this specific task."* 

**llm-compare** replaces that intuition with hard data. The tool blasts your prompt across four different providers, runs the answers through an automated judge, and spits out a clean, side-by-side breakdown of cost, latency, and quality. You get an actual data-driven decision instead of a coin flip.

## Installation

```bash
git clone https://github.com/<your-username>/llm-compare.git
cd llm-compare
pip install -e .
cp .env.example .env   # add your Anthropic/OpenAI/Groq keys
```

## Quickstart

```bash
python -m llm_compare.compare
```

Runs a sample prompt across Claude, GPT-4o-mini, Groq, and Ollama, and prints a side-by-side breakdown of the cost, latency, and judge score for each. (To try your own prompt, edit the `inp` line in `compare.py`'s `__main__` block for now — no CLI argument yet.)

## Comparison Table

| Provider | Calls | Total Cost (USD) | Avg Latency (ms) | Quality |
|---|---|---|---|---|
| claude | 5 | $0.003615 | 1992.0 | 0.9 |
| open_ai | 5 | $0.00048 | 3296.4 | 0.9 |
| groq | 5 | $0.0 | 429.2 | 0.7 |
| ollama | 5 | $0.0 | 24142.4 | 0.9 |

Groq is the clear speed winner (429ms vs. 2-24s for everyone else); OpenAI is the cheapest paid option at less than a tenth of Claude's cost for comparable quality.

## Evaluation Methodology

Each provider is scored against a fixed set of test cases spanning three categories — factual, summarization, and sentiment. Factual and sentiment cases use exact-match scoring; summarization is judged by a second LLM (LLM-as-judge), since there's no single correct string for an open-ended summary.

**Overall pass rate: 29/40 (72%)**

| Provider | Pass Rate |
|---|---|
| claude | 7/10 (70%) |
| open_ai | 8/10 (80%) |
| groq | 7/10 (70%) |
| ollama | 7/10 (70%) |

| Category | Pass Rate |
|---|---|
| factual | 12/16 (75%) |
| summarization | 8/12 (67%) |
| sentiment | 9/12 (75%) |

The lowest marks came from summarization (67% vs. 75% for the other two buckets). Every provider that tripped up on a summary case failed in identical fashion—they didn't produce a sloppy output, they just missed the narrow answer required by the exact-match and judge rules.

## Known Limitations

- **Estimated token costs:** Instead of hitting individual tokenizers (like `tiktoken`), `count_tokens()` guesses using a `~4 chars/token` rule of thumb. Costs are solid ballparks, not exact figures. 
- **No automated API failover:** If a provider goes down, the execution stops right there. There's no automatic fallback chain built into the code right now.
- **Softer summary scoring:** Summarization metrics hover around 67% (compared to 75% for facts and sentiment). That's just the nature of using an LLM-as-judge—grading open text is naturally noisier than matching exact strings.
