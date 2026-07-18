import argparse
from pathlib import Path

from .providers.base import Provider, LLMProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider
from .providers.groq_provider import GroqProvider
from .providers.ollama_provider import OllamaProvider
from .cost_dashboard import run_comparison_batch
from .eval.dataset import load_eval_dataset
from .eval.batch_runner import run_batch


PROVIDER_CLASSES: dict[Provider, type[LLMProvider]] = {
    Provider.claude: AnthropicProvider,
    Provider.open_ai: OpenAIProvider,
    Provider.groq: GroqProvider,
    Provider.ollama: OllamaProvider,
}

def get_providers(providers: list[str]) -> dict[Provider, LLMProvider]:
    """Get a dictionary of instantiated providers, keyed by Provider."""
    instances: dict[Provider, LLMProvider] = {}
    for provider in providers:
        provider = Provider(provider)
        instances[provider] = PROVIDER_CLASSES[provider]()
    return instances

def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-compare")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cost_parser = subparsers.add_parser("cost", help="Run prompt(s) across providers, get a cost/latency/quality dashboard.")
    cost_parser.add_argument("--providers", required=True, help="Comma-separated provider names, e.g. claude,groq")
    cost_parser.add_argument("--judge", default=None, help="Provider to use as judge (default: first provider)")
    cost_parser.add_argument("--prompts", required=True, help="Path to a text file, one prompt per line.")

    eval_parser = subparsers.add_parser("eval", help="Run the eval suite across providers, get a pass-rate report.")
    eval_parser.add_argument("--providers", required=True, help="Comma-separated provider names, e.g. claude,groq")
    eval_parser.add_argument("--judge", default=None, help="Provider to use as judge (default: first provider)")
    eval_parser.add_argument("--dataset", default=None, help="Path to a custom eval dataset JSON file (default: built-in)")

    args = parser.parse_args()
    if args.command == "cost":
        cmd_cost(args)
    elif args.command == "eval":
        cmd_eval(args)

def cmd_cost(args: argparse.Namespace) -> None:
    """Handle the `cost` subcommand: run prompts across providers, write a cost/latency/quality dashboard."""
    providers = get_providers(args.providers.split(","))
    provider_list = list(providers.values())
    judge = get_providers([args.judge])[Provider(args.judge)] if args.judge else provider_list[0]
    run_comparison_batch(path=Path(args.prompts), providers=provider_list, judge=judge)

def cmd_eval(args: argparse.Namespace) -> None:
    """Handle the `eval` subcommand: run the eval dataset across providers, write a pass-rate report."""
    raw_providers = get_providers(args.providers.split(","))
    providers: dict[str, LLMProvider] = {}
    for provider, llm_provider in raw_providers.items():
        providers[str(provider)] = llm_provider
    dataset = load_eval_dataset(args.dataset)
    judge = get_providers([args.judge])[Provider(args.judge)] if args.judge else list(providers.values())[0]
    run_batch(dataset=dataset, providers=providers, judge=judge)

if __name__ == "__main__":
    main()