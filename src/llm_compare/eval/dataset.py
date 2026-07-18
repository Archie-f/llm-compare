import json
from enum import StrEnum
from pathlib import Path

from .types import EvalCase

class Category(StrEnum):
    """The category of an Eval."""
    factual = "factual"
    summarization = "summarization"
    sentiment = "sentiment"

PATH_TO_EVAL_DATASET = Path(__file__).parent / "eval_dataset.json"

def load_eval_dataset(path: Path | None = None) -> list[EvalCase]:
    """Load eval cases from a JSON file, defaulting to the packaged eval_dataset.json."""
    if path is None: path = PATH_TO_EVAL_DATASET

    with open(path, "r") as f:
        dataset = [EvalCase(**case) for case in json.load(f)]

    return dataset
