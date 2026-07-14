import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm_compare.eval.types import EvalCase
from src.llm_compare.eval.batch_runner import run_batch, estimate_cost
from src.llm_compare.eval.dataset import EVAL_DATASET
from src.llm_compare.eval.harness import SCORER_DISPATCH

eval_dataset: list[EvalCase] = [
        EvalCase(
            prompt="What is the highest academic degree?",
            expected="Doctorate",
            category="factual",
            task_description="Answer with the name of the academic degree only. Do not add any other text",
        ),
        EvalCase(
            prompt="The North Atlantic Treaty Organization (NATO) was established in 1949 to counter "
                   "the rising military threat of the Soviet Union in Europe. It united North American "
                   "and European nations under a promise of collective defense, meaning an attack on "
                   "one member is an attack on all. By keeping the United States actively involved "
                   "in European security, the alliance successfully prevented the rebirth of aggressive "
                   "local militarism. This stability created a safe environment that allowed war-torn "
                   "European countries to rebuild their economies and political systems.",
            expected="",
            category="summarization",
            task_description="Summarise the given text in exactly one clear sentence.",
        )
    ]

def test_correct_dispatch():
    mock_provider = MagicMock()
    mock_provider.ask.return_value = MagicMock(text="Doctorate")

    results = run_batch(eval_dataset, {"mock": mock_provider}, persist=False)

    assert results["mock"][0].score == 1
    assert isinstance(results["mock"][1].score, float)

def test_persist_false_leaves_no_file():
    mock_provider = MagicMock()
    mock_provider.ask.return_value = MagicMock(text="Reply from model.")

    results_dir = Path("results")
    files_before = set(results_dir.iterdir()) if results_dir.exists() else set()

    run_batch(eval_dataset, {"mock": mock_provider}, persist=False)
    files_after = set(results_dir.iterdir()) if results_dir.exists() else set()

    assert files_before == files_after

def test_all_categories_are_registered():
    for case in EVAL_DATASET:
        assert case.category in SCORER_DISPATCH, (
            f"Category '{case.category}' has no scorer in SCORER_DISPATCH"
        )

def test_estimate_cost_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        estimate_cost(eval_dataset, ["unknown_provider"])


















