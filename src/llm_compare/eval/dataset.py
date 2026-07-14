from enum import StrEnum

from .types import EvalCase

class Category(StrEnum):
    """The category of an Eval."""
    factual = "factual"
    summarization = "summarization"
    sentiment = "sentiment"

EVAL_DATASET: list[EvalCase] = [
    # FACTUAL -------------------------------------------------------------------------------------------------
    EvalCase(
        prompt="What is the capital of Norway?",
        expected="Oslo",
        category=Category.factual,
        task_description="Answer with the city name only.",
    ),
    EvalCase(
        prompt="In which year was Python first released?",
        expected="1991",
        category=Category.factual,
        task_description="Answer with the year only.",
    ),
    EvalCase(
        prompt="How many centimeters is an inch?",
        expected="2.54",
        category=Category.factual,
        task_description="Answer with the centimeter value only.",
    ),
    EvalCase(
        prompt="What is the highest academic degree?",
        expected="Doctorate",
        category=Category.factual,
        task_description="Answer with the name of the academic degree only.",
    ),

    # SUMMARIZATION -------------------------------------------------------------------------------------------
    EvalCase(
        prompt="Summarise in one sentence: RAG systems retrieve relevant documents "
               "from a vector database and pass them as context to an LLM, allowing "
               "the model to answer questions about documents it was not trained on.",
        expected="",
        category=Category.summarization,
        task_description="Summarise the given text in exactly one clear sentence.",
    ),
    EvalCase(
        prompt="Summarise in one sentence: A supply-demand balance (or market equilibrium) is "
               "the economic state where the quantity of a good or service available matches "
               "the amount consumers want to buy. At this perfect alignment, the market clears "
               "without shortages or surpluses, stabilizing the price at an optimal level.",
        expected="",
        category=Category.summarization,
        task_description="Summarise the given text in exactly one clear sentence.",
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
        category=Category.summarization,
        task_description="Summarise the given text in exactly one clear sentence.",
    ),

    # SENTIMENT -----------------------------------------------------------------------------------------
    EvalCase(
        prompt="Classify the sentiment of: 'The product exceeded all my expectations.'",
        expected="positive",
        category=Category.sentiment,
        task_description="Classify as exactly one word: positive, negative, or neutral.",
    ),
    EvalCase(
        prompt="Classify the sentiment of: 'I returned the product because it was broken when the delivery arrived me.'",
        expected="negative",
        category=Category.sentiment,
        task_description="Classify as exactly one word: positive, negative, or neutral.",
    ),
    EvalCase(
        prompt="Classify the sentiment of: 'I bought this to my father for his birthday. I hope he likes it.'",
        expected="neutral",
        category=Category.sentiment,
        task_description="Classify as exactly one word: positive, negative, or neutral.",
    )
]