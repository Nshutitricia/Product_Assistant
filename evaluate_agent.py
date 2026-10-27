import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from agent import MODEL, run_agent
from logging_config import setup_logging

THRESHOLD = 0.7
RESULTS_FILE = Path(__file__).with_name("eval_results.json")

CASES = [
    ("What is the price of P002?", "Answer says 15,000 RWF"),
    ("Find the laptop", "Answer lists the Laptop (P001)"),
    ("What is the price of P999?", "Answer says the product does not exist and gives no price"),
]


def build_judge() -> AnthropicModel:
    return AnthropicModel(
        model=MODEL,
        cost_per_input_token=0.10 / 1_000_000,
        cost_per_output_token=0.50 / 1_000_000,
    )


def score_metric(metric_cls, judge, test_case) -> dict:
    metric = metric_cls(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False)
    try:
        metric.measure(test_case)
        return {
            "metric": metric_cls.__name__.removesuffix("Metric"),
            "score": metric.score,
            "passed": bool(metric.is_successful()),
            "reason": metric.reason or "(no reason returned)",
        }
    except Exception as exc:
        return {
            "metric": metric_cls.__name__.removesuffix("Metric"),
            "score": None,
            "passed": False,
            "reason": f"metric error: {type(exc).__name__}: {exc}",
        }


def evaluate() -> list[dict]:
    judge = build_judge()
    results = []
    for question, expected in CASES:
        agent_result = run_agent(question)
        entry = {
            "input": question,
            "expected": expected,
            "answer": agent_result.answer,
            "retrieval_context": agent_result.tool_outputs,
            "latency_s": round(agent_result.latency_s, 2),
            "metrics": [],
        }
        if not agent_result.success:
            for cls in (AnswerRelevancyMetric, FaithfulnessMetric):
                entry["metrics"].append(
                    {
                        "metric": cls.__name__.removesuffix("Metric"),
                        "score": None,
                        "passed": False,
                        "reason": "agent request failed",
                    }
                )
        else:
            test_case = LLMTestCase(
                input=question,
                actual_output=agent_result.answer,
                retrieval_context=agent_result.tool_outputs,
            )
            for cls in (AnswerRelevancyMetric, FaithfulnessMetric):
                entry["metrics"].append(score_metric(cls, judge, test_case))
        results.append(entry)
    return results


def shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def print_table(results: list[dict]) -> None:
    headers = ("Case", "Metric", "Score", "Result", "Reason")
    rows = [
        (
            shorten(entry["input"], 28),
            m["metric"],
            "n/a" if m["score"] is None else f"{m['score']:.2f}",
            "PASS" if m["passed"] else "FAIL",
            shorten(m["reason"], 60),
        )
        for entry in results
        for m in entry["metrics"]
    ]
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(line)
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")
    print(line)
    print("\nAnswers:")
    for entry in results:
        print(f"  Q: {entry['input']}\n  A: {shorten(entry['answer'], 150)}")


if __name__ == "__main__":
    setup_logging()
    results = evaluate()
    print_table(results)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_FILE.name} (judge model: {MODEL}, threshold {THRESHOLD})")
