import asyncio
import json
from uuid import uuid4

import pytest

from app.core.exceptions import EvaluationError
from app.evaluation.datasets import load_cases
from app.evaluation.evaluator import evaluate, passed
from app.evaluation.metrics import Trajectory, contains_phrase
from app.evaluation.runner import format_report, run_cases
from app.schemas.agent import AgentRunResponse
from app.schemas.evaluation import EvaluationCase


@pytest.mark.parametrize(("body", "expected", "match"), [
    ("It is 100.", "100", True),
    ("It is 1000.", "100", False),
    ("It is −100.", "100", False),
    ("It is 100.5.", "100", False),
    ("It is 1,100.", "100", False),
    ("Preferred:  Ｐｙｔｈｏｎ.", "python", True),
    ("A concatenated word", "cat", False),
])
def test_phrase_matching(body, expected, match):
    assert contains_phrase(body, expected) is match


def trace() -> Trajectory:
    return {
        "status": "completed",
        "final_answer": "The answer is 100.",
        "error": None,
        "current_step": 2,
        "tool_calls": [{"name": "calculator", "arguments": {"expression": "25 * 4"}}],
        "observations": [{"tool": "calculator", "result": 100, "success": True, "attempts": []}],
        "retrieved_memories": [],
    }


def test_perfect_run_scores_one():
    case = EvaluationCase(id="math", input="25 * 4", expected="100", tools=["calculator"])
    metrics = evaluate(case, trace(), 42.0)
    assert metrics.model_dump() == {
        "task_success": True, "answer_quality": 1.0, "tool_accuracy": 1.0, "tool_success": 1.0,
        "trajectory_quality": 1.0, "memory_precision": 1.0, "memory_recall": 1.0,
        "efficiency": 1.0, "steps": 2, "tool_calls": 1, "latency_ms": 42.0,
        "retry_count": 0, "errors": [], "aggregate_score": 1.0,
    }
    assert passed(metrics)


def test_missing_tool_cannot_pass_with_correct_answer():
    case = EvaluationCase(id="math", input="25 * 4", expected="100", tools=["calculator"])
    metrics = evaluate(case, trace() | {"tool_calls": [], "observations": [], "current_step": 1}, 1)
    assert metrics.task_success
    assert metrics.tool_accuracy == metrics.tool_success == 0
    assert metrics.trajectory_quality < 1
    assert not passed(metrics)


def test_incorrect_order_is_detected_even_if_all_tools_succeed():
    case = EvaluationCase(id="ordered", input="task", expected="100", tools=["datetime", "calculator"])
    base = trace()
    base["tool_calls"].append({"name": "datetime", "arguments": {}})
    base["observations"].append({"tool": "datetime", "result": {}, "success": True, "attempts": []})
    metrics = evaluate(case, base | {"current_step": 3}, 2)
    assert metrics.tool_success == 1.0
    assert metrics.tool_accuracy == 0.0
    assert not passed(metrics)


def test_relevance_criteria_retries_and_limits():
    case = EvaluationCase(
        id="memory", input="task", expected="100", tools=["calculator"], criteria=["because"],
        expected_memories=["Prefers Python.", "Lives in Boston."], max_steps=2, max_retries=0,
        max_latency_ms=100,
    )
    base = trace()
    base["observations"][0]["attempts"] = [
        {"attempt": 1, "latency_ms": 3, "error": "Transient failure.", "success": False},
        {"attempt": 2, "latency_ms": 2, "error": None, "success": True},
    ]
    base["retrieved_memories"] = ["prefers PYTHON", "Unrelated fact."]
    metrics = evaluate(case, base, 125)
    assert metrics.answer_quality == 0.5
    assert metrics.memory_precision == metrics.memory_recall == 0.5
    assert metrics.efficiency == pytest.approx(1 / 3)
    assert metrics.retry_count == 1
    assert metrics.errors == ["calculator: Transient failure."]
    assert not passed(metrics)


def test_approval_pause_is_incomplete_and_missing_observation_is_detected():
    case = EvaluationCase(id="approval", input="task", expected="100", tools=["calculator"])
    result = trace() | {"status": "awaiting_approval", "final_answer": None, "observations": []}
    metrics = evaluate(case, result, 10)
    assert not metrics.task_success
    assert metrics.tool_success == 0
    assert not passed(metrics)


@pytest.mark.parametrize("content", ["not JSON", "[]", '[{"id":"missing"}]', None])
def test_invalid_dataset_rejected(tmp_path, content):
    path = tmp_path / "cases.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(EvaluationError):
        load_cases(path)


def test_duplicate_case_ids_rejected(tmp_path):
    path = tmp_path / "cases.json"
    item = {"id": "duplicate", "input": "task", "expected": "answer"}
    path.write_text(json.dumps([item, item]))
    with pytest.raises(EvaluationError, match="unique"):
        load_cases(path)


def test_runner_aggregates_and_preserves_details():
    good, bad = [
        EvaluationCase(id=label, input="task", expected=expected, tools=["calculator"])
        for label, expected in [("good", "100"), ("bad", "200")]
    ]
    saved = []

    async def execute(case):
        return AgentRunResponse.model_validate({
            **trace(), "run_id": uuid4(), "thread_id": case.id, "max_steps": 10,
            "current_plan": [], "messages": [], "pending_approval": None,
        })

    async def store(result):
        saved.append(result)

    report = asyncio.run(run_cases([good, bad], execute, store=store))
    assert (report.total, report.passed, report.failed, report.success_rate) == (2, 1, 1, 0.5)
    assert report.average_steps == 2
    assert report.average_tool_calls == 1
    assert report.results == saved
    assert "Success rate: 50%" in format_report(report)
    assert "FAIL bad" in format_report(report)



def test_case_sensitive_answer_matching_for_formatting_tasks():
    case = EvaluationCase(id="uppercase", input="Uppercase orchard.", expected="ORCHARD", case_sensitive=True)
    base = trace() | {"tool_calls": [], "observations": [], "current_step": 1}
    assert not passed(evaluate(case, base | {"final_answer": "orchard"}, 1))
    assert passed(evaluate(case, base | {"final_answer": "ORCHARD"}, 1))


def test_date_answer_must_match_successful_tool_observation():
    case = EvaluationCase(
        id="utc", input="What is the UTC date?", expected="UTC", tools=["datetime"],
        answer_from_tool_field="date",
    )
    base = trace() | {
        "final_answer": "UTC 2026-11-26",
        "tool_calls": [{"name": "datetime", "arguments": {"timezone": "UTC"}}],
        "observations": [{
            "tool": "datetime", "result": {"date": "2026-11-26"}, "success": True, "attempts": [],
        }],
    }
    assert passed(evaluate(case, base, 1))
    assert not evaluate(case, base | {"final_answer": "UTC 2024-05-23"}, 1).task_success
    missing_date = [{**base["observations"][0], "result": {"timezone": "UTC"}}]
    assert not evaluate(case, base | {"observations": missing_date}, 1).task_success
