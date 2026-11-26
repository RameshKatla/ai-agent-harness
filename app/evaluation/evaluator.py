from itertools import zip_longest
from statistics import mean

from app.evaluation.metrics import (
    Trajectory,
    contains_phrase,
    execution_errors,
    fraction,
    retry_count,
    tokens,
)
from app.schemas.evaluation import EvaluationCase, EvaluationMetrics


class TaskSuccessEvaluator:
    def evaluate(self, case: EvaluationCase, trace: Trajectory) -> bool:
        return trace["status"] == "completed" and contains_phrase(trace["final_answer"] or "", case.expected)


class AnswerQualityEvaluator:
    def evaluate(self, case: EvaluationCase, trace: Trajectory) -> float:
        if trace["status"] != "completed":
            return 0.0
        phrases = [case.expected, *case.criteria]
        return fraction([contains_phrase(trace["final_answer"] or "", phrase) for phrase in phrases])


class ToolSelectionEvaluator:
    def evaluate(self, case: EvaluationCase, trace: Trajectory) -> tuple[float, float]:
        calls = [call["name"] for call in trace["tool_calls"]]
        observations = trace["observations"]
        accuracy = fraction([actual == expected for actual, expected in zip_longest(calls, case.tools)])
        success = fraction([
            call is not None and observation is not None
            and call == observation["tool"] and observation["success"]
            for call, observation in zip_longest(calls, observations)
        ])
        # A case that needs a tool should receive no tool-success credit if no call was made.
        return accuracy, 0.0 if case.tools and not calls else success


class TrajectoryEvaluator:
    def evaluate(self, case: EvaluationCase, trace: Trajectory) -> float:
        calls = [call["name"] for call in trace["tool_calls"]]
        observed = [observation["tool"] for observation in trace["observations"]]
        return fraction([
            trace["status"] == "completed" and bool(trace["final_answer"]) and not trace["error"],
            calls == observed,
            calls == case.tools,
            trace["current_step"] >= len(calls) + 1,
        ])


class MemoryEvaluator:
    def evaluate(self, case: EvaluationCase, trace: Trajectory) -> tuple[float, float]:
        expected = {tokens(memory) or (memory.casefold(),) for memory in case.expected_memories}
        retrieved = {tokens(memory) or (memory.casefold(),) for memory in trace["retrieved_memories"]}
        found = expected & retrieved
        return (
            len(found) / len(retrieved) if retrieved else float(not expected),
            len(found) / len(expected) if expected else 1.0,
        )


class EfficiencyEvaluator:
    def evaluate(self, case: EvaluationCase, trace: Trajectory, latency_ms: float, retries: int) -> float:
        limits = [
            (trace["current_step"], case.max_steps),
            (len(trace["tool_calls"]), case.max_tool_calls),
            (retries, case.max_retries),
            (latency_ms, case.max_latency_ms),
        ]
        return fraction([value <= limit for value, limit in limits if limit is not None])


def evaluate(case: EvaluationCase, trace: Trajectory, latency_ms: float) -> EvaluationMetrics:
    task_success = TaskSuccessEvaluator().evaluate(case, trace)
    answer_quality = AnswerQualityEvaluator().evaluate(case, trace)
    tool_accuracy, tool_success = ToolSelectionEvaluator().evaluate(case, trace)
    trajectory_quality = TrajectoryEvaluator().evaluate(case, trace)
    memory_precision, memory_recall = MemoryEvaluator().evaluate(case, trace)
    retries = retry_count(trace)
    efficiency = EfficiencyEvaluator().evaluate(case, trace, latency_ms, retries)
    aggregate = mean([
        task_success, answer_quality, tool_accuracy, tool_success, trajectory_quality,
        memory_precision, memory_recall, efficiency,
    ])
    return EvaluationMetrics(
        task_success=task_success, answer_quality=answer_quality,
        tool_accuracy=tool_accuracy, tool_success=tool_success, trajectory_quality=trajectory_quality,
        memory_precision=memory_precision, memory_recall=memory_recall, efficiency=efficiency,
        steps=trace["current_step"], tool_calls=len(trace["tool_calls"]), latency_ms=latency_ms,
        retry_count=retries, errors=execution_errors(trace), aggregate_score=aggregate,
    )


def passed(metrics: EvaluationMetrics) -> bool:
    return (
        metrics.task_success
        and metrics.answer_quality == metrics.tool_accuracy == metrics.tool_success == 1.0
        and metrics.trajectory_quality == metrics.memory_precision == metrics.memory_recall == 1.0
        and metrics.efficiency == 1.0
    )
