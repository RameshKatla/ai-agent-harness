import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

import httpx
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.executor import AgentExecutor
from app.agent.service import AgentService
from app.core.config import Settings
from app.core.exceptions import AgentError
from app.db.database import create_database_engine
from app.db.repositories import AgentRunRepository, EvaluationRepository
from app.evaluation.datasets import load_cases
from app.evaluation.evaluator import evaluate, passed
from app.evaluation.metrics import Trajectory
from app.llm.ollama_client import OllamaLLM
from app.memory.manager import MemoryManager
from app.memory.semantic import OllamaEmbedder
from app.schemas.agent import AgentRunResponse
from app.schemas.evaluation import EvaluationCase, EvaluationReport, EvaluationResult
from app.tools import BUILTIN_PERMISSIONS, build_tool_registry

ExecuteCase = Callable[[EvaluationCase], Awaitable[AgentRunResponse]]
ResultSink = Callable[[EvaluationResult], Awaitable[None]]


async def run_cases(
    cases: Sequence[EvaluationCase], execute: ExecuteCase, *, store: ResultSink | None = None,
) -> EvaluationReport:
    results = []
    for case in cases:
        started = perf_counter()
        run = await execute(case)
        latency_ms = (perf_counter() - started) * 1000
        trace: Trajectory = {
            "status": run.status,
            "final_answer": run.final_answer,
            "error": run.error,
            "current_step": run.current_step,
            "tool_calls": run.tool_calls,
            "observations": run.observations,
            "retrieved_memories": run.retrieved_memories,
        }
        metrics = evaluate(case, trace, latency_ms)
        result = EvaluationResult(
            case_id=case.id, run_id=run.run_id, status=run.status, passed=passed(metrics), metrics=metrics,
        )
        if store is not None:
            await store(result)
        results.append(result)
    total = len(results)
    successful = sum(result.passed for result in results)
    return EvaluationReport(
        results=results, total=total, passed=successful, failed=total - successful,
        success_rate=successful / total if total else 0,
        average_latency_ms=mean(result.metrics.latency_ms for result in results) if results else 0,
        average_steps=mean(result.metrics.steps for result in results) if results else 0,
        average_tool_calls=mean(result.metrics.tool_calls for result in results) if results else 0,
        average_aggregate_score=mean(result.metrics.aggregate_score for result in results) if results else 0,
    )


async def run_local(cases: Sequence[EvaluationCase], settings: Settings, persist: bool) -> EvaluationReport:
    async with httpx.AsyncClient() as http:
        llm = OllamaLLM(settings, http)
        registry = build_tool_registry(http)
        if persist:
            engine = create_database_engine(settings)
            try:
                sessions = async_sessionmaker(engine)
                memory = MemoryManager.using_database(settings, engine, OllamaEmbedder(settings, http), llm)
                service = AgentService(AgentExecutor(settings, llm, registry, memory=memory), AgentRunRepository(sessions))

                async def execute(case: EvaluationCase) -> AgentRunResponse:
                    return await service.run(
                        user_id=settings.api_user_id, thread_id=None, task=case.input,
                        permissions=BUILTIN_PERMISSIONS,
                    )

                return await run_cases(cases, execute, store=EvaluationRepository(sessions).save)
            finally:
                await engine.dispose()

        executor = AgentExecutor(settings, llm, registry)

        async def execute_ephemeral(case: EvaluationCase) -> AgentRunResponse:
            state = await executor.run(
                user_id=settings.api_user_id, thread_id=str(uuid4()), task=case.input,
                permissions=BUILTIN_PERMISSIONS,
            )
            return AgentRunResponse.model_validate(state)

        return await run_cases(cases, execute_ephemeral)


def format_report(report: EvaluationReport) -> str:
    rows = [
        f"Total cases: {report.total}", f"Passed: {report.passed}", f"Failed: {report.failed}",
        f"Success rate: {report.success_rate:.0%}",
        f"Average latency: {report.average_latency_ms:.1f} ms",
        f"Average steps: {report.average_steps:.2f}",
        f"Average tool calls: {report.average_tool_calls:.2f}",
        f"Average aggregate score: {report.average_aggregate_score:.3f}",
    ]
    rows.extend(f"{'PASS' if result.passed else 'FAIL'} {result.case_id}" for result in report.results)
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the agent against a JSON dataset.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[2] / "evals" / "agent_cases.json")
    parser.add_argument("--persist", action="store_true", help="Use migrated PostgreSQL and save results.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.min_success_rate <= 1:
        parser.error("--min-success-rate must be between 0 and 1")
    try:
        report = asyncio.run(run_local(load_cases(args.dataset), Settings(), args.persist))
    except (AgentError, ValidationError) as error:
        parser.exit(2, f"Evaluation failed: {type(error).__name__}.\n")
    print(report.model_dump_json(indent=2) if args.json else format_report(report))
    return int(report.success_rate < args.min_success_rate)


if __name__ == "__main__":
    raise SystemExit(main())
