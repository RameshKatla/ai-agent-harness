import asyncio
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.agent.executor import AgentExecutor
from app.core.config import Settings
from app.evaluation import runner
from app.evaluation.datasets import load_cases
from app.llm.ollama_client import OllamaLLM
from app.schemas.agent import AgentRunResponse
from app.schemas.evaluation import EvaluationCase, EvaluationReport
from app.tools import build_tool_registry

ROOT = Path(__file__).resolve().parents[2]


def test_committed_dataset_has_at_least_twenty_diverse_cases():
    cases = load_cases(ROOT / "evals/agent_cases.json")
    assert len(cases) >= 20
    assert any(not case.tools for case in cases)
    assert {"calculator", "datetime"} <= {tool for case in cases for tool in case.tools}
    assert any(case.criteria for case in cases)
    assert all(case.max_steps is not None and case.max_tool_calls is not None for case in cases)


def test_script_entrypoint_runs_from_another_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_evals.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "--min-success-rate" in result.stdout


@pytest.mark.parametrize(("threshold", "exit_status"), [("0.5", 0), ("0.51", 1)])
def test_cli_exits_nonzero_below_configured_success_rate(monkeypatch, capsys, threshold, exit_status):
    async def run_local(cases, settings, persist):
        assert len(cases) >= 20
        return EvaluationReport(
            results=[], total=2, passed=1, failed=1, success_rate=0.5,
            average_latency_ms=10, average_steps=1, average_tool_calls=0, average_aggregate_score=0.5,
        )

    monkeypatch.setattr(runner, "run_local", run_local)
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--json", "--min-success-rate", threshold])
    assert runner.main() == exit_status
    assert json.loads(capsys.readouterr().out)["success_rate"] == 0.5


def test_weather_umbrella_regression_with_a_fixed_forecast():
    case = EvaluationCase(
        id="weather_umbrella", input="Find the weather in New York. Do I need an umbrella?",
        expected="yes", criteria=["umbrella"], tools=["weather"],
        max_steps=2, max_tool_calls=1, max_retries=0,
    )

    def upstream(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "New York"
            return httpx.Response(200, json={"results": [{
                "name": "New York", "latitude": 40.71, "longitude": -74.01,
            }]})
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(200, json={
                "timezone": "America/New_York",
                "current": {"time": "2026-11-26T08:00", "temperature_2m": 8, "precipitation": 3, "weather_code": 61},
                "daily": {"time": ["2026-11-26"], "precipitation_probability_max": [90]},
            })
        assert request.url.host == "ollama.test"
        messages = json.loads(request.content)["messages"]
        if messages[-1]["role"] == "tool":
            result = json.loads(messages[-1]["content"])["result"]
            assert result["max_precipitation_probability_percent"] == 90
            decision = {"action": "final", "answer": "Yes, bring an umbrella."}
        else:
            decision = {"action": "tool", "name": "weather", "arguments": {"location": "New York"}}
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": json.dumps({"decision": decision})}, "done": True,
        })

    async def exercise():
        settings = Settings(_env_file=None, ollama_base_url="http://ollama.test")
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            executor = AgentExecutor(settings, OllamaLLM(settings, client), build_tool_registry(client))

            async def execute(item):
                state = await executor.run(
                    user_id="regression", thread_id=str(uuid4()), task=item.input, permissions={"weather:use"},
                )
                return AgentRunResponse.model_validate(state)

            return await runner.run_cases([case], execute)

    report = asyncio.run(exercise())
    assert report.passed == 1
    assert report.results[0].metrics.tool_success == 1
