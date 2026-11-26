import asyncio
from dataclasses import replace
from datetime import datetime

import httpx
import pytest

from app.core.exceptions import PolicyViolationError, ToolInputError, ToolTimeoutError
from app.tools import build_tool_registry
from app.tools.calculator import CALCULATOR
from app.tools.datetime_tool import DATETIME
from app.tools.registry import ToolRegistry


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("25 * 4", 100), (" (2 + 3) ** 2 ", 25), ("-7 // 2 + 1.5", -2.5)],
)
def test_calculator(expression, expected):
    assert asyncio.run(CALCULATOR.invoke({"expression": expression})) == expected


@pytest.mark.parametrize("expression", [
    "abs(1)", "(1).__class__", "True + 2", "1/0", "10 ** (10 ** 10)",
    "1e999", "(-1)**0.5", "1+" * 150, "1 +", 10,
])
def test_calculator_rejects_unsupported_input(expression):
    with pytest.raises(ToolInputError):
        asyncio.run(CALCULATOR.invoke({"expression": expression}))


def test_registry_permissions_and_validation():
    async def exercise():
        async with httpx.AsyncClient() as http:
            registry = build_tool_registry(http)
            assert [tool.name for tool in registry.available({"weather:use"})] == ["weather"]
            assert registry.get("calculator").input_schema.model_json_schema()["required"] == [
                "expression"
            ]
            with pytest.raises(ValueError, match="already registered"):
                registry.register(CALCULATOR)
            with pytest.raises(ToolInputError, match="Unknown tool"):
                registry.get("unknown")
            with pytest.raises(PolicyViolationError):
                await registry.execute("calculator", {"expression": "1"}, permissions=set())
            with pytest.raises(ToolInputError):
                await registry.execute(
                    "calculator", {"expression": "1", "extra": True},
                    permissions={"calculator:use"},
                )

    asyncio.run(exercise())


def test_tool_timeout():
    async def slow(arguments):
        await asyncio.sleep(10)

    registry = ToolRegistry()
    registry.register(replace(CALCULATOR, timeout_seconds=0.001, implementation=slow))
    with pytest.raises(ToolTimeoutError):
        asyncio.run(registry.execute(
            "calculator", {"expression": "1"}, permissions={"calculator:use"}
        ))


@pytest.mark.parametrize("arguments", [{}, {"timezone": "America/New_York"}])
def test_datetime_returns_zoned_timestamp(arguments):
    result = asyncio.run(DATETIME.invoke(arguments))
    assert result["timezone"] == arguments.get("timezone", "UTC")
    assert datetime.fromisoformat(result["datetime"]).utcoffset() is not None


def test_datetime_rejects_invalid_timezone():
    with pytest.raises(ToolInputError):
        asyncio.run(DATETIME.invoke({"timezone": "../UTC"}))
