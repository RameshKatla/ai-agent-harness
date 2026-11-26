import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.exceptions import (
    PolicyViolationError,
    ToolError,
    ToolInputError,
    ToolTimeoutError,
)

logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("Retry count must be nonnegative.")


@dataclass(frozen=True)
class Tool[Input: ToolInput]:
    name: str
    description: str
    input_schema: type[Input]
    permission: str
    timeout_seconds: float
    retry_policy: RetryPolicy
    implementation: Callable[[Input], Awaitable[object]]
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Tool timeout must be finite and positive.")

    def validate_arguments(self, arguments: Mapping[str, object]) -> Input:
        try:
            return self.input_schema.model_validate(arguments)
        except ValidationError:
            raise ToolInputError("Invalid arguments for the requested tool.") from None
    async def invoke(self, arguments: Mapping[str, object]) -> object:
        validated = self.validate_arguments(arguments)
        started = perf_counter()
        status = "failed"
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await self.implementation(validated)
            status = "completed"
            return result
        except TimeoutError:
            raise ToolTimeoutError("The tool exceeded its execution deadline.") from None
        except ToolError:
            raise
        except Exception:
            raise ToolError("The tool failed during execution.") from None
        finally:
            logger.info(
                "Tool %s %s in %.1f ms", self.name, status, (perf_counter() - started) * 1000
            )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolInputError("Unknown tool.") from None

    def available(self, permissions: Collection[str]) -> tuple[Tool, ...]:
        return tuple(tool for tool in self._tools.values() if tool.permission in permissions)

    async def execute(
        self, name: str, arguments: Mapping[str, object], *, permissions: Collection[str]
    ) -> object:
        tool = self.get(name)
        if tool.permission not in permissions:
            raise PolicyViolationError("Permission denied for the requested tool.")
        return await tool.invoke(arguments)
