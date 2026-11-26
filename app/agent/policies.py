import logging
from collections.abc import Collection
from time import perf_counter

from app.agent.state import AgentState, Observation, ToolAttempt, ToolCall
from app.core.config import Settings
from app.core.exceptions import PolicyViolationError, ToolError, ToolTimeoutError, TransientToolError
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PolicyEngine:
    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self._settings = settings
        self._registry = registry

    def check_agent_step(self, state: AgentState) -> None:
        if state["current_step"] >= self._settings.max_agent_steps:
            raise PolicyViolationError("Agent step limit reached.")

    def check_tool_call(self, state: AgentState) -> None:
        if len(state["tool_calls"]) >= self._settings.max_tool_calls:
            raise PolicyViolationError("Tool-call limit reached.")
        self.check_agent_step(state)

    def requires_approval(self, call: ToolCall, permissions: Collection[str]) -> bool:
        try:
            tool = self._registry.get(call["name"])
            if tool.permission not in permissions:
                return False
            if not (tool.requires_approval or tool.name in self._settings.approval_required_tools):
                return False
            tool.validate_arguments(call["arguments"])
        except ToolError:
            return False
        return True

    async def execute(
        self, call: ToolCall, permissions: Collection[str]
    ) -> Observation:
        attempts: list[ToolAttempt] = []
        observation: Observation = {
            "tool": call["name"], "result": None, "success": False, "attempts": attempts,
        }
        try:
            tool = self._registry.get(call["name"])
            if tool.permission not in permissions:
                raise PolicyViolationError("Permission denied for the requested tool.")
        except (ToolError, PolicyViolationError) as error:
            observation["result"] = str(error)
            return observation

        retries = min(self._settings.max_retries, tool.retry_policy.max_retries)
        for attempt in range(1, retries + 2):
            started = perf_counter()
            error_message = None
            should_retry = False
            try:
                observation["result"] = await tool.invoke(call["arguments"])
                observation["success"] = True
            except ToolError as error:
                error_message = str(error)
                observation["result"] = error_message
                should_retry = isinstance(error, (TransientToolError, ToolTimeoutError))
            latency_ms = (perf_counter() - started) * 1000
            attempts.append({
                "attempt": attempt,
                "latency_ms": latency_ms,
                "error": error_message,
                "success": observation["success"],
            })
            logger.info(
                "Tool %s attempt %d success=%s latency_ms=%.1f error_type=%s",
                tool.name, attempt, observation["success"], latency_ms,
                "transient" if should_retry else "permanent" if error_message else None,
            )
            if not should_retry:
                break
        return observation
