import httpx

from app.tools.calculator import CALCULATOR
from app.tools.datetime_tool import DATETIME
from app.tools.registry import ToolRegistry
from app.tools.weather import weather_tool


def build_tool_registry(http: httpx.AsyncClient) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CALCULATOR)
    registry.register(DATETIME)
    registry.register(weather_tool(http))
    return registry
