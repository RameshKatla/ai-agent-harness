import httpx

from app.core.redis import RedisServices
from app.tools.calculator import CALCULATOR
from app.tools.datetime_tool import DATETIME
from app.tools.registry import ToolRegistry
from app.tools.weather import weather_tool


BUILTIN_PERMISSIONS = frozenset({"calculator:use", "datetime:use", "weather:use"})


def build_tool_registry(
    http: httpx.AsyncClient, *, cache: RedisServices | None = None, weather_cache_ttl: int = 120,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CALCULATOR)
    registry.register(DATETIME)
    registry.register(weather_tool(http, cache=cache, cache_ttl=weather_cache_ttl))
    return registry
