from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from app.core.exceptions import ToolInputError
from app.tools.registry import RetryPolicy, Tool, ToolInput


class DatetimeInput(ToolInput):
    timezone: str = Field(default="UTC", min_length=1, max_length=128)


async def current_datetime(arguments: DatetimeInput) -> dict[str, str]:
    try:
        zone = ZoneInfo(arguments.timezone)
    except (ValueError, ZoneInfoNotFoundError):
        raise ToolInputError("Use a valid IANA timezone such as America/New_York.") from None
    return {"timezone": zone.key, "datetime": datetime.now(zone).isoformat()}


DATETIME = Tool(
    name="datetime",
    description="Return the current date and time for an IANA timezone, defaulting to UTC.",
    input_schema=DatetimeInput,
    permission="datetime:use",
    timeout_seconds=1,
    retry_policy=RetryPolicy(),
    implementation=current_datetime,
)
