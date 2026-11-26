import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime


request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if event := getattr(record, "agent_event", None):
            payload["agent_event"] = event
        if request_id := request_id_context.get():
            payload["request_id"] = request_id
        if request := getattr(record, "http_request", None):
            payload["http_request"] = request
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    logger = logging.getLogger("app")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
