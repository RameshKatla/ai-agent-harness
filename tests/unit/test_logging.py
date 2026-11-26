import json
import logging
from datetime import datetime

from app.observability.logging import JsonFormatter


def test_json_logging_formats_message_and_timestamp():
    record = logging.LogRecord("app", logging.INFO, __file__, 1, "phase %d", (1,), None)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "phase 1"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app"
    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None
