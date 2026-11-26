import asyncio

import httpx
import pytest

from app.core.exceptions import ToolError, ToolInputError, TransientToolError
from app.tools.weather import weather_tool


def run_weather(handler):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await weather_tool(http).invoke({"location": "New York, US"})

    return asyncio.run(run())


def test_weather_resolves_location_and_returns_rain_forecast():
    def handler(request):
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "New York, US"
            return httpx.Response(200, json={"results": [{
                "name": "New York", "latitude": 40.71, "longitude": -74.01,
                "admin1": "New York", "country": "United States",
            }]})
        assert request.url.host == "api.open-meteo.com"
        assert request.url.params["latitude"] == "40.71"
        assert request.url.params["daily"] == "precipitation_probability_max"
        return httpx.Response(200, json={
            "timezone": "America/New_York",
            "current": {
                "time": "2026-11-26T08:00", "temperature_2m": 8.5,
                "precipitation": 1.2, "weather_code": 61,
            },
            "daily": {"time": ["2026-11-26"], "precipitation_probability_max": [85]},
        })

    result = run_weather(handler)
    assert result == {
        "location": "New York, United States",
        "timezone": "America/New_York",
        "observed_at": "2026-11-26T08:00",
        "temperature_celsius": 8.5,
        "precipitation_mm": 1.2,
        "weather_code": 61,
        "forecast_date": "2026-11-26",
        "max_precipitation_probability_percent": 85,
    }


@pytest.mark.parametrize(("status", "payload", "error_type"), [
    (200, {}, ToolInputError),
    (200, {"results": [{"name": "New York"}]}, ToolError),
    (401, {"error": "private upstream detail"}, ToolError),
    (429, {}, TransientToolError),
    (503, {}, TransientToolError),
])
def test_weather_classifies_failures(status, payload, error_type):
    with pytest.raises(error_type) as error:
        run_weather(lambda request: httpx.Response(status, json=payload))
    assert type(error.value) is error_type
    assert "private upstream detail" not in str(error.value)
