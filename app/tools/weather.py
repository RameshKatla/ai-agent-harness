import httpx
from pydantic import BaseModel, Field

from app.core.exceptions import ToolError, ToolInputError, TransientToolError
from app.core.redis import RedisServices
from app.tools.registry import RetryPolicy, Tool, ToolInput

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherInput(ToolInput):
    location: str = Field(min_length=2, max_length=120)


class Place(BaseModel):
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    admin1: str | None = None
    country: str | None = None


class Locations(BaseModel):
    results: list[Place] = Field(default_factory=list)


class CurrentWeather(BaseModel):
    time: str
    temperature_2m: float = Field(allow_inf_nan=False)
    precipitation: float = Field(ge=0, allow_inf_nan=False)
    weather_code: int


class DailyWeather(BaseModel):
    time: list[str] = Field(min_length=1, max_length=1)
    precipitation_probability_max: list[int | None] = Field(min_length=1, max_length=1)


class Forecast(BaseModel):
    timezone: str
    current: CurrentWeather
    daily: DailyWeather


class WeatherResult(BaseModel):
    location: str
    timezone: str
    observed_at: str
    temperature_celsius: float
    precipitation_mm: float
    weather_code: int
    forecast_date: str
    max_precipitation_probability_percent: int | None


async def _get[Response: BaseModel](
    http: httpx.AsyncClient, url: str, params: dict[str, str | int | float],
    schema: type[Response],
) -> Response:
    try:
        response = await http.get(url, params=params, follow_redirects=False)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status in (408, 429) or status >= 500:
            raise TransientToolError("The weather provider is temporarily unavailable.") from None
        raise ToolError("The weather provider rejected the request.") from None
    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
        raise TransientToolError("The weather provider could not be reached.") from None
    except httpx.RequestError:
        raise ToolError("The weather request failed.") from None

    try:
        return schema.model_validate_json(response.content)
    except ValueError:
        raise ToolError("The weather provider returned an invalid response.") from None


def weather_tool(
    http: httpx.AsyncClient, *, cache: RedisServices | None = None, cache_ttl: int = 120,
) -> Tool[WeatherInput]:
    async def forecast(arguments: WeatherInput) -> dict[str, object]:
        key = " ".join(arguments.location.split()).casefold()
        if cache is not None:
            cached = await cache.get_cached("weather-v1", key, WeatherResult)
            if cached is not None:
                return cached.model_dump()
        locations = await _get(
            http, GEOCODING_URL, {"name": arguments.location, "count": 1}, Locations
        )
        if not locations.results:
            raise ToolInputError("Location not found. Try adding a region or country.")
        place = locations.results[0]
        weather = await _get(
            http,
            FORECAST_URL,
            {
                "latitude": place.latitude,
                "longitude": place.longitude,
                "timezone": "auto",
                "current": "temperature_2m,precipitation,weather_code",
                "daily": "precipitation_probability_max",
                "forecast_days": 1,
            },
            Forecast,
        )
        result = WeatherResult(
            location=", ".join(dict.fromkeys(
                name for name in (place.name, place.admin1, place.country) if name
            )),
            timezone=weather.timezone,
            observed_at=weather.current.time,
            temperature_celsius=weather.current.temperature_2m,
            precipitation_mm=weather.current.precipitation,
            weather_code=weather.current.weather_code,
            forecast_date=weather.daily.time[0],
            max_precipitation_probability_percent=weather.daily.precipitation_probability_max[0],
        )
        if cache is not None:
            await cache.set_cached("weather-v1", key, result, ttl=cache_ttl)
        return result.model_dump()

    return Tool(
        name="weather",
        description="Get current conditions and today's maximum precipitation probability for a location.",
        input_schema=WeatherInput,
        permission="weather:use",
        timeout_seconds=10,
        retry_policy=RetryPolicy(max_retries=2),
        implementation=forecast,
    )
