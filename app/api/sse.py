from collections.abc import AsyncGenerator
from contextlib import aclosing

from anyio import CancelScope
from starlette.responses import StreamingResponse
from starlette.types import Send

from app.observability.events import AgentEvent


class EventStreamResponse(StreamingResponse):
    def __init__(self, events: AsyncGenerator[AgentEvent | None, None]) -> None:
        self._frames = self._encode(events)
        super().__init__(
            self._frames, media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            with CancelScope(shield=True):
                await self._frames.aclose()

    @staticmethod
    async def _encode(
        events: AsyncGenerator[AgentEvent | None, None]
    ) -> AsyncGenerator[str, None]:
        async with aclosing(events):
            async for event in events:
                if event is None:
                    yield ": keep-alive\n\n"
                else:
                    yield (
                        f"id: {event.run_id}:{event.sequence}\n"
                        f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
                    )
