import logging
from uuid import uuid4

from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response
from app.observability.logging import request_id_context

logger = logging.getLogger(__name__)


class RequestSecurityMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        context_token = request_id_context.set(request_id)
        status = None
        finished = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, finished
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [
                    (key, value) for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            elif message["type"] == "http.response.body":
                finished = not message.get("more_body", False)
            await send(message)

        try:
            result = await self._read_body(scope, receive, send_with_id, request_id)
            if result is None:
                return

            body = bytes(result)
            del result
            delivered = False

            async def replay_body() -> Message:
                nonlocal delivered, body
                if delivered:
                    return await receive()
                delivered = True
                message = {"type": "http.request", "body": body, "more_body": False}
                body = b""
                return message

            await self.app(scope, replay_body, send_with_id)
        except (ClientDisconnect, OSError) as error:
            if isinstance(error, ClientDisconnect) or status is not None:
                return
            await self._unexpected(scope, receive, send_with_id, request_id, status, finished)
        except Exception:
            await self._unexpected(scope, receive, send_with_id, request_id, status, finished)
        finally:
            if status is not None:
                route = scope.get("route")
                logger.info("HTTP request", extra={
                    "http_request": {
                        "route": getattr(route, "path", None),
                        "status": status,
                    },
                })
            request_id_context.reset(context_token)

    async def _read_body(
        self, scope: Scope, receive: Receive, send: Send, request_id: str,
    ) -> bytearray | None:
        lengths = [value for key, value in scope["headers"] if key.lower() == b"content-length"]
        if lengths:
            length = lengths[0]
            if len(lengths) != 1 or len(length) > 20 or not length.isdigit():
                await error_response(
                    400, "invalid_content_length", "Invalid Content-Length header.", request_id,
                )(scope, receive, send)
                return None
            if int(length) > self._max_body_bytes:
                await self._too_large(scope, receive, send, request_id)
                return None

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self._max_body_bytes:
                await self._too_large(scope, receive, send, request_id)
                return None
            body.extend(chunk)
            if not message.get("more_body", False):
                return body

    @staticmethod
    async def _too_large(scope: Scope, receive: Receive, send: Send, request_id: str) -> None:
        await error_response(
            413, "request_too_large", "Request body exceeds the configured limit.", request_id,
        )(scope, receive, send)

    @staticmethod
    async def _unexpected(
        scope: Scope, receive: Receive, send: Send, request_id: str, status: int | None, finished: bool,
    ) -> None:
        logger.error("Unhandled HTTP request error")
        if status is None:
            await error_response(
                500, "internal_error", "An internal error occurred.", request_id,
            )(scope, receive, send)
        elif not finished:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
