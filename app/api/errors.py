from http import HTTPStatus
from typing import cast

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.core.exceptions import (
    AgentError,
    ApprovalConflictError,
    EvaluationError,
    LLMError,
    MemoryError,
    PolicyViolationError,
    RedisUnavailableError,
    StorageError,
    ToolInputError,
    ToolTimeoutError,
    TransientToolError,
)


class ValidationIssue(BaseModel):
    source: str
    code: str


class ErrorResponse(BaseModel):
    code: str
    detail: str
    request_id: str
    issues: list[ValidationIssue] | None = Field(default=None)


def error_response(
    status: int, code: str, detail: str, request_id: str, *,
    issues: list[ValidationIssue] | None = None, headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(code=code, detail=detail, request_id=request_id, issues=issues)
    return JSONResponse(
        status_code=status, content=body.model_dump(exclude_none=True), headers=headers,
    )


_HTTP_ERRORS = {
    "API access is not configured.": "authentication_unavailable",
    "Invalid or missing bearer token.": "unauthorized",
    "Rate limit exceeded.": "rate_limited",
    "Thread not found.": "not_found",
    "Memory not found.": "not_found",
}


async def http_error(request: Request, error: Exception) -> JSONResponse:
    error = cast(HTTPException, error)
    if isinstance(error.detail, str) and error.detail in _HTTP_ERRORS:
        code, detail = _HTTP_ERRORS[error.detail], error.detail
    else:
        try:
            status = HTTPStatus(error.status_code)
            code, detail = status.name.lower(), status.phrase
        except ValueError:
            code, detail = "http_error", "The request could not be completed."
    return error_response(error.status_code, code, detail, request.state.request_id, headers=error.headers)


async def validation_error(request: Request, error: Exception) -> JSONResponse:
    error = cast(RequestValidationError, error)
    issues = []
    for item in error.errors()[:20]:
        source = item["loc"][0] if item["loc"] else "request"
        if source not in {"body", "path", "query", "header", "cookie"}:
            source = "request"
        issues.append(ValidationIssue(source=source, code=item["type"]))
    return error_response(
        422, "validation_error", "Request validation failed.", request.state.request_id, issues=issues,
    )


_AGENT_ERRORS: tuple[tuple[type[AgentError], int, str, str], ...] = (
    (ApprovalConflictError, 409, "approval_conflict", "The thread is not ready for this approval action."),
    (MemoryError, 503, "memory_unavailable", "The memory operation is temporarily unavailable."),
    (StorageError, 503, "storage_unavailable", "The storage operation is temporarily unavailable."),
    (RedisUnavailableError, 503, "rate_limit_unavailable", "Rate limiting is temporarily unavailable."),
    (LLMError, 502, "model_unavailable", "The model request failed."),
    (ToolInputError, 422, "invalid_tool_input", "Invalid tool arguments."),
    (ToolTimeoutError, 504, "tool_timeout", "The tool exceeded its execution deadline."),
    (TransientToolError, 503, "tool_unavailable", "The tool is temporarily unavailable."),
    (PolicyViolationError, 403, "policy_violation", "The operation was denied by policy."),
    (EvaluationError, 422, "evaluation_error", "The evaluation request could not be processed."),
    (AgentError, 500, "agent_error", "The agent request could not be completed."),
)


async def agent_error(request: Request, error: Exception) -> JSONResponse:
    for kind, status, code, detail in _AGENT_ERRORS:
        if isinstance(error, kind):
            return error_response(status, code, detail, request.state.request_id)
    return error_response(500, "internal_error", "An internal error occurred.", request.state.request_id)
