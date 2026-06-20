"""core/error_handlers.py — Global FastAPI exception middleware.

Registers handlers that map every exception type to the correct HTTP status
code and a structured JSON body.  Stack traces are logged server-side and
NEVER forwarded to the client.

Response format (all errors):
  {
    "error":       "<error_code>",
    "message":     "<human-readable description>",
    "details":     { ... },          // optional extra fields
    "request_id":  "<uuid>"          // correlation ID from X-Request-ID header
  }
"""
from __future__ import annotations

import logging
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import BaseAppException

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


def _request_id(request: Request) -> str:
    """Return the correlation ID from the request header, or generate one."""
    return request.headers.get(_REQUEST_ID_HEADER, str(uuid.uuid4()))


def _error_body(
    error_code: str,
    message: str,
    request_id: str,
    details: dict | None = None,
) -> dict:
    return {
        "error": error_code,
        "message": message,
        "details": details or {},
        "request_id": request_id,
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to *app*."""

    @app.exception_handler(BaseAppException)
    async def handle_app_exception(request: Request, exc: BaseAppException) -> JSONResponse:
        """Handle all typed application exceptions with the correct HTTP code."""
        rid = _request_id(request)
        # Log at WARNING for 4xx, ERROR for 5xx
        if exc.status_code >= 500:
            logger.error(
                "app_exception request_id=%s error_code=%s status=%s message=%s",
                rid, exc.error_code, exc.status_code, exc.message,
                exc_info=True,
            )
        else:
            logger.warning(
                "app_exception request_id=%s error_code=%s status=%s message=%s",
                rid, exc.error_code, exc.status_code, exc.message,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.error_code, exc.message, rid, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_pydantic_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Map Pydantic request-body validation failures to HTTP 400."""
        rid = _request_id(request)
        logger.warning("validation_error request_id=%s errors=%s", rid, exc.errors())
        return JSONResponse(
            status_code=400,
            content=_error_body(
                "VALIDATION_ERROR",
                "Request body validation failed.",
                rid,
                {"validation_errors": exc.errors()},
            ),
        )

    @app.exception_handler(PydanticValidationError)
    async def handle_pydantic_internal(request: Request, exc: PydanticValidationError) -> JSONResponse:
        """Map internal Pydantic validation failures to HTTP 422."""
        rid = _request_id(request)
        logger.warning("internal_validation_error request_id=%s", rid, exc_info=True)
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "BUSINESS_RULE_VIOLATION",
                "The request could not be processed due to a constraint violation.",
                rid,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all: any unhandled exception → HTTP 500.

        The stack trace is logged but NEVER included in the client response.
        """
        rid = _request_id(request)
        logger.error(
            "unhandled_exception request_id=%s exc_type=%s",
            rid, type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=_error_body(
                "INTERNAL_ERROR",
                "An unexpected internal error occurred.",
                rid,
            ),
        )
