"""core/exceptions.py — Typed application exception hierarchy.

HTTP status code mapping:
  ValidationError      → 400  (bad client input)
  BusinessRuleError    → 422  (valid input, but violates domain constraints)
  NotFoundError        → 404
  InternalError        → 500  (unexpected server failure)

Stack traces are NEVER exposed to clients — they are logged server-side only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class BaseAppException(Exception):
    """Root exception for all application-defined errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details: Dict[str, Any] = details or {}
        self.error_code = error_code or self.__class__.__name__


class ValidationError(BaseAppException):
    """Raised when client input fails schema or format validation (HTTP 400)."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "VALIDATION_ERROR",
    ) -> None:
        super().__init__(message, status_code=400, details=details, error_code=error_code)


class BusinessRuleError(BaseAppException):
    """Raised when a request violates a domain business rule (HTTP 422).

    Use this for semantically valid requests that cannot be fulfilled due to
    warehouse constraints (e.g. zero walkable pick nodes, no feasible batching).
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "BUSINESS_RULE_VIOLATION",
    ) -> None:
        super().__init__(message, status_code=422, details=details, error_code=error_code)


class NotFoundError(BaseAppException):
    """Raised when a requested resource does not exist (HTTP 404)."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "NOT_FOUND",
    ) -> None:
        super().__init__(message, status_code=404, details=details, error_code=error_code)


class InternalError(BaseAppException):
    """Raised for unexpected internal failures (HTTP 500).

    Message shown to client must NOT contain stack traces or internal paths.
    """

    def __init__(
        self,
        message: str = "An unexpected internal error occurred.",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "INTERNAL_ERROR",
    ) -> None:
        super().__init__(message, status_code=500, details=details, error_code=error_code)
