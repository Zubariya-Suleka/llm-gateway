"""Errors that map to an HTTP status and an OpenAI-style error body."""


class GatewayError(Exception):
    status_code = 500
    error_type = "gateway_error"

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        # Seconds the caller should wait before trying again, if known.
        self.retry_after = retry_after


class BadRequest(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"


class BudgetExhausted(GatewayError):
    """The local limiter refused: a daily budget is used up. Never retried."""

    status_code = 429
    error_type = "rate_limit_error"


class ProviderError(GatewayError):
    """A provider call failed. `retryable` defaults to: 429 or any 5xx."""

    error_type = "provider_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, retry_after=retry_after)
        if retryable is None:
            retryable = status_code == 429 or status_code >= 500
        self.retryable = retryable
