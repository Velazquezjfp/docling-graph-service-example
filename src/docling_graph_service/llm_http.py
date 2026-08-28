"""One small httpx client for the OpenAI-compatible endpoints (chat completions, embeddings)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMHTTPError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LLMClient:
    """Bearer-authenticated JSON client with bounded retries (429/5xx/timeouts/connect errors)."""

    def __init__(self, base_url: str, api_key: str | None, timeout_s: float, *, max_attempts: int = 3,
                 backoff_s: float = 1.0, transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def post_json(self, path: str, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                kwargs: dict[str, Any] = {"json": payload}
                if timeout_s is not None:
                    kwargs["timeout"] = httpx.Timeout(timeout_s, connect=10.0)
                resp = self._client.post(path, **kwargs)
                if resp.status_code in RETRY_STATUS and attempt < self.max_attempts:
                    last = LLMHTTPError(f"{path}: HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
                    self._sleep(attempt)
                    continue
                if resp.status_code >= 400:
                    raise LLMHTTPError(f"{path}: HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
                return resp.json()
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last = exc
                if attempt < self.max_attempts:
                    self._sleep(attempt)
                    continue
        raise LLMHTTPError(f"{path}: giving up after {self.max_attempts} attempts: {last}") from last

    def get_json(self, path: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
        resp = self._client.get(path, timeout=timeout_s)
        if resp.status_code >= 400:
            raise LLMHTTPError(f"{path}: HTTP {resp.status_code}", resp.status_code)
        return resp.json()

    def probe(self) -> str:
        """Cheap liveness check for /v1/capabilities: 'ok', or the error text."""
        try:
            self.get_json("/models", timeout_s=5.0)
            return "ok"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return f"unreachable: {exc}"[:200]

    def _sleep(self, attempt: int) -> None:
        delay = self.backoff_s * (2 ** (attempt - 1))
        log.warning("LLM endpoint retry %d/%d in %.1fs", attempt, self.max_attempts, delay)
        time.sleep(delay)
