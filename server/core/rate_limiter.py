"""Token-Bucket Rate Limiter for Google AI Studio Free Tier (RPM & TPM Tracking)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Any


@dataclass
class TokenBucketRateLimiter:
    """Tracks sliding-window Requests Per Minute (RPM) and Tokens Per Minute (TPM)
    so J.A.R.V.I.S. seamlessly spills over to local Ollama before hitting hard API errors.
    """

    rpm_limit: int = 15
    tpm_limit: int = 250000
    window_seconds: float = 60.0
    _request_timestamps: Deque[float] = field(default_factory=deque)
    _token_events: Deque[tuple[float, int]] = field(default_factory=deque)
    total_cloud_requests: int = 0
    total_local_requests: int = 0
    total_fallback_requests: int = 0
    last_latency_ms: float = 0.0
    active_tier: str = "standby"

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        while self._token_events and self._token_events[0][0] < cutoff:
            self._token_events.popleft()

    @property
    def current_rpm(self) -> int:
        self._prune(time.time())
        return len(self._request_timestamps)

    @property
    def current_tpm(self) -> int:
        self._prune(time.time())
        return sum(tokens for _, tokens in self._token_events)

    def can_use_cloud(self, estimated_tokens: int = 1200) -> bool:
        now = time.time()
        self._prune(now)
        if len(self._request_timestamps) >= self.rpm_limit:
            return False
        if self.current_tpm + estimated_tokens > self.tpm_limit:
            return False
        return True

    def record_cloud_usage(self, tokens_used: int, latency_ms: float) -> None:
        now = time.time()
        self._prune(now)
        self._request_timestamps.append(now)
        self._token_events.append((now, max(1, tokens_used)))
        self.total_cloud_requests += 1
        self.last_latency_ms = round(latency_ms, 1)
        self.active_tier = "gemini-cloud"

    def record_local_usage(self, latency_ms: float, tier_name: str = "ollama-local") -> None:
        if tier_name == "ollama-local":
            self.total_local_requests += 1
        else:
            self.total_fallback_requests += 1
        self.last_latency_ms = round(latency_ms, 1)
        self.active_tier = tier_name

    def snapshot(self) -> Dict[str, Any]:
        rpm = self.current_rpm
        tpm = self.current_tpm
        return {
            "rpm_current": rpm,
            "rpm_limit": self.rpm_limit,
            "rpm_pct": round((rpm / max(1, self.rpm_limit)) * 100, 1),
            "tpm_current": tpm,
            "tpm_limit": self.tpm_limit,
            "tpm_pct": round((tpm / max(1, self.tpm_limit)) * 100, 1),
            "total_cloud_requests": self.total_cloud_requests,
            "total_local_requests": self.total_local_requests,
            "total_fallback_requests": self.total_fallback_requests,
            "last_latency_ms": self.last_latency_ms,
            "active_tier": self.active_tier,
        }

