"""ORCH Director: the single disciplined conductor of a task-stage attempt.

:class:`OrchDirector` runs one ``(task_id, stage, seat, attempt)`` attempt
through the full NULL lifecycle in a fixed order that callers cannot
reorder: input validation, the F4 credential gate, key derivation,
single-owner lease acquisition, exactly-once replay, execution, call
capture, the matrix-enforced cache write, and a best-effort lease
release.  The pinned invariant is the F4 contract: no lease, no
capture, and no cache write ever happen without a credential that
:func:`~nullcore.orch.cache_matrix.verify_credential` has already
accepted, and the executor never runs while a captured replay already
holds the byte-identical response.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping

from nullcore.orch.cache_key import make_cache_key
from nullcore.orch.cache_matrix import CacheStatus, CacheTier, verify_credential
from nullcore.orch.cache_store import CacheStore
from nullcore.orch.capture import CallCapture, NullReplayReader, ReplayNotFoundError
from nullcore.orch.idempotency import make_idempotency_key
from nullcore.orch.lease import LeaseError, LeaseManager
from nullcore.orch.state_machine import OrchState

__all__ = ["DirectorError", "OrchDirector"]


class DirectorError(Exception):
    """Raised when director inputs are invalid or a step cannot proceed."""


def _require_non_empty_str(value: object, name: str) -> None:
    """Raise DirectorError unless value is a non-empty str."""
    if not isinstance(value, str):
        raise DirectorError(f"{name} must be a str, got {type(value).__name__}")
    if not value:
        raise DirectorError(f"{name} must be a non-empty str")


def _require_non_negative_int(value: object, name: str) -> None:
    """Raise DirectorError unless value is an int >= 0 (bool rejected)."""
    if isinstance(value, bool):
        raise DirectorError(f"{name} must be an int, got bool")
    if not isinstance(value, int):
        raise DirectorError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise DirectorError(f"{name} must be >= 0, got {value}")


class OrchDirector:
    """Conduct one task-stage attempt through the fixed NULL lifecycle."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock_ms: Callable[[], int],
        lease_ttl_ms: int = 30_000,
    ) -> None:
        """Validate inputs and eagerly initialize every schema."""
        if not isinstance(conn, sqlite3.Connection):
            raise DirectorError(
                f"conn must be a sqlite3.Connection, got {type(conn).__name__}"
            )
        if not callable(clock_ms):
            raise DirectorError("clock_ms must be callable")
        if isinstance(lease_ttl_ms, bool) or not isinstance(lease_ttl_ms, int):
            raise DirectorError(
                "lease_ttl_ms must be a positive int, "
                f"got {type(lease_ttl_ms).__name__}"
            )
        if lease_ttl_ms <= 0:
            raise DirectorError(
                f"lease_ttl_ms must be a positive int, got {lease_ttl_ms}"
            )
        self._lease = LeaseManager(conn, clock_ms)
        self._capture = CallCapture(conn)
        self._reader = NullReplayReader(conn)
        self._cache = CacheStore(conn)
        self._clock_ms = clock_ms
        self._lease_ttl_ms = lease_ttl_ms

    def run_attempt(
        self,
        task_id: str,
        stage: str,
        seat: str,
        attempt: int,
        lifecycle_state: OrchState,
        stage_vector: Mapping[str, str],
        tier: CacheTier,
        status: CacheStatus,
        degraded_flag: bool,
        config_hash: str,
        request_bytes: bytes,
        executor: Callable[[bytes], bytes],
    ) -> tuple[bytes, bool]:
        """Run one attempt through the fixed NULL lifecycle.

        Returns ``(response_bytes, from_replay)`` where ``from_replay``
        is True exactly when a captured replay served the response and
        the executor was not called.
        """
        # 1. INPUT VALIDATION: reject every malformed argument first.
        _require_non_empty_str(task_id, "task_id")
        _require_non_empty_str(stage, "stage")
        _require_non_empty_str(seat, "seat")
        _require_non_empty_str(config_hash, "config_hash")
        _require_non_negative_int(attempt, "attempt")
        if not isinstance(lifecycle_state, OrchState):
            raise DirectorError(
                "lifecycle_state must be an OrchState member, "
                f"got {type(lifecycle_state).__name__}"
            )
        if not isinstance(tier, CacheTier):
            raise DirectorError(
                f"tier must be a CacheTier member, got {type(tier).__name__}"
            )
        if not isinstance(status, CacheStatus):
            raise DirectorError(
                f"status must be a CacheStatus member, got {type(status).__name__}"
            )
        if not isinstance(degraded_flag, bool):
            raise DirectorError(
                "degraded_flag must be a bool, "
                f"got {type(degraded_flag).__name__}"
            )
        if not isinstance(request_bytes, bytes):
            raise DirectorError(
                f"request_bytes must be bytes, got {type(request_bytes).__name__}"
            )
        if not callable(executor):
            raise DirectorError("executor must be callable")
        # 2. CREDENTIAL GATE (F4 CONTRACT): zero side effects before this
        #    passes; a MatrixViolationError propagates unchanged.
        verify_credential(lifecycle_state, stage_vector, status)
        # 3. DERIVE KEYS.
        idem_key = make_idempotency_key(task_id, stage, seat, attempt)
        cache_key = make_cache_key(config_hash, task_id, stage, seat, attempt)
        # 4. LEASE ACQUIRE: single-owner gating per attempt token.
        upgrade_token = idem_key
        owner_id = seat
        acquired = self._lease.acquire(upgrade_token, owner_id, self._lease_ttl_ms)
        if not acquired:
            raise DirectorError("lease held by another owner")
        try:
            # 5. REPLAY CHECK (exactly-once).
            try:
                cached_response = self._reader.replay(upgrade_token)
            except ReplayNotFoundError:
                pass
            else:
                return cached_response, True
            # 6. EXECUTE.
            response_bytes = executor(request_bytes)
            if not isinstance(response_bytes, bytes):
                raise DirectorError("executor must return bytes")
            # 7. CAPTURE STORE.
            self._capture.store(
                upgrade_token,
                task_id,
                stage,
                seat,
                attempt,
                request_bytes,
                response_bytes,
                self._clock_ms,
            )
            # 8. CACHE WRITE: store re-checks matrix and flag invariant.
            self._cache.write(
                cache_key,
                config_hash,
                task_id,
                stage,
                seat,
                attempt,
                tier,
                status,
                degraded_flag,
                response_bytes,
                self._clock_ms,
            )
            # 10. RETURN on the executed path.
            return response_bytes, False
        finally:
            # 9. RELEASE LEASE: best-effort, always runs once acquired.
            try:
                self._lease.release(upgrade_token, owner_id)
            except LeaseError:
                pass
