from __future__ import annotations
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final
from nullcore.orch.idempotency import IdempotencyKeyError, make_idempotency_key

__all__ = ["CaptureConflictError", "ReplayNotFoundError", "CallCapture", "NullReplayReader"]
class CaptureConflictError(Exception): pass
class ReplayNotFoundError(Exception): pass

_CALLS_SCHEMA: Final[str] = """CREATE TABLE IF NOT EXISTS calls (idempotency_key TEXT PRIMARY KEY, task_id TEXT NOT NULL, stage TEXT NOT NULL, seat TEXT NOT NULL, attempt INTEGER NOT NULL, request_bytes BLOB NOT NULL, response_bytes BLOB NOT NULL, created_at_ms INTEGER NOT NULL);"""

def _require_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value: raise CaptureConflictError(f"{name} must be a non-empty str")
def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise CaptureConflictError(f"{name} must be an int >= 0")
def _require_bytes(value: object, name: str) -> None:
    if not isinstance(value, bytes): raise CaptureConflictError(f"{name} must be bytes")
def _as_bytes_like(value: object) -> object:
    if isinstance(value, (bytearray, memoryview)): return bytes(value)
    return value

class CallCapture:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn; self._conn.isolation_level = None
        self._conn.execute("PRAGMA busy_timeout=5000;"); self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;"); self._conn.execute(_CALLS_SCHEMA)

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        conn = self._conn; conn.execute("BEGIN IMMEDIATE")
        try: yield
        except BaseException:
            try: conn.execute("ROLLBACK")
            except sqlite3.Error: pass
            raise
        conn.execute("COMMIT")

    def store(self, idempotency_key: str, task_id: str, stage: str, seat: str, attempt: int, request_bytes: bytes, response_bytes: bytes, clock_ms: Callable[[], int]) -> None:
        _require_non_empty_str(idempotency_key, "idempotency_key"); _require_non_empty_str(task_id, "task_id")
        _require_non_empty_str(stage, "stage"); _require_non_empty_str(seat, "seat")
        _require_non_negative_int(attempt, "attempt"); _require_bytes(request_bytes, "request_bytes"); _require_bytes(response_bytes, "response_bytes")
        if not callable(clock_ms): raise CaptureConflictError("clock_ms must be callable")
        try: expected_key = make_idempotency_key(task_id, stage, seat, attempt)
        except IdempotencyKeyError as error: raise CaptureConflictError(f"invalid capture identity: {error}") from error
        if expected_key != idempotency_key: raise CaptureConflictError("idempotency_key does not bind to identity fields")
        created_at_ms = clock_ms()
        if isinstance(created_at_ms, bool) or not isinstance(created_at_ms, int): raise CaptureConflictError("clock_ms must return an int")
        with self._immediate_transaction():
            row = self._conn.execute("SELECT task_id, stage, seat, attempt, request_bytes, response_bytes FROM calls WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if row is None:
                self._conn.execute("INSERT INTO calls (idempotency_key, task_id, stage, seat, attempt, request_bytes, response_bytes, created_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (idempotency_key, task_id, stage, seat, attempt, request_bytes, response_bytes, created_at_ms))
                return
            stored = (row[0], row[1], row[2], row[3], _as_bytes_like(row[4]), _as_bytes_like(row[5]))
            incoming = (task_id, stage, seat, attempt, request_bytes, response_bytes)
            if stored != incoming: raise CaptureConflictError("conflicting capture")

class NullReplayReader:
    def __init__(self, conn: sqlite3.Connection) -> None: self._conn = conn
    def replay(self, idempotency_key: str) -> bytes:
        if not isinstance(idempotency_key, str) or not idempotency_key: raise ReplayNotFoundError("invalid key")
        table_row = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'calls'").fetchone()
        if table_row is None: raise ReplayNotFoundError(f"no captured call for idempotency key: {idempotency_key}")
        row = self._conn.execute("SELECT response_bytes FROM calls WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
        if row is None: raise ReplayNotFoundError(f"no captured call for idempotency key: {idempotency_key}")
        stored = row[0]
        if isinstance(stored, (bytearray, memoryview)): return bytes(stored)
        if isinstance(stored, bytes): return stored
        raise ReplayNotFoundError(f"captured response is not bytes for idempotency key: {idempotency_key}")