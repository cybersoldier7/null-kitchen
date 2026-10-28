from __future__ import annotations
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final
from nullcore.orch.cache_key import CacheKeyError, make_cache_key
from nullcore.orch.cache_matrix import CacheStatus, CacheTier, MatrixViolationError, can_write

__all__ = ["CacheMissError", "CacheEvictedError", "CachePrimacyError", "CacheConflictError", "CacheStore"]
class CacheMissError(Exception): pass
class CacheEvictedError(Exception): pass
class CachePrimacyError(Exception): pass
class CacheConflictError(Exception): pass

_CACHE_SCHEMA: Final[str] = """CREATE TABLE IF NOT EXISTS cache_entries (cache_key TEXT PRIMARY KEY, config_hash TEXT NOT NULL, task_id TEXT NOT NULL, stage TEXT NOT NULL, seat TEXT NOT NULL, attempt INTEGER NOT NULL, tier TEXT NOT NULL, status TEXT NOT NULL, degraded_flag INTEGER NOT NULL, payload_bytes BLOB NOT NULL, injected_at_ms INTEGER NOT NULL, evicted_at_ms INTEGER);"""

def _require_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value: raise CacheConflictError(f"{name} must be a non-empty str")
def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise CacheConflictError(f"{name} must be an int >= 0")
def _require_bytes(value: object, name: str) -> None:
    if not isinstance(value, bytes): raise CacheConflictError(f"{name} must be bytes")
def _as_bytes_like(value: object) -> object:
    if isinstance(value, (bytearray, memoryview)): return bytes(value)
    return value

class CacheStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn; self._conn.isolation_level = None
        self._conn.execute("PRAGMA busy_timeout=5000;"); self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;"); self._conn.execute(_CACHE_SCHEMA)

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        conn = self._conn; conn.execute("BEGIN IMMEDIATE")
        try: yield
        except BaseException:
            try: conn.execute("ROLLBACK")
            except sqlite3.Error: pass
            raise
        conn.execute("COMMIT")

    def write(self, cache_key: str, config_hash: str, task_id: str, stage: str, seat: str, attempt: int, tier: CacheTier, status: CacheStatus, degraded_flag: bool, payload_bytes: bytes, clock_ms: Callable[[], int]) -> None:
        _require_non_empty_str(cache_key, "cache_key"); _require_non_empty_str(config_hash, "config_hash")
        _require_non_empty_str(task_id, "task_id"); _require_non_empty_str(stage, "stage"); _require_non_empty_str(seat, "seat")
        _require_non_negative_int(attempt, "attempt")
        if not isinstance(tier, CacheTier): raise CacheConflictError("tier must be CacheTier")
        if not isinstance(status, CacheStatus): raise CacheConflictError("status must be CacheStatus")
        if not isinstance(degraded_flag, bool): raise CacheConflictError("degraded_flag must be bool")
        _require_bytes(payload_bytes, "payload_bytes")
        if not callable(clock_ms): raise CacheConflictError("clock_ms must be callable")
        try: expected_key = make_cache_key(config_hash, task_id, stage, seat, attempt)
        except CacheKeyError as error: raise CacheConflictError(f"invalid identity: {error}") from error
        if expected_key != cache_key: raise CacheConflictError("cache_key does not bind")
        if not can_write(status, tier): raise MatrixViolationError("WRITE_MATRIX forbids")
        if degraded_flag != (status is CacheStatus.DEGRADED): raise MatrixViolationError("degraded_flag mismatch")
        injected_at_ms = clock_ms()
        if isinstance(injected_at_ms, bool) or not isinstance(injected_at_ms, int): raise CacheConflictError("clock_ms must return int")
        with self._immediate_transaction():
            row = self._conn.execute("SELECT config_hash, task_id, stage, seat, attempt, tier, status, degraded_flag, payload_bytes, injected_at_ms, evicted_at_ms FROM cache_entries WHERE cache_key = ?", (cache_key,)).fetchone()
            if row is None:
                self._conn.execute("INSERT INTO cache_entries (cache_key, config_hash, task_id, stage, seat, attempt, tier, status, degraded_flag, payload_bytes, injected_at_ms, evicted_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)", (cache_key, config_hash, task_id, stage, seat, attempt, tier.value, status.value, 1 if degraded_flag else 0, payload_bytes, injected_at_ms))
                return
            if row[10] is not None: raise CacheEvictedError("evicted tombstone")
            stored = (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], _as_bytes_like(row[8]), row[9])
            incoming = (config_hash, task_id, stage, seat, attempt, tier.value, status.value, 1 if degraded_flag else 0, payload_bytes, injected_at_ms)
            if stored != incoming: raise CacheConflictError("conflicting entry")

    def read_primary(self, cache_key: str) -> bytes:
        if not isinstance(cache_key, str) or not cache_key: raise CacheMissError("invalid key")
        row = self._conn.execute("SELECT evicted_at_ms, degraded_flag, payload_bytes FROM cache_entries WHERE cache_key = ?", (cache_key,)).fetchone()
        if row is None: raise CacheMissError("no entry")
        if row[0] is not None: raise CacheEvictedError("evicted tombstone")
        if row[1] == 1: raise CachePrimacyError("flagged entry not primacy-servable")
        payload_bytes = row[2]
        if isinstance(payload_bytes, (bytearray, memoryview)): return bytes(payload_bytes)
        return payload_bytes

    def read_invention_diff(self, cache_key: str) -> bytes:
        if not isinstance(cache_key, str) or not cache_key: raise CacheMissError("invalid key")
        row = self._conn.execute("SELECT evicted_at_ms, degraded_flag, tier, payload_bytes FROM cache_entries WHERE cache_key = ?", (cache_key,)).fetchone()
        if row is None: raise CacheMissError("no entry")
        if row[0] is not None: raise CacheEvictedError("evicted tombstone")
        if row[1] != 1: raise CachePrimacyError("clean entry must use read_primary")
        if row[2] != CacheTier.T0.value: raise CachePrimacyError("invention-diff restricted to T0")
        payload_bytes = row[3]
        if isinstance(payload_bytes, (bytearray, memoryview)): return bytes(payload_bytes)
        return payload_bytes

    def evict(self, cache_key: str, clock_ms: Callable[[], int]) -> bool:
        if not callable(clock_ms): raise CacheConflictError("clock_ms must be callable")
        evicted_at_ms = clock_ms()
        if isinstance(evicted_at_ms, bool) or not isinstance(evicted_at_ms, int): raise CacheConflictError("clock_ms must return int")
        with self._immediate_transaction():
            row = self._conn.execute("SELECT evicted_at_ms FROM cache_entries WHERE cache_key = ?", (cache_key,)).fetchone()
            if row is None or row[0] is not None: return False
            self._conn.execute("UPDATE cache_entries SET evicted_at_ms = ? WHERE cache_key = ?", (evicted_at_ms, cache_key))
        return True