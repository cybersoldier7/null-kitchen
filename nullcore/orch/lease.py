from __future__ import annotations
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

__all__ = ["LeaseError", "LeaseManager"]
class LeaseError(Exception): pass

_LEASE_SCHEMA: Final[str] = """CREATE TABLE IF NOT EXISTS leases (upgrade_token TEXT PRIMARY KEY, owner_id TEXT NOT NULL, expires_at_ms INTEGER NOT NULL);"""

def _require_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value: raise LeaseError(f"{name} must be a non-empty str")
def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0: raise LeaseError(f"{name} must be a positive int")

class LeaseManager:
    def __init__(self, conn: sqlite3.Connection, clock_ms: Callable[[], int]) -> None:
        if not callable(clock_ms): raise LeaseError("clock_ms must be callable")
        self._conn = conn; self._clock_ms = clock_ms
        self._conn.isolation_level = None
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_LEASE_SCHEMA)

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        conn = self._conn; conn.execute("BEGIN IMMEDIATE")
        try: yield
        except BaseException:
            try: conn.execute("ROLLBACK")
            except sqlite3.Error: pass
            raise
        conn.execute("COMMIT")

    def _now_ms(self) -> int:
        now_ms = self._clock_ms()
        if isinstance(now_ms, bool) or not isinstance(now_ms, int): raise LeaseError("clock_ms must return an int")
        return now_ms

    def acquire(self, token: str, owner_id: str, ttl_ms: int) -> bool:
        _require_non_empty_str(token, "token"); _require_non_empty_str(owner_id, "owner_id"); _require_positive_int(ttl_ms, "ttl_ms")
        now_ms = self._now_ms(); expires_at_ms = now_ms + ttl_ms
        with self._immediate_transaction():
            row = self._conn.execute("SELECT expires_at_ms FROM leases WHERE upgrade_token = ?", (token,)).fetchone()
            if row is not None:
                if now_ms < row[0]: return False
                self._conn.execute("UPDATE leases SET owner_id = ?, expires_at_ms = ? WHERE upgrade_token = ?", (owner_id, expires_at_ms, token))
            else:
                self._conn.execute("INSERT INTO leases (upgrade_token, owner_id, expires_at_ms) VALUES (?, ?, ?)", (token, owner_id, expires_at_ms))
        return True

    def release(self, token: str, owner_id: str) -> bool:
        _require_non_empty_str(token, "token"); _require_non_empty_str(owner_id, "owner_id")
        now_ms = self._now_ms()
        with self._immediate_transaction():
            row = self._conn.execute("SELECT owner_id, expires_at_ms FROM leases WHERE upgrade_token = ?", (token,)).fetchone()
            if row is None: return False
            if row[0] != owner_id: return False
            if now_ms >= row[1]: return False
            self._conn.execute("DELETE FROM leases WHERE upgrade_token = ?", (token,))
        return True