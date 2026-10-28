from __future__ import annotations

import sqlite3
import tempfile
import unittest

from nullcore.orch.cache_key import CacheKeyError, make_cache_key
from nullcore.orch.cache_matrix import (
    WRITE_MATRIX, CacheStatus, CacheTier, MatrixViolationError, can_write, verify_credential,
)
from nullcore.orch.cache_store import (
    CacheConflictError, CacheEvictedError, CacheMissError, CachePrimacyError, CacheStore,
)
from nullcore.orch.state_machine import OrchState


class FakeClock:
    def __init__(self, now_ms: int = 0) -> None:
        self._now_ms = now_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance_ms(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class TestCacheMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = f"{self._temp_dir.name}/cache_matrix.sqlite3"
        self._connections: list[sqlite3.Connection] = []
        self._clock = FakeClock()
        self._conn = self._new_connection()
        self._store = CacheStore(self._conn)

    def tearDown(self) -> None:
        for conn in reversed(self._connections):
            conn.close()
        self._connections.clear()
        self._temp_dir.cleanup()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        self._connections.append(conn)
        return conn

    def _write_entry(self, config_hash, task_id, stage, seat, attempt, tier, status, degraded_flag, payload_bytes):
        key = make_cache_key(config_hash, task_id, stage, seat, attempt)
        self._store.write(key, config_hash, task_id, stage, seat, attempt, tier, status, degraded_flag, payload_bytes, self._clock)
        return key

    def test_write_matrix_is_exact_and_immutable(self) -> None:
        self.assertEqual(len(WRITE_MATRIX), 3)
        self.assertEqual(dict(WRITE_MATRIX), {
            CacheStatus.TERMINAL_VERIFIED: frozenset({CacheTier.T0, CacheTier.T1, CacheTier.T2, CacheTier.T3}),
            CacheStatus.PROVISIONAL: frozenset({CacheTier.T2}),
            CacheStatus.DEGRADED: frozenset({CacheTier.T0}),
        })
        with self.assertRaises(TypeError):
            WRITE_MATRIX[CacheStatus.DEGRADED] = frozenset({CacheTier.T1})

    def test_can_write_exhaustive_table(self) -> None:
        expected = {
            CacheStatus.TERMINAL_VERIFIED: {CacheTier.T0: True, CacheTier.T1: True, CacheTier.T2: True, CacheTier.T3: True},
            CacheStatus.PROVISIONAL: {CacheTier.T0: False, CacheTier.T1: False, CacheTier.T2: True, CacheTier.T3: False},
            CacheStatus.DEGRADED: {CacheTier.T0: True, CacheTier.T1: False, CacheTier.T2: False, CacheTier.T3: False},
        }
        for status in CacheStatus:
            for tier in CacheTier:
                with self.subTest(status=status, tier=tier):
                    self.assertIs(can_write(status, tier), expected[status][tier])
        self.assertFalse(can_write("DEGRADED", CacheTier.T0))
        self.assertFalse(can_write(CacheStatus.DEGRADED, "T0"))

    def test_cache_key_config_drift_sensitivity(self) -> None:
        base_key = make_cache_key("cfg-hash-a", "task-1", "S3", "seat-alpha", 2)
        self.assertEqual(base_key, make_cache_key("cfg-hash-a", "task-1", "S3", "seat-alpha", 2))
        drifted_key = make_cache_key("cfg-hash-b", "task-1", "S3", "seat-alpha", 2)
        self.assertNotEqual(base_key, drifted_key)
        self.assertEqual(len(base_key), 64)
        self.assertEqual(len(drifted_key), 64)
        self.assertNotEqual(base_key, make_cache_key("cfg-hash-a", "task-2", "S3", "seat-alpha", 2))
        self.assertNotEqual(base_key, make_cache_key("cfg-hash-a", "task-1", "S4", "seat-alpha", 2))
        self.assertNotEqual(base_key, make_cache_key("cfg-hash-a", "task-1", "S3", "seat-beta", 2))
        self.assertNotEqual(base_key, make_cache_key("cfg-hash-a", "task-1", "S3", "seat-alpha", 3))
        with self.assertRaises(CacheKeyError):
            make_cache_key("", "task-1", "S3", "seat-alpha", 2)
        with self.assertRaises(CacheKeyError):
            make_cache_key("cfg-hash-a", "task-1", "S3", "seat-alpha", -1)
        with self.assertRaises(CacheKeyError):
            make_cache_key("cfg-hash-a", "task-1", "S3", "seat-alpha", True)

    def test_verify_credential_terminal_verified_requires_stage_vector(self) -> None:
        full_vector = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}
        self.assertIs(verify_credential(OrchState.VERIFIED, full_vector, CacheStatus.TERMINAL_VERIFIED), CacheStatus.TERMINAL_VERIFIED)
        with self.assertRaises(MatrixViolationError):
            verify_credential(OrchState.VERIFIED, {"S5": "PASSED", "S6": "PASSED"}, CacheStatus.TERMINAL_VERIFIED)
        with self.assertRaises(MatrixViolationError):
            verify_credential(OrchState.VERIFIED, {"S5": "PASSED", "S6": "FAILED", "S7": "PASSED"}, CacheStatus.TERMINAL_VERIFIED)
        with self.assertRaises(MatrixViolationError):
            verify_credential(OrchState.VERIFIED, {"S5": "SKIPPED", "S6": "PASSED", "S7": "PASSED"}, CacheStatus.TERMINAL_VERIFIED)
        with self.assertRaises(MatrixViolationError):
            verify_credential(OrchState.VERIFIED, None, CacheStatus.TERMINAL_VERIFIED)

    def test_verify_credential_lifecycle_consistency(self) -> None:
        full_vector = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}
        with self.assertRaises(MatrixViolationError):
            verify_credential(OrchState.PROVISIONAL, full_vector, CacheStatus.TERMINAL_VERIFIED)
        with self.assertRaises(MatrixViolationError):
            verify_credential(OrchState.GATING_ASYNC, full_vector, CacheStatus.TERMINAL_VERIFIED)
        self.assertIs(verify_credential(OrchState.PROVISIONAL, full_vector, CacheStatus.PROVISIONAL), CacheStatus.PROVISIONAL)
        self.assertIs(verify_credential(OrchState.GATING_ASYNC, full_vector, CacheStatus.PROVISIONAL), CacheStatus.PROVISIONAL)
        self.assertIs(verify_credential(OrchState.VERIFIED, full_vector, CacheStatus.PROVISIONAL), CacheStatus.PROVISIONAL)
        for claimed in CacheStatus:
            with self.subTest(claimed=claimed):
                with self.assertRaises(MatrixViolationError):
                    verify_credential(OrchState.RECALLED, full_vector, claimed)
                with self.assertRaises(MatrixViolationError):
                    verify_credential(OrchState.UPGRADE_FAILED, full_vector, claimed)
        self.assertIs(verify_credential(OrchState.DRAFTING, full_vector, CacheStatus.DEGRADED), CacheStatus.DEGRADED)

    def test_verify_credential_rejects_non_enum_lifecycle(self) -> None:
        full_vector = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}
        cases = (
            ("PROVISIONAL", CacheStatus.PROVISIONAL, "str"),
            ("VERIFIED", CacheStatus.TERMINAL_VERIFIED, "str"),
            ("RECALLED", CacheStatus.DEGRADED, "str"),
            (None, CacheStatus.DEGRADED, "NoneType"),
            (0, CacheStatus.DEGRADED, "int"),
        )
        for bad_lifecycle, claimed, type_name in cases:
            with self.subTest(lifecycle_state=bad_lifecycle, claimed=claimed):
                with self.assertRaises(MatrixViolationError) as caught:
                    verify_credential(bad_lifecycle, full_vector, claimed)
                self.assertIn(type_name, str(caught.exception))

    def test_store_write_policy_enforcement(self) -> None:
        for attempt, tier in enumerate((CacheTier.T0, CacheTier.T1, CacheTier.T2, CacheTier.T3)):
            with self.subTest(tier=tier):
                self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-terminal", attempt, tier, CacheStatus.TERMINAL_VERIFIED, False, b"terminal-payload")
        self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-provisional", 0, CacheTier.T2, CacheStatus.PROVISIONAL, False, b"provisional-payload")
        for attempt, tier in enumerate((CacheTier.T0, CacheTier.T1, CacheTier.T3)):
            with self.subTest(tier=tier):
                with self.assertRaises(MatrixViolationError):
                    self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-provisional-bad", attempt, tier, CacheStatus.PROVISIONAL, False, b"provisional-payload")
        self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-degraded", 0, CacheTier.T0, CacheStatus.DEGRADED, True, b"degraded-payload")
        for attempt, tier in enumerate((CacheTier.T1, CacheTier.T2, CacheTier.T3)):
            with self.subTest(tier=tier):
                with self.assertRaises(MatrixViolationError):
                    self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-degraded-bad", attempt, tier, CacheStatus.DEGRADED, True, b"degraded-payload")
        with self.assertRaises(MatrixViolationError):
            self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-degraded-bad", 3, CacheTier.T0, CacheStatus.DEGRADED, False, b"degraded-payload")
        with self.assertRaises(MatrixViolationError):
            self._write_entry("cfg-matrix", "task-matrix", "S9", "seat-terminal-flagged", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, True, b"terminal-payload")

    def test_store_key_binding_rejects_mismatched_key(self) -> None:
        correct_key = make_cache_key("cfg-bind-a", "task-bind", "S2", "seat-bind", 0)
        foreign_key = make_cache_key("cfg-bind-b", "task-bind", "S2", "seat-bind", 0)
        with self.assertRaises(CacheConflictError):
            self._store.write(foreign_key, "cfg-bind-a", "task-bind", "S2", "seat-bind", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"bind-payload", self._clock)
        with self.assertRaises(CacheMissError):
            self._store.read_primary(correct_key)
        with self.assertRaises(CacheMissError):
            self._store.read_primary(foreign_key)

    def test_store_idempotent_rewrite_and_conflict(self) -> None:
        key = self._write_entry("cfg-idem", "task-idem", "S1", "seat-idem", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"idem-payload")
        self._write_entry("cfg-idem", "task-idem", "S1", "seat-idem", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"idem-payload")
        with self.assertRaises(CacheConflictError):
            self._write_entry("cfg-idem", "task-idem", "S1", "seat-idem", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"divergent-payload")
        self.assertEqual(self._store.read_primary(key), b"idem-payload")

    def test_primacy_read_separation(self) -> None:
        clean_t0_key = self._write_entry("cfg-primacy", "task-primacy", "S1", "seat-clean-t0", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"clean-t0-payload")
        self.assertEqual(self._store.read_primary(clean_t0_key), b"clean-t0-payload")
        with self.assertRaises(CachePrimacyError):
            self._store.read_invention_diff(clean_t0_key)
        flagged_t0_key = self._write_entry("cfg-primacy", "task-primacy", "S1", "seat-flagged-t0", 0, CacheTier.T0, CacheStatus.DEGRADED, True, b"flagged-t0-payload")
        self.assertEqual(self._store.read_invention_diff(flagged_t0_key), b"flagged-t0-payload")
        with self.assertRaises(CachePrimacyError):
            self._store.read_primary(flagged_t0_key)
        clean_t2_key = self._write_entry("cfg-primacy", "task-primacy", "S1", "seat-clean-t2", 0, CacheTier.T2, CacheStatus.PROVISIONAL, False, b"clean-t2-payload")
        with self.assertRaises(CachePrimacyError):
            self._store.read_invention_diff(clean_t2_key)
        with self.assertRaises(CacheMissError):
            self._store.read_primary("missing-cache-key")
        with self.assertRaises(CacheMissError):
            self._store.read_invention_diff("missing-cache-key")

    def test_eviction_tombstone_and_retention(self) -> None:
        key = self._write_entry("cfg-evict", "task-evict", "S1", "seat-evict", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"evict-payload")
        self._clock.advance_ms(5_000)
        self.assertTrue(self._store.evict(key, self._clock))
        self.assertFalse(self._store.evict(key, self._clock))
        self.assertFalse(self._store.evict("missing-evict-key", self._clock))
        with self.assertRaises(CacheEvictedError):
            self._store.read_primary(key)
        with self.assertRaises(CacheEvictedError):
            self._store.read_invention_diff(key)
        with self.assertRaises(CacheEvictedError):
            self._store.write(key, "cfg-evict", "task-evict", "S1", "seat-evict", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"evict-payload", self._clock)
        retained_rows = self._conn.execute("SELECT COUNT(*) FROM cache_entries WHERE cache_key = ? AND evicted_at_ms IS NOT NULL", (key,)).fetchone()
        self.assertEqual(retained_rows[0], 1)
        total_rows = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        self.assertEqual(total_rows[0], 1)

    def test_store_pragmas(self) -> None:
        busy_timeout = self._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(busy_timeout, 5000)
        journal_mode = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(journal_mode).lower(), "wal")

    def test_reads_never_mutate(self) -> None:
        clean_key = self._write_entry("cfg-read", "task-read", "S1", "seat-clean", 0, CacheTier.T0, CacheStatus.TERMINAL_VERIFIED, False, b"read-payload")
        flagged_key = self._write_entry("cfg-read", "task-read", "S1", "seat-flagged", 0, CacheTier.T0, CacheStatus.DEGRADED, True, b"read-flagged-payload")
        changes_before_reads = self._conn.total_changes
        self.assertEqual(self._store.read_primary(clean_key), b"read-payload")
        self.assertEqual(self._store.read_invention_diff(flagged_key), b"read-flagged-payload")
        self.assertEqual(self._conn.total_changes, changes_before_reads)


if __name__ == "__main__":
    unittest.main()