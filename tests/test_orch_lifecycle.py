from __future__ import annotations

import hashlib
import math
import sqlite3
import tempfile
import time
import unittest

from nullcore.orch.capture import (
    CallCapture, CaptureConflictError, NullReplayReader, ReplayNotFoundError,
)
from nullcore.orch.idempotency import IdempotencyKeyError, make_idempotency_key
from nullcore.orch.lease import LeaseError, LeaseManager
from nullcore.orch.state_machine import (
    LEGAL_TRANSITIONS, IllegalStateTransition, OrchState, can_transition, transition,
)


class FakeClock:
    def __init__(self, now_ms: int = 0) -> None:
        self._now_ms = now_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance_ms(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class TestOrchLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = f"{self._temp_dir.name}/orch_lifecycle.sqlite3"
        self._connections: list[sqlite3.Connection] = []
        self._clock = FakeClock()

    def tearDown(self) -> None:
        for conn in reversed(self._connections):
            conn.close()
        self._connections.clear()
        self._temp_dir.cleanup()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        self._connections.append(conn)
        return conn

    def test_legal_lifecycle_transitions(self) -> None:
        legal_pairs = [
            (OrchState.DRAFTING, OrchState.PROVISIONAL),
            (OrchState.PROVISIONAL, OrchState.GATING_ASYNC),
            (OrchState.GATING_ASYNC, OrchState.VERIFIED),
            (OrchState.GATING_ASYNC, OrchState.RECALLED),
            (OrchState.GATING_ASYNC, OrchState.UPGRADE_FAILED),
            (OrchState.UPGRADE_FAILED, OrchState.RECALLED),
        ]
        for current, desired in legal_pairs:
            with self.subTest(current=current, desired=desired):
                self.assertTrue(can_transition(current, desired))
                self.assertIs(transition(current, desired), desired)

    def test_illegal_lifecycle_transitions(self) -> None:
        illegal_pairs = [
            (OrchState.DRAFTING, OrchState.VERIFIED),
            (OrchState.PROVISIONAL, OrchState.VERIFIED),
            (OrchState.GATING_ASYNC, OrchState.DRAFTING),
            (OrchState.VERIFIED, OrchState.RECALLED),
            (OrchState.RECALLED, OrchState.PROVISIONAL),
            (OrchState.UPGRADE_FAILED, OrchState.VERIFIED),
        ]
        for current, desired in illegal_pairs:
            with self.subTest(current=current, desired=desired):
                self.assertFalse(can_transition(current, desired))
                with self.assertRaises(IllegalStateTransition) as raised:
                    transition(current, desired)
                message = str(raised.exception)
                self.assertIn(current.value, message)
                self.assertIn(desired.value, message)

    def test_self_transitions_are_illegal(self) -> None:
        for state in OrchState:
            with self.subTest(state=state):
                self.assertFalse(can_transition(state, state))
                with self.assertRaises(IllegalStateTransition):
                    transition(state, state)

    def test_state_space_is_exact(self) -> None:
        expected_values = {"DRAFTING", "PROVISIONAL", "GATING_ASYNC", "VERIFIED", "RECALLED", "UPGRADE_FAILED"}
        self.assertEqual({state.value for state in OrchState}, expected_values)
        for state in OrchState:
            self.assertIsInstance(state, str)

    def test_legal_transitions_mapping_is_immutable_and_total(self) -> None:
        self.assertEqual(set(LEGAL_TRANSITIONS.keys()), {state for state in OrchState})
        for targets in LEGAL_TRANSITIONS.values():
            self.assertIsInstance(targets, frozenset)
        with self.assertRaises(TypeError):
            LEGAL_TRANSITIONS[OrchState.DRAFTING] = frozenset({OrchState.VERIFIED})

    def test_lease_singularity(self) -> None:
        manager = LeaseManager(self._new_connection(), self._clock)
        self.assertTrue(manager.acquire("upgrade-token-1", "owner-A", 1_000))
        self.assertFalse(manager.acquire("upgrade-token-1", "owner-B", 1_000))
        self.assertFalse(manager.acquire("upgrade-token-1", "owner-A", 1_000))
        self.assertFalse(manager.release("upgrade-token-1", "owner-B"))
        self.assertTrue(manager.release("upgrade-token-1", "owner-A"))
        self.assertTrue(manager.acquire("upgrade-token-1", "owner-B", 1_000))

    def test_double_release_fails(self) -> None:
        manager = LeaseManager(self._new_connection(), self._clock)
        self.assertTrue(manager.acquire("upgrade-token-4", "owner-A", 1_000))
        self.assertTrue(manager.release("upgrade-token-4", "owner-A"))
        self.assertFalse(manager.release("upgrade-token-4", "owner-A"))

    def test_lease_acquire_rejects_invalid_inputs(self) -> None:
        manager = LeaseManager(self._new_connection(), self._clock)
        for token, owner_id, ttl_ms in [("", "owner-A", 100), ("token", "", 100), ("token", "owner-A", 0), ("token", "owner-A", -1)]:
            with self.subTest(token=token, owner_id=owner_id, ttl_ms=ttl_ms):
                with self.assertRaises(LeaseError):
                    manager.acquire(token, owner_id, ttl_ms)
        with self.assertRaises(LeaseError):
            manager.acquire("token", "owner-A", True)
        with self.assertRaises(LeaseError):
            manager.acquire("token", "owner-A", 100.0)
        with self.assertRaises(LeaseError):
            manager.acquire(None, "owner-A", 100)
        with self.assertRaises(LeaseError):
            manager.release("", "owner-A")
        with self.assertRaises(LeaseError):
            manager.release("token", "")

    def test_lease_enforced_across_connections(self) -> None:
        manager_a = LeaseManager(self._new_connection(), self._clock)
        manager_b = LeaseManager(self._new_connection(), self._clock)
        self.assertTrue(manager_a.acquire("shared-token", "owner-A", 500))
        self.assertFalse(manager_b.acquire("shared-token", "owner-B", 500))
        self._clock.advance_ms(500)
        self.assertTrue(manager_b.acquire("shared-token", "owner-B", 500))
        self.assertTrue(manager_b.release("shared-token", "owner-B"))

    def test_lease_database_uses_wal_mode(self) -> None:
        conn = self._new_connection()
        LeaseManager(conn, self._clock)
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(journal_mode).lower(), "wal")

    def test_lease_database_sets_busy_timeout(self) -> None:
        conn = self._new_connection()
        LeaseManager(conn, self._clock)
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(busy_timeout, 5000)

    def test_lease_expiry_allows_takeover(self) -> None:
        manager = LeaseManager(self._new_connection(), self._clock)
        self.assertTrue(manager.acquire("upgrade-token-2", "owner-A", 100))
        self._clock.advance_ms(100)
        self.assertTrue(manager.acquire("upgrade-token-2", "owner-B", 100))

    def test_release_expired_lease_fails(self) -> None:
        manager = LeaseManager(self._new_connection(), self._clock)
        self.assertFalse(manager.release("never-acquired", "owner-A"))
        self.assertTrue(manager.acquire("upgrade-token-3", "owner-A", 50))
        self._clock.advance_ms(50)
        self.assertFalse(manager.release("upgrade-token-3", "owner-A"))
        self.assertFalse(manager.release("upgrade-token-3", "owner-B"))

    def test_kill_mid_gating_recovery(self) -> None:
        manager = LeaseManager(self._new_connection(), self._clock)
        state = transition(OrchState.DRAFTING, OrchState.PROVISIONAL)
        state = transition(state, OrchState.GATING_ASYNC)
        self.assertIs(state, OrchState.GATING_ASYNC)
        token = "upgrade-token-gating"
        self.assertTrue(manager.acquire(token, "owner-A", 250))
        self.assertFalse(manager.acquire(token, "owner-B", 250))
        self._clock.advance_ms(1_000)
        self.assertTrue(manager.acquire(token, "owner-B", 250))
        state = transition(state, OrchState.RECALLED)
        self.assertIs(state, OrchState.RECALLED)

    def test_idempotency_key_determinism(self) -> None:
        first = make_idempotency_key("task-1", "gating", "seat-1", 0)
        self.assertEqual(first, make_idempotency_key("task-1", "gating", "seat-1", 0))
        self.assertNotEqual(first, make_idempotency_key("task-1", "gating", "seat-1", 1))
        self.assertNotEqual(first, make_idempotency_key("task-1", "gating", "seat-2", 0))
        self.assertNotEqual(first, make_idempotency_key("task-1", "upgrade", "seat-1", 0))
        self.assertNotEqual(first, make_idempotency_key("task-2", "gating", "seat-1", 0))
        self.assertEqual(len(first), 64)
        expected = hashlib.sha256("task-1\x1fgating\x1fseat-1\x1f0".encode("utf-8")).hexdigest()
        self.assertEqual(first, expected)

    def test_idempotency_key_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key("", "gating", "seat-1", 0)
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key("task-1", "", "seat-1", 0)
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key("task-1", "gating", "", 0)
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key("task-1", "gating", "seat-1", -1)
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key("task-1", "gating", "seat-1", True)
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key("task-1", "gating", "seat-1", 1.5)
        with self.assertRaises(IdempotencyKeyError):
            make_idempotency_key(None, "gating", "seat-1", 0)

    def test_capture_and_replay_roundtrip(self) -> None:
        capture = CallCapture(self._new_connection())
        key = make_idempotency_key("task-1", "gating", "seat-1", 0)
        request_bytes = b"\x00\x01request"
        response_bytes = b"\xff\x00response"
        capture.store(key, "task-1", "gating", "seat-1", 0, request_bytes, response_bytes, self._clock)
        reader_conn = self._new_connection()
        reader = NullReplayReader(reader_conn)
        replayed = reader.replay(key)
        self.assertIsInstance(replayed, bytes)
        self.assertEqual(replayed, response_bytes)
        self.assertEqual(reader.replay(key), response_bytes)
        changes_before_replay = reader_conn.total_changes
        self.assertEqual(reader.replay(key), response_bytes)
        self.assertEqual(reader_conn.total_changes, changes_before_replay)
        self._clock.advance_ms(7_000)
        capture.store(key, "task-1", "gating", "seat-1", 0, request_bytes, response_bytes, self._clock)
        self.assertEqual(reader.replay(key), response_bytes)
        other_key = make_idempotency_key("task-1", "gating", "seat-1", 1)
        capture.store(other_key, "task-1", "gating", "seat-1", 1, request_bytes, response_bytes, self._clock)
        self.assertEqual(reader.replay(other_key), response_bytes)
        conflicts = [
            ("task-2", "gating", "seat-1", 0, request_bytes, response_bytes),
            ("task-1", "upgrade", "seat-1", 0, request_bytes, response_bytes),
            ("task-1", "gating", "seat-2", 0, request_bytes, response_bytes),
            ("task-1", "gating", "seat-1", 1, request_bytes, response_bytes),
            ("task-1", "gating", "seat-1", 0, b"\x00\x01different", response_bytes),
            ("task-1", "gating", "seat-1", 0, request_bytes, b"\xff\x00different"),
        ]
        for task_id, stage, seat, attempt, req, resp in conflicts:
            with self.subTest(task_id=task_id, stage=stage, seat=seat, attempt=attempt):
                with self.assertRaises(CaptureConflictError):
                    capture.store(key, task_id, stage, seat, attempt, req, resp, self._clock)
        self.assertEqual(reader.replay(key), response_bytes)
        with self.assertRaises(ReplayNotFoundError):
            reader.replay("missing-idempotency-key")

    def test_capture_store_rejects_invalid_inputs(self) -> None:
        capture = CallCapture(self._new_connection())
        with self.assertRaises(CaptureConflictError):
            capture.store("", "task-1", "gating", "seat-1", 0, b"req", b"res", self._clock)
        with self.assertRaises(CaptureConflictError):
            capture.store("key", "", "gating", "seat-1", 0, b"req", b"res", self._clock)
        with self.assertRaises(CaptureConflictError):
            capture.store("key", "task-1", "", "seat-1", 0, b"req", b"res", self._clock)
        with self.assertRaises(CaptureConflictError):
            capture.store("key", "task-1", "gating", "", 0, b"req", b"res", self._clock)
        with self.assertRaises(CaptureConflictError):
            capture.store("key", "task-1", "gating", "seat-1", True, b"req", b"res", self._clock)
        with self.assertRaises(CaptureConflictError):
            capture.store("key", "task-1", "gating", "seat-1", 1, "req", b"res", self._clock)
        with self.assertRaises(CaptureConflictError):
            capture.store("key", "task-1", "gating", "seat-1", 1, b"req", bytearray(b"res"), self._clock)

    def test_capture_store_rejects_negative_attempt(self) -> None:
        capture = CallCapture(self._new_connection())
        with self.assertRaises(CaptureConflictError):
            capture.store("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "task-1", "gating", "seat-1", -1, b"\x00\x01request", b"\xff\x00response", self._clock)

    def test_capture_store_rejects_mismatched_idempotency_key(self) -> None:
        capture = CallCapture(self._new_connection())
        wrong_key = make_idempotency_key("task-1", "gating", "seat-1", 1)
        with self.assertRaises(CaptureConflictError):
            capture.store(wrong_key, "task-1", "gating", "seat-1", 0, b"\x00\x01request", b"\xff\x00response", self._clock)

    def test_capture_store_accepts_matching_idempotency_key(self) -> None:
        capture = CallCapture(self._new_connection())
        key = make_idempotency_key("task-1", "gating", "seat-1", 0)
        response_bytes = b"\xff\x00response"
        capture.store(key, "task-1", "gating", "seat-1", 0, b"\x00\x01request", response_bytes, self._clock)
        reader = NullReplayReader(self._new_connection())
        self.assertEqual(reader.replay(key), response_bytes)

    def test_capture_database_uses_wal_mode(self) -> None:
        conn = self._new_connection()
        CallCapture(conn)
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(journal_mode).lower(), "wal")

    def test_capture_database_sets_busy_timeout(self) -> None:
        conn = self._new_connection()
        CallCapture(conn)
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertEqual(busy_timeout, 5000)

    def test_replay_on_uninitialized_database_raises_not_found(self) -> None:
        reader = NullReplayReader(self._new_connection())
        with self.assertRaises(ReplayNotFoundError):
            reader.replay("any-key")

    def test_s4a_state_transition_p95_latency(self) -> None:
        current = OrchState.DRAFTING
        desired = OrchState.PROVISIONAL
        for _ in range(1_000):
            transition(current, desired)
        samples: list[int] = []
        perf_counter_ns = time.perf_counter_ns
        for _ in range(10_000):
            started_ns = perf_counter_ns()
            transition(current, desired)
            finished_ns = perf_counter_ns()
            samples.append(finished_ns - started_ns)
        samples.sort()
        p95_rank = math.ceil(0.95 * len(samples))
        p95_ns = samples[p95_rank - 1]
        self.assertLess(p95_ns, 2_000_000, f"p95 transition latency {p95_ns} ns is not below 2,000,000 ns")


if __name__ == "__main__":
    unittest.main()