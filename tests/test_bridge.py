"""Deterministic unit tests for the task envelope and the ORCH bridge.

Coverage: full envelope field validation with the field name in every
rejection message, envelope frozen-ness, ``with_attempt`` copying,
StageVector insertion order and immutability (including record input
validation), bridge constructor validation, submit argument type
validation, unknown-seat denial with an untouched vector, the S6
recording contract for the executed, replay, executor-failure,
credential-denial, and lease-contention paths, seats() ordering, and
registry immutability.  All timing comes from a fake monotonic clock;
every test owns a throwaway SQLite database whose connection is closed
in tearDown; no test sleeps or touches the network.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import tempfile
import unittest

from nullcore.orch.bridge import BridgeError, OrchBridge
from nullcore.orch.cache_matrix import (
    CacheStatus,
    CacheTier,
    MatrixViolationError,
)
from nullcore.orch.capture import CallCapture
from nullcore.orch.director import DirectorError, OrchDirector
from nullcore.orch.idempotency import make_idempotency_key
from nullcore.orch.lease import LeaseManager
from nullcore.orch.state_machine import OrchState
from nullcore.orch.task_envelope import (
    StageVector,
    TaskEnvelope,
    TaskEnvelopeError,
)

FULL_VECTOR = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}


class FakeClock:
    """Deterministic injectable monotonic millisecond clock."""

    def __init__(self, now_ms: int = 0) -> None:
        self._now_ms = now_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance_ms(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class RecordingExecutor:
    """Callable that counts calls and returns a fixed response."""

    def __init__(self, response: bytes) -> None:
        self._response = response
        self.calls = 0

    def __call__(self, request: bytes) -> bytes:
        self.calls += 1
        return self._response


def make_envelope(**overrides) -> TaskEnvelope:
    """Return a fully valid VERIFIED envelope with overrides applied."""
    fields = {
        "task_id": "task-bridge",
        "stage": "S1",
        "seat": "seat-alpha",
        "attempt": 0,
        "config_hash": "cfg-bridge",
        "request_bytes": b"request",
        "lifecycle_state": OrchState.VERIFIED,
        "stage_vector": dict(FULL_VECTOR),
        "tier": CacheTier.T0,
        "status": CacheStatus.TERMINAL_VERIFIED,
        "degraded_flag": False,
    }
    fields.update(overrides)
    return TaskEnvelope(**fields)


class TestTaskEnvelope(unittest.TestCase):
    """TaskEnvelope validation, frozen-ness, copying, and StageVector."""

    def test_envelope_rejects_each_malformed_field(self) -> None:
        cases = [
            ("empty task_id", {"task_id": ""}, "task_id"),
            ("non-str task_id", {"task_id": 7}, "task_id"),
            ("empty stage", {"stage": ""}, "stage"),
            ("empty seat", {"seat": ""}, "seat"),
            ("bool attempt", {"attempt": True}, "attempt"),
            ("negative attempt", {"attempt": -1}, "attempt"),
            ("non-int attempt", {"attempt": "0"}, "attempt"),
            ("empty config_hash", {"config_hash": ""}, "config_hash"),
            ("non-bytes request", {"request_bytes": "request"}, "request_bytes"),
            ("str lifecycle_state", {"lifecycle_state": "VERIFIED"}, "lifecycle_state"),
            ("None lifecycle_state", {"lifecycle_state": None}, "lifecycle_state"),
            ("str tier", {"tier": "T0"}, "tier"),
            ("str status", {"status": "TERMINAL_VERIFIED"}, "status"),
            ("str degraded_flag", {"degraded_flag": "yes"}, "degraded_flag"),
            ("int degraded_flag", {"degraded_flag": 1}, "degraded_flag"),
            ("non-mapping stage_vector", {"stage_vector": ["S5"]}, "stage_vector"),
            ("non-str stage_vector value", {"stage_vector": {"S5": 1}}, "stage_vector"),
            ("non-str stage_vector key", {"stage_vector": {5: "PASSED"}}, "stage_vector"),
        ]
        for label, overrides, field_name in cases:
            with self.subTest(case=label):
                with self.assertRaises(TaskEnvelopeError) as caught:
                    make_envelope(**overrides)
                self.assertIn(field_name, str(caught.exception))

    def test_envelope_is_frozen(self) -> None:
        envelope = make_envelope()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            envelope.attempt = 99

    def test_with_attempt_returns_validated_copy(self) -> None:
        envelope = make_envelope(attempt=0)
        copied = envelope.with_attempt(3)
        self.assertIsNot(copied, envelope)
        self.assertEqual(copied.attempt, 3)
        self.assertEqual(copied.task_id, envelope.task_id)
        self.assertEqual(copied.stage, envelope.stage)
        self.assertEqual(copied.seat, envelope.seat)
        self.assertEqual(copied.config_hash, envelope.config_hash)
        self.assertEqual(copied.request_bytes, envelope.request_bytes)
        self.assertEqual(copied.lifecycle_state, envelope.lifecycle_state)
        self.assertEqual(copied.stage_vector, envelope.stage_vector)
        self.assertEqual(copied.tier, envelope.tier)
        self.assertEqual(copied.status, envelope.status)
        self.assertEqual(copied.degraded_flag, envelope.degraded_flag)
        # The original envelope is untouched.
        self.assertEqual(envelope.attempt, 0)
        for bad_attempt in (True, -2, "3", None):
            with self.subTest(bad_attempt=bad_attempt):
                with self.assertRaises(TaskEnvelopeError):
                    envelope.with_attempt(bad_attempt)

    def test_stage_vector_snapshot_is_immutable_and_ordered(self) -> None:
        vector = StageVector()
        # Both views are safe when nothing has been recorded yet.
        self.assertEqual(len(vector.snapshot()), 0)
        self.assertEqual(len(vector.notes()), 0)
        vector.record("S5", True, "upstream")
        vector.record("S6", True, "executed")
        vector.record("S7", False, "pending")
        self.assertEqual(
            list(vector.snapshot().items()),
            [("S5", "PASSED"), ("S6", "PASSED"), ("S7", "FAILED")],
        )
        self.assertEqual(
            list(vector.notes().items()),
            [("S5", "upstream"), ("S6", "executed"), ("S7", "pending")],
        )
        with self.assertRaises(TypeError):
            vector.snapshot()["S8"] = "PASSED"
        with self.assertRaises(TypeError):
            vector.notes()["S6"] = "rewritten"
        # Snapshots are copies: later records never mutate them.
        frozen_snapshot = vector.snapshot()
        vector.record("S8", True, "late")
        self.assertNotIn("S8", frozen_snapshot)
        self.assertEqual(len(frozen_snapshot), 3)

    def test_stage_vector_record_validates_inputs(self) -> None:
        vector = StageVector()
        bad_cases = [
            ("empty stage_id", ("", True, "")),
            ("non-str stage_id", (None, True, "")),
            ("int passed", ("S5", 1, "")),
            ("str passed", ("S5", "yes", "")),
            ("non-str note", ("S5", True, 5)),
        ]
        for label, args in bad_cases:
            with self.subTest(case=label):
                with self.assertRaises(TaskEnvelopeError):
                    vector.record(*args)
        # No malformed record ever touched the vector.
        self.assertEqual(len(vector.snapshot()), 0)
        self.assertEqual(len(vector.notes()), 0)


class TestOrchBridge(unittest.TestCase):
    """OrchBridge validation, routing, and the S6 recording contract."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = f"{self._temp_dir.name}/bridge.sqlite3"
        self._clock = FakeClock()
        self._conn = sqlite3.connect(self._db_path)
        self._director = OrchDirector(self._conn, self._clock)
        self._executor = RecordingExecutor(b"bridge-response")
        self._bridge = OrchBridge(self._director, {"seat-alpha": self._executor})

    def tearDown(self) -> None:
        self._conn.close()
        self._temp_dir.cleanup()

    def test_bridge_constructor_validates_inputs(self) -> None:
        with self.assertRaises(BridgeError):
            OrchBridge("not-a-director", {"seat-alpha": self._executor})
        with self.assertRaises(BridgeError):
            OrchBridge(self._director, {"seat-alpha": "not-callable"})
        with self.assertRaises(BridgeError):
            OrchBridge(self._director, {"": self._executor})
        with self.assertRaises(BridgeError):
            OrchBridge(self._director, {5: self._executor})
        with self.assertRaises(BridgeError):
            OrchBridge(self._director, [("seat-alpha", self._executor)])

    def test_submit_validates_argument_types(self) -> None:
        vector = StageVector()
        with self.assertRaises(BridgeError):
            self._bridge.submit("not-an-envelope", vector)
        with self.assertRaises(BridgeError):
            self._bridge.submit(make_envelope(), "not-a-vector")
        self.assertEqual(len(vector.snapshot()), 0)

    def test_unknown_seat_raises_and_leaves_vector_empty(self) -> None:
        vector = StageVector()
        envelope = make_envelope(seat="seat-ghost")
        with self.assertRaises(BridgeError) as caught:
            self._bridge.submit(envelope, vector)
        self.assertIn("seat-ghost", str(caught.exception))
        self.assertEqual(len(vector.snapshot()), 0)
        self.assertEqual(len(vector.notes()), 0)
        self.assertEqual(self._executor.calls, 0)

    def test_happy_path_records_s6_passed_executed(self) -> None:
        vector = StageVector()
        response, from_replay = self._bridge.submit(make_envelope(), vector)
        self.assertEqual(response, b"bridge-response")
        self.assertFalse(from_replay)
        self.assertEqual(self._executor.calls, 1)
        self.assertEqual(vector.snapshot(), {"S6": "PASSED"})
        self.assertEqual(vector.notes(), {"S6": "executed"})

    def test_replay_path_records_s6_passed_replay(self) -> None:
        envelope = make_envelope()
        idem_key = make_idempotency_key(
            envelope.task_id, envelope.stage, envelope.seat, envelope.attempt
        )
        CallCapture(self._conn).store(
            idem_key,
            envelope.task_id,
            envelope.stage,
            envelope.seat,
            envelope.attempt,
            envelope.request_bytes,
            b"replayed-response",
            self._clock,
        )
        vector = StageVector()
        response, from_replay = self._bridge.submit(envelope, vector)
        self.assertEqual(response, b"replayed-response")
        self.assertTrue(from_replay)
        self.assertEqual(self._executor.calls, 0)
        self.assertEqual(vector.snapshot(), {"S6": "PASSED"})
        self.assertEqual(vector.notes(), {"S6": "replay"})

    def test_executor_exception_records_failure_and_propagates(self) -> None:
        def exploding_executor(request: bytes) -> bytes:
            raise RuntimeError("executor exploded")

        bridge = OrchBridge(self._director, {"seat-alpha": exploding_executor})
        vector = StageVector()
        with self.assertRaises(RuntimeError):
            bridge.submit(make_envelope(), vector)
        self.assertEqual(vector.snapshot(), {"S6": "FAILED"})
        self.assertEqual(vector.notes(), {"S6": "RuntimeError"})

    def test_credential_denial_records_failure_and_propagates(self) -> None:
        envelope = make_envelope(
            lifecycle_state=OrchState.PROVISIONAL,
            status=CacheStatus.TERMINAL_VERIFIED,
        )
        vector = StageVector()
        with self.assertRaises(MatrixViolationError):
            self._bridge.submit(envelope, vector)
        self.assertEqual(self._executor.calls, 0)
        self.assertEqual(vector.snapshot(), {"S6": "FAILED"})
        self.assertEqual(vector.notes(), {"S6": "MatrixViolationError"})

    def test_lease_contention_records_failure_and_propagates(self) -> None:
        envelope = make_envelope()
        idem_key = make_idempotency_key(
            envelope.task_id, envelope.stage, envelope.seat, envelope.attempt
        )
        LeaseManager(self._conn, self._clock).acquire(idem_key, "seat-other", 1_000)
        vector = StageVector()
        with self.assertRaises(DirectorError):
            self._bridge.submit(envelope, vector)
        self.assertEqual(vector.snapshot(), {"S6": "FAILED"})
        self.assertEqual(vector.notes(), {"S6": "DirectorError"})

    def test_seats_returns_sorted_tuple(self) -> None:
        bridge = OrchBridge(
            self._director,
            {
                "seat-zeta": RecordingExecutor(b"z"),
                "seat-alpha": RecordingExecutor(b"a"),
                "seat-mid": RecordingExecutor(b"m"),
            },
        )
        seats = bridge.seats()
        self.assertIsInstance(seats, tuple)
        self.assertEqual(seats, ("seat-alpha", "seat-mid", "seat-zeta"))

    def test_registry_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            self._bridge._executors["seat-new"] = RecordingExecutor(b"x")
        self.assertEqual(self._bridge.seats(), ("seat-alpha",))


if __name__ == "__main__":
    unittest.main()
