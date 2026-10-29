"""Deterministic unit tests for the ORCH Director sequencing layer.

Coverage: the happy path through all fixed steps, the exactly-once
replay hit, the F4 credential contract (zero side effects on a denied
credential), single-owner lease gating across two directors sharing one
database, executor exception and non-bytes result handling, the evicted
tombstone propagation, the cache store's defense-in-depth matrix gate,
full input validation, the best-effort lease release (including after
expiry and when release raises), and the white-box binding of the
idempotency and cache keys to the attempt identity.  All timing comes
from a fake monotonic clock; no test sleeps or touches the network.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest

from nullcore.orch.cache_key import make_cache_key
from nullcore.orch.cache_matrix import CacheStatus, CacheTier, MatrixViolationError
from nullcore.orch.cache_store import CacheEvictedError, CacheStore
from nullcore.orch.capture import CallCapture, NullReplayReader
from nullcore.orch.director import DirectorError, OrchDirector
from nullcore.orch.idempotency import make_idempotency_key
from nullcore.orch.lease import LeaseError, LeaseManager
from nullcore.orch.state_machine import OrchState

FULL_VECTOR = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}

_COUNT_QUERIES = {
    "leases": "SELECT COUNT(*) FROM leases",
    "calls": "SELECT COUNT(*) FROM calls",
    "cache_entries": "SELECT COUNT(*) FROM cache_entries",
}


class FakeClock:
    """Deterministic injectable monotonic millisecond clock."""

    def __init__(self, now_ms: int = 0) -> None:
        self._now_ms = now_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance_ms(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class RecordingExecutor:
    """Callable that records every request and returns a fixed response."""

    def __init__(self, response: bytes) -> None:
        self._response = response
        self.calls = 0
        self.requests: list[bytes] = []

    def __call__(self, request: bytes) -> bytes:
        self.calls += 1
        self.requests.append(request)
        return self._response


class TestDirector(unittest.TestCase):
    """OrchDirector sequencing, gating, and key-binding tests."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = f"{self._temp_dir.name}/director.sqlite3"
        self._connections: list[sqlite3.Connection] = []
        self._clock = FakeClock()
        self._conn = self._new_connection()
        self._director = OrchDirector(self._conn, self._clock)

    def tearDown(self) -> None:
        for conn in reversed(self._connections):
            conn.close()
        self._connections.clear()
        self._temp_dir.cleanup()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        self._connections.append(conn)
        return conn

    def _count_rows(self, table: str) -> int:
        row = self._conn.execute(_COUNT_QUERIES[table]).fetchone()
        return row[0]

    def _run_attempt(self, base_executor, **overrides):
        """Invoke run_attempt with the standard identity plus overrides.

        The default identity is a fully VERIFIED credential claiming
        TERMINAL_VERIFIED on a clean T0 entry.  ``executor`` inside
        ``overrides`` replaces the base executor (used by the input
        validation test to pass a non-callable).
        """
        executor = overrides.pop("executor", base_executor)
        settings = {
            "task_id": "task-director",
            "stage": "S1",
            "seat": "seat-alpha",
            "attempt": 0,
            "lifecycle_state": OrchState.VERIFIED,
            "stage_vector": FULL_VECTOR,
            "tier": CacheTier.T0,
            "status": CacheStatus.TERMINAL_VERIFIED,
            "degraded_flag": False,
            "config_hash": "cfg-director",
            "request_bytes": b"request",
        }
        settings.update(overrides)
        return self._director.run_attempt(
            settings["task_id"],
            settings["stage"],
            settings["seat"],
            settings["attempt"],
            settings["lifecycle_state"],
            settings["stage_vector"],
            settings["tier"],
            settings["status"],
            settings["degraded_flag"],
            settings["config_hash"],
            settings["request_bytes"],
            executor,
        )

    # ------------------------------------------------------------------
    # 1. Happy path: execute, capture, and cache in one fixed sweep.
    # ------------------------------------------------------------------

    def test_happy_path_executes_captures_and_caches(self) -> None:
        executor = RecordingExecutor(b"happy-response")
        response, from_replay = self._run_attempt(
            base_executor=executor, request_bytes=b"request-happy"
        )
        self.assertEqual(response, b"happy-response")
        self.assertFalse(from_replay)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(executor.requests, [b"request-happy"])
        idem_key = make_idempotency_key("task-director", "S1", "seat-alpha", 0)
        cache_key = make_cache_key("cfg-director", "task-director", "S1", "seat-alpha", 0)
        self.assertEqual(
            NullReplayReader(self._conn).replay(idem_key), b"happy-response"
        )
        self.assertEqual(
            CacheStore(self._conn).read_primary(cache_key), b"happy-response"
        )
        self.assertEqual(self._count_rows("calls"), 1)
        self.assertEqual(self._count_rows("cache_entries"), 1)
        self.assertEqual(self._count_rows("leases"), 0)

    # ------------------------------------------------------------------
    # 2. Replay hit: no execution, no cache rewrite.
    # ------------------------------------------------------------------

    def test_replay_hit_skips_executor_and_does_not_rewrite_cache(self) -> None:
        idem_key = make_idempotency_key("task-director", "S1", "seat-alpha", 0)
        cache_key = make_cache_key("cfg-director", "task-director", "S1", "seat-alpha", 0)
        CallCapture(self._conn).store(
            idem_key,
            "task-director",
            "S1",
            "seat-alpha",
            0,
            b"request",
            b"replayed-response",
            self._clock,
        )
        CacheStore(self._conn).write(
            cache_key,
            "cfg-director",
            "task-director",
            "S1",
            "seat-alpha",
            0,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            b"sentinel-payload",
            self._clock,
        )
        executor = RecordingExecutor(b"happy-response")
        response, from_replay = self._run_attempt(base_executor=executor)
        self.assertEqual(response, b"replayed-response")
        self.assertTrue(from_replay)
        self.assertEqual(executor.calls, 0)
        # The sentinel payload survives: the director never rewrote the
        # cache row on the replay path.
        self.assertEqual(
            CacheStore(self._conn).read_primary(cache_key), b"sentinel-payload"
        )
        self.assertEqual(self._count_rows("cache_entries"), 1)
        self.assertEqual(self._count_rows("leases"), 0)

    # ------------------------------------------------------------------
    # 3. F4 contract: a denied credential blocks every side effect.
    # ------------------------------------------------------------------

    def test_f4_contract_credential_failure_blocks_every_side_effect(self) -> None:
        executor = RecordingExecutor(b"never-used")
        with self.assertRaises(MatrixViolationError):
            self._run_attempt(
                base_executor=executor, lifecycle_state=OrchState.PROVISIONAL
            )
        self.assertEqual(executor.calls, 0)
        self.assertEqual(self._count_rows("leases"), 0)
        self.assertEqual(self._count_rows("calls"), 0)
        self.assertEqual(self._count_rows("cache_entries"), 0)

    # ------------------------------------------------------------------
    # 4. Lease singularity across two directors on one database.
    # ------------------------------------------------------------------

    def test_lease_singularity_across_directors(self) -> None:
        conn_b = self._new_connection()
        director_b = OrchDirector(conn_b, self._clock)
        executor_b = RecordingExecutor(b"never-used")
        idem_key = make_idempotency_key("task-director", "S1", "seat-alpha", 0)
        denied_errors: list[DirectorError] = []
        observed_rows: list[tuple[str, int]] = []
        first_requests: list[bytes] = []

        def holding_executor(request: bytes) -> bytes:
            first_requests.append(request)
            # While the first director holds the lease, the row is live
            # and owned by the attempt seat with the default TTL.
            row = conn_b.execute(
                "SELECT owner_id, expires_at_ms FROM leases "
                "WHERE upgrade_token = ?",
                (idem_key,),
            ).fetchone()
            observed_rows.append((row[0], row[1]))
            # The second director, on its own connection, attempts the
            # identical identity while the lease is actively held.
            try:
                director_b.run_attempt(
                    "task-director",
                    "S1",
                    "seat-alpha",
                    0,
                    OrchState.VERIFIED,
                    FULL_VECTOR,
                    CacheTier.T0,
                    CacheStatus.TERMINAL_VERIFIED,
                    False,
                    "cfg-director",
                    b"request",
                    executor_b,
                )
            except DirectorError as error:
                denied_errors.append(error)
            return b"first-response"

        response, from_replay = self._run_attempt(base_executor=holding_executor)
        self.assertEqual(response, b"first-response")
        self.assertFalse(from_replay)
        self.assertEqual(first_requests, [b"request"])
        self.assertEqual(observed_rows, [("seat-alpha", 30_000)])
        self.assertEqual(len(denied_errors), 1)
        self.assertIn("lease held by another owner", str(denied_errors[0]))
        self.assertEqual(executor_b.calls, 0)
        self.assertEqual(self._count_rows("leases"), 0)
        # A third attempt on a NEW attempt number acquires its own
        # token and completes the full executed path.
        executor_c = RecordingExecutor(b"third-response")
        response_third, from_replay_third = self._run_attempt(
            base_executor=executor_c, attempt=1
        )
        self.assertEqual(response_third, b"third-response")
        self.assertFalse(from_replay_third)
        self.assertEqual(executor_c.calls, 1)
        self.assertEqual(self._count_rows("calls"), 2)
        self.assertEqual(self._count_rows("cache_entries"), 2)

    # ------------------------------------------------------------------
    # 5. Executor exception: lease released, nothing written.
    # ------------------------------------------------------------------

    def test_executor_exception_releases_lease_and_writes_nothing(self) -> None:
        calls = 0

        def flaky_executor(request: bytes) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("executor exploded")
            return b"recovered-response"

        with self.assertRaises(RuntimeError):
            self._run_attempt(base_executor=flaky_executor)
        self.assertEqual(calls, 1)
        self.assertEqual(self._count_rows("leases"), 0)
        self.assertEqual(self._count_rows("calls"), 0)
        self.assertEqual(self._count_rows("cache_entries"), 0)
        # The lease was released in the finally block, so the same
        # identity succeeds on a subsequent attempt.
        response, from_replay = self._run_attempt(base_executor=flaky_executor)
        self.assertEqual(response, b"recovered-response")
        self.assertFalse(from_replay)
        self.assertEqual(calls, 2)
        self.assertEqual(self._count_rows("calls"), 1)
        self.assertEqual(self._count_rows("cache_entries"), 1)

    # ------------------------------------------------------------------
    # 6. Non-bytes executor result: DirectorError, nothing written.
    # ------------------------------------------------------------------

    def test_executor_non_bytes_response_raises_and_writes_nothing(self) -> None:
        calls = 0

        def str_executor(request: bytes) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                return "not-bytes"
            return b"recovered-response"

        with self.assertRaises(DirectorError) as caught:
            self._run_attempt(base_executor=str_executor)
        self.assertEqual(str(caught.exception), "executor must return bytes")
        self.assertEqual(calls, 1)
        self.assertEqual(self._count_rows("leases"), 0)
        self.assertEqual(self._count_rows("calls"), 0)
        self.assertEqual(self._count_rows("cache_entries"), 0)
        # The lease was released, so the same identity succeeds later.
        response, from_replay = self._run_attempt(base_executor=str_executor)
        self.assertEqual(response, b"recovered-response")
        self.assertFalse(from_replay)
        self.assertEqual(calls, 2)
        self.assertEqual(self._count_rows("calls"), 1)

    # ------------------------------------------------------------------
    # 7. Evicted cache tombstone: capture happens, cache write raises.
    # ------------------------------------------------------------------

    def test_cache_evicted_tombstone_propagates(self) -> None:
        cache_key = make_cache_key("cfg-director", "task-director", "S1", "seat-alpha", 0)
        cache = CacheStore(self._conn)
        cache.write(
            cache_key,
            "cfg-director",
            "task-director",
            "S1",
            "seat-alpha",
            0,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            b"original-payload",
            self._clock,
        )
        self._clock.advance_ms(1_000)
        self.assertTrue(cache.evict(cache_key, self._clock))
        executor = RecordingExecutor(b"fresh-response")
        with self.assertRaises(CacheEvictedError):
            self._run_attempt(base_executor=executor)
        # The executor ran and the capture store recorded the call
        # before the cache write raised on the tombstone.
        self.assertEqual(executor.calls, 1)
        self.assertEqual(self._count_rows("calls"), 1)
        self.assertEqual(self._count_rows("leases"), 0)
        # The lease was released despite the propagation: a second run
        # on the same identity acquires again and replays the capture.
        response, from_replay = self._run_attempt(base_executor=executor)
        self.assertEqual(response, b"fresh-response")
        self.assertTrue(from_replay)
        self.assertEqual(executor.calls, 1)

    # ------------------------------------------------------------------
    # 8. Defense in depth: the store's own matrix and flag gates.
    # ------------------------------------------------------------------

    def test_matrix_defense_in_depth_at_store(self) -> None:
        # A DEGRADED claim with a clean flag passes the credential gate
        # (DEGRADED accepts a DRAFTING lifecycle), so the attempt runs
        # and the violation surfaces at the cache store's flag
        # invariant, which fires inside cache.write before any database
        # access: no cache row ever appears.
        executor = RecordingExecutor(b"degraded-response")
        with self.assertRaises(MatrixViolationError):
            self._run_attempt(
                base_executor=executor,
                lifecycle_state=OrchState.DRAFTING,
                tier=CacheTier.T0,
                status=CacheStatus.DEGRADED,
                degraded_flag=False,
            )
        self.assertEqual(executor.calls, 1)
        # Capture precedes the cache write by design, so the call row
        # exists; the cache store itself never wrote a row.
        self.assertEqual(self._count_rows("calls"), 1)
        self.assertEqual(self._count_rows("cache_entries"), 0)
        self.assertEqual(self._count_rows("leases"), 0)
        # Defense-in-depth existence check, independent of the
        # director: the store's own WRITE_MATRIX gate rejects an illegal
        # (status, tier) cell outright.
        illegal_cache_key = make_cache_key(
            "cfg-director", "task-matrix", "S2", "seat-matrix", 0
        )
        with self.assertRaises(MatrixViolationError):
            CacheStore(self._conn).write(
                illegal_cache_key,
                "cfg-director",
                "task-matrix",
                "S2",
                "seat-matrix",
                0,
                CacheTier.T0,
                CacheStatus.PROVISIONAL,
                False,
                b"payload",
                self._clock,
            )
        self.assertEqual(self._count_rows("cache_entries"), 0)

    # ------------------------------------------------------------------
    # 9. Input validation: every malformed argument raises first.
    # ------------------------------------------------------------------

    def test_run_attempt_validates_inputs(self) -> None:
        executor = RecordingExecutor(b"never-used")
        cases = [
            ("empty task_id", {"task_id": ""}),
            ("empty stage", {"stage": ""}),
            ("empty seat", {"seat": ""}),
            ("empty config_hash", {"config_hash": ""}),
            ("bool attempt", {"attempt": True}),
            ("negative attempt", {"attempt": -1}),
            ("non-int attempt", {"attempt": "0"}),
            ("non-enum lifecycle", {"lifecycle_state": "VERIFIED"}),
            ("non-enum tier", {"tier": "T0"}),
            ("non-enum status", {"status": "TERMINAL_VERIFIED"}),
            ("non-bool degraded_flag", {"degraded_flag": "yes"}),
            ("non-bytes request", {"request_bytes": "not-bytes"}),
            ("non-callable executor", {"executor": None}),
        ]
        for label, overrides in cases:
            with self.subTest(case=label):
                with self.assertRaises(DirectorError):
                    self._run_attempt(base_executor=executor, **overrides)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(self._count_rows("leases"), 0)
        self.assertEqual(self._count_rows("calls"), 0)
        self.assertEqual(self._count_rows("cache_entries"), 0)

    # ------------------------------------------------------------------
    # 10. Best-effort release: expiry and refusal are both swallowed.
    # ------------------------------------------------------------------

    def test_release_is_best_effort_after_expiry(self) -> None:
        # Failure mode demonstration with a bare LeaseManager: once the
        # TTL has passed, release fails by returning False and the row
        # is left for expiry-based takeover.
        bare_lease = LeaseManager(self._conn, self._clock)
        self.assertTrue(bare_lease.acquire("token-expiry", "seat-expiry", 50))
        self._clock.advance_ms(100)
        self.assertFalse(bare_lease.release("token-expiry", "seat-expiry"))
        # End-to-end: the slow executor runs the clock past the lease
        # TTL, so the finally-block release targets an expired lease
        # and fails; the best-effort swallow still returns the happy
        # tuple unchanged.

        def slow_executor(request: bytes) -> bytes:
            self._clock.advance_ms(100)
            return b"slow-response"

        director = OrchDirector(self._conn, self._clock, lease_ttl_ms=50)
        response, from_replay = director.run_attempt(
            "task-expiry",
            "S1",
            "seat-expiry-director",
            0,
            OrchState.VERIFIED,
            FULL_VECTOR,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            "cfg-expiry",
            b"request",
            slow_executor,
        )
        self.assertEqual(response, b"slow-response")
        self.assertFalse(from_replay)
        # Wrapper check: force release to raise LeaseError outright on
        # a fresh identity; the director must swallow the error and
        # still return the happy tuple.

        class RefusingLease:
            """Delegate acquisition to the real lease; release raises."""

            def __init__(self, inner: LeaseManager) -> None:
                self._inner = inner

            def acquire(self, token: str, owner_id: str, ttl_ms: int) -> bool:
                return self._inner.acquire(token, owner_id, ttl_ms)

            def release(self, token: str, owner_id: str) -> bool:
                raise LeaseError("release refused by wrapper")

        director._lease = RefusingLease(director._lease)
        executor = RecordingExecutor(b"wrapped-response")
        response, from_replay = director.run_attempt(
            "task-expiry",
            "S1",
            "seat-wrapped",
            0,
            OrchState.VERIFIED,
            FULL_VECTOR,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            "cfg-expiry",
            b"request",
            executor,
        )
        self.assertEqual(response, b"wrapped-response")
        self.assertFalse(from_replay)
        self.assertEqual(executor.calls, 1)

    # ------------------------------------------------------------------
    # 11. White-box key binding: the director derives the exact keys.
    # ------------------------------------------------------------------

    def test_idempotency_and_cache_keys_bind_to_identity(self) -> None:
        task_id, stage, seat = "task-keybind", "S1", "seat-keybind"
        idem_key = make_idempotency_key(task_id, stage, seat, 0)
        cache_key = make_cache_key("cfg-keybind", task_id, stage, seat, 0)
        # A config hash drift alone produces a different cache key.
        self.assertNotEqual(
            cache_key, make_cache_key("cfg-keybind-b", task_id, stage, seat, 0)
        )
        # Pre-seed the capture store at the expected idempotency key:
        # the director must replay from exactly that key.
        CallCapture(self._conn).store(
            idem_key,
            task_id,
            stage,
            seat,
            0,
            b"request",
            b"replayed-response",
            self._clock,
        )
        idle_executor = RecordingExecutor(b"never-used")
        response, from_replay = self._director.run_attempt(
            task_id,
            stage,
            seat,
            0,
            OrchState.VERIFIED,
            FULL_VECTOR,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            "cfg-keybind",
            b"request",
            idle_executor,
        )
        self.assertEqual(response, b"replayed-response")
        self.assertTrue(from_replay)
        self.assertEqual(idle_executor.calls, 0)
        # Two executing runs with drifted configs land in distinct
        # cache rows.  The replay identity excludes config_hash, so a
        # config-drifted re-execution rides a new attempt number, and
        # each run caches under its own config-bound key.
        executor_a = RecordingExecutor(b"response-a")
        response_a, from_replay_a = self._director.run_attempt(
            task_id,
            stage,
            seat,
            1,
            OrchState.VERIFIED,
            FULL_VECTOR,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            "cfg-keybind-a",
            b"request",
            executor_a,
        )
        self.assertEqual(response_a, b"response-a")
        self.assertFalse(from_replay_a)
        self.assertEqual(executor_a.calls, 1)
        executor_b = RecordingExecutor(b"response-b")
        response_b, from_replay_b = self._director.run_attempt(
            task_id,
            stage,
            seat,
            2,
            OrchState.VERIFIED,
            FULL_VECTOR,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            "cfg-keybind-b",
            b"request",
            executor_b,
        )
        self.assertEqual(response_b, b"response-b")
        self.assertFalse(from_replay_b)
        self.assertEqual(executor_b.calls, 1)
        key_a = make_cache_key("cfg-keybind-a", task_id, stage, seat, 1)
        key_b = make_cache_key("cfg-keybind-b", task_id, stage, seat, 2)
        self.assertNotEqual(key_a, key_b)
        cache = CacheStore(self._conn)
        self.assertEqual(cache.read_primary(key_a), b"response-a")
        self.assertEqual(cache.read_primary(key_b), b"response-b")
        self.assertEqual(self._count_rows("cache_entries"), 2)


if __name__ == "__main__":
    unittest.main()
