"""Deterministic unit tests for the web door over the ORCH bridge.

The FastAPI route handlers are exercised directly: ``build_router``
closes a real OrchBridge over a real OrchDirector backed by a
throwaway SQLite database, and every test pulls the endpoint callables
out of ``router.routes`` and calls them synchronously with
``TaskRequest`` instances -- no HTTP client, no server, no network.
Coverage: the happy executed payload, the replay payload with an
untouched executor, the full 4xx mapping (invalid base64, invalid enum
strings for every mapped field, envelope rejection naming the field,
unknown seat, credential denial, lease contention, evicted tombstone,
cache conflict), the seats listing, the router construction gate, and
the unmapped executor exception that must propagate unchanged.  All
timing comes from a fake monotonic clock; the connection is closed in
tearDown; no test sleeps or touches the network.
"""

from __future__ import annotations

import base64
import sqlite3
import tempfile
import unittest

from fastapi import HTTPException

from nullcore.orch.bridge import BridgeError, OrchBridge
from nullcore.orch.cache_matrix import CacheStatus, CacheTier
from nullcore.orch.cache_key import make_cache_key
from nullcore.orch.cache_store import CacheStore
from nullcore.orch.capture import CallCapture
from nullcore.orch.director import OrchDirector
from nullcore.orch.idempotency import make_idempotency_key
from nullcore.orch.lease import LeaseManager
from nullcore.orch.web_routes import TaskRequest, build_router

FULL_VECTOR = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}


class FakeClock:
    """Deterministic injectable monotonic millisecond clock."""

    def __init__(self, now_ms: int = 0) -> None:
        self._now_ms = now_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance_ms(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class CountingExecutor:
    """Callable that counts calls and returns a fixed response."""

    def __init__(self, response: bytes) -> None:
        self._response = response
        self.calls = 0

    def __call__(self, request: bytes) -> bytes:
        self.calls += 1
        return self._response


def make_request(**overrides) -> TaskRequest:
    """Return a fully valid VERIFIED wire request with overrides."""
    fields = {
        "task_id": "task-web",
        "stage": "S1",
        "seat": "seat-alpha",
        "attempt": 0,
        "config_hash": "cfg-web",
        "request_b64": base64.b64encode(b"request").decode("ascii"),
        "lifecycle_state": "VERIFIED",
        "stage_vector": dict(FULL_VECTOR),
        "tier": "T0",
        "status": "TERMINAL_VERIFIED",
        "degraded_flag": False,
    }
    fields.update(overrides)
    return TaskRequest(**fields)


def task_endpoint(router) -> callable:
    """Pull the POST /api/task endpoint callable out of the router."""
    return next(
        route.endpoint for route in router.routes if route.path == "/api/task"
    )


def seats_endpoint(router) -> callable:
    """Pull the GET /api/seats endpoint callable out of the router."""
    return next(
        route.endpoint for route in router.routes if route.path == "/api/seats"
    )


class TestWebRoutes(unittest.TestCase):
    """Route construction, payload shapes, and the 4xx mapping."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = f"{self._temp_dir.name}/web_routes.sqlite3"
        self._clock = FakeClock()
        self._conn = sqlite3.connect(self._db_path)
        self._director = OrchDirector(self._conn, self._clock)
        self._executor = CountingExecutor(b"web-response")
        self._bridge = OrchBridge(self._director, {"seat-alpha": self._executor})
        self._router = build_router(self._bridge)
        self._submit = task_endpoint(self._router)
        self._seats = seats_endpoint(self._router)

    def tearDown(self) -> None:
        self._conn.close()
        self._temp_dir.cleanup()

    def test_build_router_rejects_non_bridge(self) -> None:
        for bad_bridge in ("not-a-bridge", None, object()):
            with self.subTest(bad_bridge=bad_bridge):
                with self.assertRaises(BridgeError):
                    build_router(bad_bridge)

    def test_happy_path_returns_executed_payload(self) -> None:
        payload = self._submit(make_request())
        self.assertEqual(
            set(payload), {"response_b64", "from_replay", "stage_vector", "notes"}
        )
        self.assertEqual(
            payload["response_b64"],
            base64.b64encode(b"web-response").decode("ascii"),
        )
        self.assertFalse(payload["from_replay"])
        self.assertEqual(payload["stage_vector"], {"S6": "PASSED"})
        self.assertEqual(payload["notes"], {"S6": "executed"})
        self.assertEqual(self._executor.calls, 1)

    def test_replay_path_returns_replay_payload(self) -> None:
        task_id, stage, seat, attempt = "task-web", "S1", "seat-alpha", 0
        idem_key = make_idempotency_key(task_id, stage, seat, attempt)
        CallCapture(self._conn).store(
            idem_key,
            task_id,
            stage,
            seat,
            attempt,
            b"request",
            b"replayed-response",
            self._clock,
        )
        payload = self._submit(make_request())
        self.assertEqual(
            payload["response_b64"],
            base64.b64encode(b"replayed-response").decode("ascii"),
        )
        self.assertTrue(payload["from_replay"])
        self.assertEqual(self._executor.calls, 0)
        self.assertEqual(payload["stage_vector"], {"S6": "PASSED"})
        self.assertEqual(payload["notes"], {"S6": "replay"})

    def test_bad_enum_strings_map_to_400(self) -> None:
        cases = [
            ("lifecycle_state", {"lifecycle_state": "NOT_A_STATE"}),
            ("tier", {"tier": "T9"}),
            ("status", {"status": "NO_SUCH_STATUS"}),
        ]
        for field_name, overrides in cases:
            with self.subTest(field=field_name):
                with self.assertRaises(HTTPException) as caught:
                    self._submit(make_request(**overrides))
                self.assertEqual(caught.exception.status_code, 400)
                self.assertIn(field_name, caught.exception.detail)

    def test_bad_request_b64_maps_to_400(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            self._submit(make_request(request_b64="%%%not-base64%%%"))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("request_b64", caught.exception.detail)

    def test_empty_task_id_maps_to_400_naming_the_field(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            self._submit(make_request(task_id=""))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("task_id", caught.exception.detail)

    def test_unknown_seat_maps_to_400(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            self._submit(make_request(seat="seat-ghost"))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("seat-ghost", caught.exception.detail)

    def test_credential_denial_maps_to_403(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            self._submit(
                make_request(lifecycle_state="PROVISIONAL", status="TERMINAL_VERIFIED")
            )
        self.assertEqual(caught.exception.status_code, 403)
        self.assertEqual(self._executor.calls, 0)

    def test_lease_preheld_maps_to_409(self) -> None:
        idem_key = make_idempotency_key("task-web", "S1", "seat-alpha", 0)
        LeaseManager(self._conn, self._clock).acquire(idem_key, "seat-other", 60_000)
        with self.assertRaises(HTTPException) as caught:
            self._submit(make_request())
        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("lease held", caught.exception.detail)

    def test_evicted_tombstone_maps_to_410(self) -> None:
        cache_key = make_cache_key("cfg-web", "task-web", "S1", "seat-alpha", 0)
        cache = CacheStore(self._conn)
        cache.write(
            cache_key,
            "cfg-web",
            "task-web",
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
        with self.assertRaises(HTTPException) as caught:
            self._submit(make_request())
        self.assertEqual(caught.exception.status_code, 410)
        self.assertEqual(self._executor.calls, 1)

    def test_cache_conflict_maps_to_409(self) -> None:
        cache_key = make_cache_key("cfg-web", "task-web", "S1", "seat-alpha", 0)
        CacheStore(self._conn).write(
            cache_key,
            "cfg-web",
            "task-web",
            "S1",
            "seat-alpha",
            0,
            CacheTier.T0,
            CacheStatus.TERMINAL_VERIFIED,
            False,
            b"original-payload",
            self._clock,
        )
        with self.assertRaises(HTTPException) as caught:
            self._submit(make_request())
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self._executor.calls, 1)

    def test_seats_endpoint_returns_sorted_list(self) -> None:
        self.assertEqual(self._seats(), {"seats": ["seat-alpha"]})
        executor = CountingExecutor(b"shared")
        bridge = OrchBridge(
            self._director,
            {
                "seat-zeta": executor,
                "seat-alpha": executor,
                "seat-mid": executor,
            },
        )
        payload = seats_endpoint(build_router(bridge))()
        self.assertEqual(payload, {"seats": ["seat-alpha", "seat-mid", "seat-zeta"]})
        self.assertIsInstance(payload["seats"], list)

    def test_executor_exception_propagates_unmapped(self) -> None:
        def exploding_executor(request: bytes) -> bytes:
            raise RuntimeError("executor exploded")

        bridge = OrchBridge(self._director, {"seat-alpha": exploding_executor})
        handler = task_endpoint(build_router(bridge))
        with self.assertRaises(RuntimeError):
            handler(make_request())


if __name__ == "__main__":
    unittest.main()
