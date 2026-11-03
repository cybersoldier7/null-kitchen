"""Deterministic unit tests for the SSE status stream over the ORCH bridge.

The FastAPI route handler is exercised directly: ``build_stream_router``
closes a real OrchBridge over a real OrchDirector backed by a throwaway
SQLite database, the ``GET /api/stream/{task_id}`` endpoint callable is
pulled out of ``router.routes``, and every test calls it synchronously
with plain keyword arguments -- no HTTP client, no server, no network.
The returned ``StreamingResponse`` is consumed through its
``body_iterator`` into a plain list of frame strings.  starlette wraps
the endpoint's synchronous generator with ``iterate_in_threadpool``,
so the body iterator is an async generator even though every frame
comes from plain synchronous code; a short ``asyncio.run`` drain (no
server, no network, frames in yield order) collects it.  Because each
frame is pulled from a threadpool worker, ``setUp`` opens its SQLite
connection with ``check_same_thread=False`` -- the drain keeps access
strictly sequential, so no locking concerns arise.

The happy-path credential is ``VERIFIED`` lifecycle with a
``PROVISIONAL`` claim on ``T2``: a query string cannot carry a
stage-vector mapping, so every envelope the door builds carries an
empty credential vector, and a ``TERMINAL_VERIFIED`` claim -- which
the F4 gate requires to show S5, S6, and S7 PASSED -- would be
refused.  No stage evidence is fabricated.

Coverage: the executed stream (start, verdict with S6 PASSED and the
executed note, done), the replay stream (S6 PASSED with the replay
note and an idle executor), the bad-enum stream (a single error frame
naming the field, then done, with no start), the credential-denial
stream (S6 FAILED MatrixViolationError), and the executor-exception
stream (S6 FAILED RuntimeError).  All timing comes from a fake
monotonic clock; the connection is closed in tearDown; no test sleeps
or touches the network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
import tempfile
import unittest

from fastapi.responses import StreamingResponse

from nullcore.orch.bridge import OrchBridge
from nullcore.orch.director import OrchDirector
from nullcore.orch.stream_routes import build_stream_router


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


def stream_endpoint(router):
    """Pull the GET /api/stream/{task_id} endpoint out of the router."""
    return next(
        route.endpoint
        for route in router.routes
        if route.path == "/api/stream/{task_id}"
    )


def drain(response: StreamingResponse) -> list[str]:
    """Collect a StreamingResponse body into a plain list of strings.

    starlette wraps the endpoint's sync generator with
    ``iterate_in_threadpool``, so the body iterator is async; the
    short asyncio drain below collects the frames in yield order
    without any server or network.
    """
    body = response.body_iterator

    async def collect() -> list[str]:
        return [frame async for frame in body]

    return asyncio.run(collect())


def parse_events(frames: list[str]) -> list[tuple[str, dict]]:
    """Parse SSE frames into (event name, data payload) pairs."""
    events: list[tuple[str, dict]] = []
    for frame in frames:
        if not (isinstance(frame, str) and frame.endswith("\n\n")):
            raise AssertionError(
                f"frame is not a newline-terminated SSE frame: {frame!r}"
            )
        lines = frame[:-2].split("\n")
        if (
            len(lines) != 2
            or not lines[0].startswith("event: ")
            or not lines[1].startswith("data: ")
        ):
            raise AssertionError(
                f"frame is not a two-line SSE event: {frame!r}"
            )
        events.append(
            (
                lines[0][len("event: "):],
                json.loads(lines[1][len("data: "):]),
            )
        )
    return events


class TestStreamRoutes(unittest.TestCase):
    """Event sequences for the five streamed outcomes."""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = f"{self._temp_dir.name}/stream_routes.sqlite3"
        # check_same_thread=False: starlette pulls each frame through
        # a threadpool worker, so the connection must be usable across
        # threads; the drain keeps access strictly sequential.
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._clock = FakeClock()
        self._director = OrchDirector(self._conn, self._clock)
        self._executor = CountingExecutor(b"sse-response")
        self._bridge = OrchBridge(
            self._director, {"seat-alpha": self._executor}
        )
        self._router = build_stream_router(self._bridge)
        self._stream = stream_endpoint(self._router)

    def tearDown(self) -> None:
        self._conn.close()
        self._temp_dir.cleanup()

    def _events(self, endpoint=None, **overrides) -> list[tuple[str, dict]]:
        """Call a stream endpoint with valid params plus overrides."""
        stream = self._stream if endpoint is None else endpoint
        params = {
            "task_id": "task-sse",
            "stage": "S1",
            "seat": "seat-alpha",
            "attempt": 0,
            "config_hash": "cfg-sse",
            "request_b64": base64.b64encode(b"request").decode("ascii"),
            "lifecycle_state": "VERIFIED",
            "tier": "T2",
            "status": "PROVISIONAL",
            "degraded_flag": False,
        }
        params.update(overrides)
        response = stream(**params)
        self.assertIsInstance(response, StreamingResponse)
        self.assertEqual(response.media_type, "text/event-stream")
        return parse_events(drain(response))

    def test_happy_path_yields_start_verdict_done(self) -> None:
        events = self._events()
        self.assertEqual(
            [name for name, _payload in events],
            ["start", "verdict", "done"],
        )
        self.assertEqual(events[0][1], {})
        self.assertEqual(
            events[1][1],
            {
                "stage_vector": {"S6": "PASSED"},
                "notes": {"S6": "executed"},
                "from_replay": False,
            },
        )
        self.assertEqual(events[2][1], {})
        self.assertEqual(self._executor.calls, 1)

    def test_replay_path_yields_start_verdict_replay_done(self) -> None:
        first = self._events()
        self.assertEqual(
            [name for name, _payload in first],
            ["start", "verdict", "done"],
        )
        self.assertFalse(first[1][1]["from_replay"])
        second = self._events()
        self.assertEqual(
            [name for name, _payload in second],
            ["start", "verdict", "done"],
        )
        self.assertEqual(
            second[1][1],
            {
                "stage_vector": {"S6": "PASSED"},
                "notes": {"S6": "replay"},
                "from_replay": True,
            },
        )
        self.assertEqual(self._executor.calls, 1)

    def test_bad_enum_yields_error_then_done(self) -> None:
        cases = [
            ("lifecycle_state", "NOT_A_STATE"),
            ("tier", "T9"),
            ("status", "NO_SUCH_STATUS"),
        ]
        for field, bad_value in cases:
            with self.subTest(field=field):
                events = self._events(**{field: bad_value})
                self.assertEqual(
                    [name for name, _payload in events],
                    ["error", "done"],
                )
                self.assertIn(field, events[0][1]["error"])
                self.assertEqual(events[1][1], {})

    def test_credential_denial_yields_error_then_done(self) -> None:
        events = self._events(
            lifecycle_state="PROVISIONAL", status="TERMINAL_VERIFIED"
        )
        self.assertEqual(
            [name for name, _payload in events],
            ["start", "error", "done"],
        )
        self.assertEqual(
            events[1][1],
            {
                "stage_vector": {"S6": "FAILED"},
                "notes": {"S6": "MatrixViolationError"},
                "error": "MatrixViolationError",
            },
        )
        self.assertEqual(self._executor.calls, 0)

    def test_executor_exception_yields_error_then_done(self) -> None:
        def exploding_executor(request: bytes) -> bytes:
            raise RuntimeError("executor exploded")

        bridge = OrchBridge(
            self._director, {"seat-alpha": exploding_executor}
        )
        events = self._events(
            endpoint=stream_endpoint(build_stream_router(bridge))
        )
        self.assertEqual(
            [name for name, _payload in events],
            ["start", "error", "done"],
        )
        self.assertEqual(
            events[1][1],
            {
                "stage_vector": {"S6": "FAILED"},
                "notes": {"S6": "RuntimeError"},
                "error": "RuntimeError",
            },
        )


if __name__ == "__main__":
    unittest.main()
