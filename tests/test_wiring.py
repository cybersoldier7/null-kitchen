"""Deterministic unit tests for the M2 state backend and app wiring.

The tests drive :func:`~nullcore.orch.wiring.build_services` exactly
as a host process would: build the stack, inspect the returned
mapping, and exercise the ``/api/task`` endpoint callable directly
(pulled out of ``router.routes``) -- no HTTP client, no server, no
network.  The dummy ``seat-alpha`` executor answers every request with
``b"dummy-response"``; the clock is the real wall clock, but no test
sleeps or depends on timing, so the suite stays deterministic.

Coverage: the four-key services mapping with the exact types, the
pinned connection pragmas (the 5000 ms busy timeout everywhere, the
RAM journal preserved on ``:memory:``, and WAL plus
``synchronous=NORMAL`` on file-backed databases), the two M1e routes
on the assembled router, and a full end-to-end submission over
``":memory:"`` whose executed payload carries the strict base64
response, ``from_replay`` False, and the recorded ``S6`` verdict.
Every connection a test opens is closed by that test; the only files
touched live under a throwaway temporary directory.
"""

from __future__ import annotations

import base64
import sqlite3
import tempfile
import unittest

from fastapi import APIRouter

from nullcore.orch.bridge import OrchBridge
from nullcore.orch.director import OrchDirector
from nullcore.orch.state_backend import create_connection
from nullcore.orch.web_routes import TaskRequest
from nullcore.orch.wiring import build_services

FULL_VECTOR = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}

SERVICE_KEYS = {"connection", "director", "bridge", "router"}


def task_endpoint(router: APIRouter):
    """Pull the POST /api/task endpoint callable out of the router."""
    return next(
        route.endpoint for route in router.routes if route.path == "/api/task"
    )


class TestWiring(unittest.TestCase):
    """The composition root assembles a complete, working ORCH stack."""

    def test_build_services_returns_correct_keys(self) -> None:
        services = build_services()
        try:
            self.assertEqual(set(services), SERVICE_KEYS)
            self.assertIsInstance(services["connection"], sqlite3.Connection)
            self.assertIsInstance(services["director"], OrchDirector)
            self.assertIsInstance(services["bridge"], OrchBridge)
            self.assertIsInstance(services["router"], APIRouter)
        finally:
            services["connection"].close()

    def test_connection_pragmas(self) -> None:
        # A raw in-memory connection takes the busy timeout and keeps
        # its RAM journal: the WAL pair must not be requested there.
        memory_conn = create_connection()
        try:
            self.assertEqual(
                memory_conn.execute("PRAGMA busy_timeout").fetchone()[0],
                5000,
            )
            self.assertEqual(
                memory_conn.execute("PRAGMA journal_mode").fetchone()[0],
                "memory",
            )
        finally:
            memory_conn.close()
        # The assembled stack's connection carries the same timeout.
        services = build_services()
        try:
            self.assertEqual(
                services["connection"]
                .execute("PRAGMA busy_timeout")
                .fetchone()[0],
                5000,
            )
        finally:
            services["connection"].close()
        # A file-backed connection additionally takes WAL journaling
        # and synchronous=NORMAL for durable cross-process sharing.
        with tempfile.TemporaryDirectory() as temp_dir:
            file_conn = create_connection(f"{temp_dir}/state.sqlite3")
            try:
                self.assertEqual(
                    file_conn.execute("PRAGMA busy_timeout").fetchone()[0],
                    5000,
                )
                self.assertEqual(
                    file_conn.execute("PRAGMA journal_mode").fetchone()[0],
                    "wal",
                )
                self.assertEqual(
                    file_conn.execute("PRAGMA synchronous").fetchone()[0],
                    1,
                )
            finally:
                file_conn.close()

    def test_router_has_routes(self) -> None:
        services = build_services()
        try:
            paths = {route.path for route in services["router"].routes}
            self.assertIn("/api/task", paths)
            self.assertIn("/api/seats", paths)
        finally:
            services["connection"].close()

    def test_end_to_end_wiring(self) -> None:
        services = build_services(":memory:")
        submit = task_endpoint(services["router"])
        request = TaskRequest(
            task_id="task-m2",
            stage="S1",
            seat="seat-alpha",
            attempt=0,
            config_hash="cfg-m2",
            request_b64=base64.b64encode(b"request").decode("ascii"),
            lifecycle_state="VERIFIED",
            stage_vector=dict(FULL_VECTOR),
            tier="T0",
            status="TERMINAL_VERIFIED",
            degraded_flag=False,
        )
        try:
            payload = submit(request)
            self.assertEqual(
                set(payload),
                {"response_b64", "from_replay", "stage_vector", "notes"},
            )
            self.assertEqual(
                payload["response_b64"],
                base64.b64encode(b"dummy-response").decode("ascii"),
            )
            self.assertFalse(payload["from_replay"])
            self.assertEqual(payload["stage_vector"], {"S6": "PASSED"})
            self.assertEqual(payload["notes"], {"S6": "executed"})
        finally:
            services["connection"].close()


if __name__ == "__main__":
    unittest.main()
