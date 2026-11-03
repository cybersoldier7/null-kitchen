"""Cross-thread reachability regression for the state backend.

FastAPI serves synchronous endpoints on threadpool workers, so any
connection owned by the composition root must be usable from a thread
other than the one that created it.  This regression test pins that
property: a connection from create_connection() is driven from a
fresh thread, and a full OrchDirector attempt must succeed there.
Historically this exact seam raised sqlite3.ProgrammingError in
production while every same-thread test stayed green.
"""

from __future__ import annotations

import threading
import unittest

from nullcore.orch.cache_matrix import CacheStatus, CacheTier
from nullcore.orch.director import OrchDirector
from nullcore.orch.state_machine import OrchState
from nullcore.orch.state_backend import create_connection

FULL_VECTOR = {"S5": "PASSED", "S6": "PASSED", "S7": "PASSED"}


class TestThreadReach(unittest.TestCase):
    """The state backend connection survives a thread hop."""

    def test_director_attempt_succeeds_on_foreign_thread(self) -> None:
        conn = create_connection()
        results: list = []

        def worker() -> None:
            director = OrchDirector(conn, lambda: 0)
            try:
                response, from_replay = director.run_attempt(
                    "task-thread",
                    "S1",
                    "seat-alpha",
                    0,
                    OrchState.VERIFIED,
                    FULL_VECTOR,
                    CacheTier.T0,
                    CacheStatus.TERMINAL_VERIFIED,
                    False,
                    "cfg-thread",
                    b"request",
                    lambda req: b"thread-response",
                )
                results.append((response, from_replay))
            except BaseException as exc:
                results.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        outcome = results[0]
        self.assertNotIsInstance(outcome, BaseException)
        self.assertEqual(outcome, (b"thread-response", False))
        conn.close()


if __name__ == "__main__":
    unittest.main()
