"""Composition root wiring the full ORCH stack into one services dict.

:func:`build_services` is the single assembly point of the NULL
Orchestrator Core: it opens the state backend connection, mints the
real wall clock, closes both into an
:class:`~nullcore.orch.director.OrchDirector`, registers the dummy
``seat-alpha`` executor on an :class:`~nullcore.orch.bridge.OrchBridge`,
and exposes the M1e web door through
:func:`~nullcore.orch.web_routes.build_router`.  The returned mapping
is everything a host process needs: mount ``"router"`` on an ASGI
application, or drive ``"bridge"`` and ``"director"`` directly
in-process.

The function is a pure factory: no module-level state, no I/O at
import time, and every collaborator is constructed fresh per call, so
two calls yield two fully independent stacks, each owning its own
connection.  The clock is deliberately the real one --
``time.time_ns()`` folded to whole milliseconds -- because this module
is the seam between the deterministic core (which accepts any
injectable clock) and a live process; callers needing determinism
build their own stack with a fake clock instead of rewiring this one.

The Errand 1 ``app.py`` stays untouched: mounting this router is a
host-level decision reserved for a later errand.
"""

from __future__ import annotations

import time

from nullcore.orch.bridge import OrchBridge
from nullcore.orch.director import OrchDirector
from nullcore.orch.state_backend import create_connection
from nullcore.orch.web_routes import build_router

__all__ = ["build_services"]


def build_services(db_path: str = ":memory:") -> dict:
    """Assemble one complete ORCH stack over ``db_path``.

    Args:
        db_path: SQLite database path handed to
            :func:`~nullcore.orch.state_backend.create_connection`.
            The default ``":memory:"`` keeps the whole stack private
            to this process; a file path shares the state backend
            across processes.

    Returns:
        A dict with four keys: ``"connection"`` (the open
        :class:`sqlite3.Connection` the stack runs on -- the caller
        owns closing it), ``"director"`` (the
        :class:`~nullcore.orch.director.OrchDirector` over that
        connection and the real wall clock), ``"bridge"`` (the
        :class:`~nullcore.orch.bridge.OrchBridge` over the director
        and the dummy seat registry), and ``"router"`` (the
        :class:`fastapi.APIRouter` web door over the bridge).
    """
    connection = create_connection(db_path)
    # Real wall clock: nanoseconds folded to whole milliseconds.  The
    # deterministic core never reads a clock itself; only this
    # composition root pins a live process to real time.
    director = OrchDirector(
        connection,
        lambda: int(time.time_ns() // 1_000_000),
    )
    # Dummy seat registry: every seat-alpha request gets the same
    # canned response until a real EXEC executor replaces it.
    executors = {"seat-alpha": lambda req: b"dummy-response"}
    bridge = OrchBridge(director, executors)
    router = build_router(bridge)
    return {
        "connection": connection,
        "director": director,
        "bridge": bridge,
        "router": router,
  }
