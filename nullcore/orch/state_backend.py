"""SQLite state backend: one connection factory with the pinned pragmas.

:func:`create_connection` is the single place an ORCH process opens a
database.  It validates the path, connects, and installs the pinned
pragmas before the connection escapes: a 5000 ms busy timeout on every
connection so lock contention between concurrent writers retries
instead of failing immediately, plus write-ahead journaling and
``synchronous=NORMAL`` on every file-backed connection for durable
cross-process sharing.  A ``:memory:`` database skips the WAL pair
because SQLite keeps an in-memory journal in RAM -- requesting WAL
there is a silent no-op -- and a private in-memory database has
nothing to gain from it.

The connection is opened with ``check_same_thread=False`` because
FastAPI serves synchronous endpoints on threadpool workers: a
connection created at import time in the main thread is always used
from a foreign thread in production.  SQLite's serialized build, the
WAL journal, and the busy timeout keep that cross-thread access safe;
the Python-level guard would only forbid a usage pattern every ORCH
host relies on.

The function owns no schema: the M1a primitives that the
:class:`~nullcore.orch.director.OrchDirector` composes install their
own tables and re-run the same pragmas, which is harmless because
every pragma statement is idempotent.  There is no module-level state
and no I/O at import time; the caller owns the returned connection
and is responsible for closing it.
"""

from __future__ import annotations

import sqlite3

__all__ = ["create_connection"]


def create_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Open ``db_path`` and install the pinned ORCH pragmas.

    Args:
        db_path: SQLite database path.  ``":memory:"`` (the default)
            keeps the database private to this process and skips the
            WAL journaling pair; any other path is opened on disk and
            takes write-ahead journaling with ``synchronous=NORMAL``.

    Returns:
        A live :class:`sqlite3.Connection` with the busy timeout (and,
        on a file-backed path, the WAL pair) already applied, usable
        from any thread.  Closing it belongs to the caller.

    Raises:
        ValueError: when ``db_path`` is not a str or is empty.  An
            empty path would silently create a private temporary
            database, so it is rejected up front rather than passed to
            SQLite.
    """
    if not isinstance(db_path, str):
        raise ValueError(
            f"db_path must be a str, got {type(db_path).__name__}"
        )
    if not db_path:
        raise ValueError("db_path must be a non-empty str")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=5000;")
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    return conn
