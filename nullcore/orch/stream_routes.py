"""Server-Sent Events status stream over the ORCH bridge.

``build_stream_router(bridge)`` closes one validated
:class:`~nullcore.orch.bridge.OrchBridge` into a fresh
:class:`fastapi.APIRouter` exposing exactly one synchronous endpoint:
``GET /api/stream/{task_id}``.  The handler validates the wire input
exactly like the M1e submit door -- strict base64 decoding, enum
mapping through the value constructors, and the M1d envelope build --
but reports every outcome as Server-Sent Events on a
``text/event-stream`` response instead of a single JSON body:

* ``start`` carries the initial snapshot of a fresh
  :class:`~nullcore.orch.task_envelope.StageVector`;
* ``verdict`` carries the final snapshot, the notes, and the replay
  flag of a successful submit;
* ``error`` carries the final snapshot, the notes, and the exception
  type name of a failed submit (the bridge has already recorded
  ``S6`` ``FAILED`` with that type as the note);
* ``done`` is the empty terminator that ends every stream.

A validation failure never starts a stream: the response is a single
``error`` frame naming the offending field, followed by the ``done``
terminator.  Because a query string cannot carry a stage-vector
mapping, every envelope built here carries an empty credential stage
vector -- an SSE caller starts with no prior verdicts -- so a
``TERMINAL_VERIFIED`` claim, which the F4 gate requires to show S5,
S6, and S7 ``PASSED``, is refused while ``PROVISIONAL`` and
``DEGRADED`` claims remain available.  No stage evidence is ever
fabricated on the caller's behalf.

The module holds no state at import time and performs no I/O: the
router is built per call, every frame is produced by a plain
synchronous generator, and the database work belongs to the injected
bridge's director.  No network happens here -- FastAPI owns the
socket, this module only renders frames.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from nullcore.orch.bridge import BridgeError, OrchBridge
from nullcore.orch.cache_matrix import CacheStatus, CacheTier
from nullcore.orch.state_machine import OrchState
from nullcore.orch.task_envelope import (
    StageVector,
    TaskEnvelope,
    TaskEnvelopeError,
)

__all__ = ["build_stream_router"]


def _frame(event: str, payload: dict) -> str:
    """Render one SSE frame: an event line, a data line, a blank line."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _validation_error_stream(detail: str) -> Iterator[str]:
    """Yield the stream of a request that never starts.

    One ``error`` frame carrying the detail, then the ``done``
    terminator that ends every stream this module produces.
    """
    yield _frame("error", {"error": detail})
    yield _frame("done", {})


def _error_response(detail: str) -> StreamingResponse:
    """Build the two-frame event stream for a validation failure."""
    return StreamingResponse(
        _validation_error_stream(detail), media_type="text/event-stream"
    )


def build_stream_router(bridge: OrchBridge) -> APIRouter:
    """Close one OrchBridge into a fresh APIRouter with one stream route.

    Raises:
        BridgeError: when ``bridge`` is not an
            :class:`~nullcore.orch.bridge.OrchBridge`.
    """
    if not isinstance(bridge, OrchBridge):
        raise BridgeError(
            f"bridge must be an OrchBridge, got {type(bridge).__name__}"
        )
    router = APIRouter()

    @router.get("/api/stream/{task_id}")
    def stream_task(
        task_id: str,
        stage: str,
        seat: str,
        attempt: int = 0,
        config_hash: str = Query(...),
        request_b64: str = Query(...),
        lifecycle_state: str = Query(...),
        tier: str = Query(...),
        status: str = Query(...),
        degraded_flag: bool = False,
    ) -> StreamingResponse:
        """Stream one task-stage attempt as Server-Sent Events."""
        # 1. Payload decoding: strict base64 only, exactly like the
        #    M1e submit door.  A failure never starts a stream.
        try:
            request_bytes = base64.b64decode(request_b64, validate=True)
        except ValueError as exc:
            return _error_response(f"request_b64 is not valid base64: {exc}")
        # 2. Enum mapping: wire strings onto the M1a/M1b members
        #    through their value constructors.
        try:
            mapped_lifecycle = OrchState(lifecycle_state)
        except (KeyError, ValueError):
            return _error_response(
                f"lifecycle_state is not a valid OrchState: "
                f"{lifecycle_state!r}"
            )
        try:
            mapped_tier = CacheTier(tier)
        except (KeyError, ValueError):
            return _error_response(
                f"tier is not a valid CacheTier: {tier!r}"
            )
        try:
            mapped_status = CacheStatus(status)
        except (KeyError, ValueError):
            return _error_response(
                f"status is not a valid CacheStatus: {status!r}"
            )
        # 3. Envelope construction: the M1d validation messages ride
        #    into the error frame unchanged so the caller sees the
        #    offending field.  The credential stage vector is empty --
        #    a query string cannot carry prior verdicts, and none are
        #    invented here.
        try:
            envelope = TaskEnvelope(
                task_id=task_id,
                stage=stage,
                seat=seat,
                attempt=attempt,
                config_hash=config_hash,
                request_bytes=request_bytes,
                lifecycle_state=mapped_lifecycle,
                stage_vector={},
                tier=mapped_tier,
                status=mapped_status,
                degraded_flag=degraded_flag,
            )
        except TaskEnvelopeError as exc:
            return _error_response(str(exc))
        # 4. The recorder the stream reports on, fresh per request.
        vector = StageVector()

        # 5. The event stream: start, submit, verdict-or-error, done.
        def event_generator() -> Iterator[str]:
            yield _frame("start", dict(vector.snapshot()))
            try:
                _response, from_replay = bridge.submit(envelope, vector)
            except Exception as exc:
                # The bridge has already recorded the S6 FAILED
                # verdict with the exception type as the note; the
                # frame reports that final state and the type name.
                yield _frame(
                    "error",
                    {
                        "stage_vector": dict(vector.snapshot()),
                        "notes": dict(vector.notes()),
                        "error": type(exc).__name__,
                    },
                )
            else:
                yield _frame(
                    "verdict",
                    {
                        "stage_vector": dict(vector.snapshot()),
                        "notes": dict(vector.notes()),
                        "from_replay": from_replay,
                    },
                )
            yield _frame("done", {})

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )

    return router
