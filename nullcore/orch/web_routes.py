"""Web door for the ORCH bridge: FastAPI routes over the M1d bridge.

``build_router(bridge)`` closes one validated
:class:`~nullcore.orch.bridge.OrchBridge` into a fresh
:class:`fastapi.APIRouter` exposing exactly two synchronous endpoints:

* ``POST /api/task`` decodes a strict-base64 payload, maps the wire
  strings onto the M1a/M1b enums through their value constructors,
  builds a validated :class:`~nullcore.orch.task_envelope.TaskEnvelope`,
  and submits it through the bridge onto a fresh
  :class:`~nullcore.orch.task_envelope.StageVector`.  Every mapped
  rejection carries an HTTP status: malformed wire input is 400, a
  denied credential is 403, an evicted cache tombstone is 410,
  capture/cache conflicts and lease contention are 409, and any other
  director or bridge failure is 400.  Exceptions outside the map (an
  executor failure, for instance) propagate unchanged.
* ``GET /api/seats`` returns the sorted registered seat names.

The module holds no state at import time and performs no I/O: the
router is built per call, the database work belongs to the injected
bridge's director, and every request runs the full fixed M1c
lifecycle.  No network happens here -- FastAPI owns the socket, this
module only translates values and exceptions.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nullcore.orch.bridge import BridgeError, OrchBridge
from nullcore.orch.cache_matrix import CacheStatus, CacheTier, MatrixViolationError
from nullcore.orch.cache_store import CacheConflictError, CacheEvictedError
from nullcore.orch.capture import CaptureConflictError
from nullcore.orch.director import DirectorError
from nullcore.orch.state_machine import OrchState
from nullcore.orch.task_envelope import (
    StageVector,
    TaskEnvelope,
    TaskEnvelopeError,
)

__all__ = ["TaskRequest", "build_router"]


class TaskRequest(BaseModel):
    """Wire shape of one task-stage attempt submission."""

    task_id: str
    stage: str
    seat: str
    attempt: int = 0
    config_hash: str
    request_b64: str
    lifecycle_state: str
    stage_vector: dict[str, str]
    tier: str
    status: str
    degraded_flag: bool = False


def _map_enum(raw: str, enum_cls: type, field_name: str):
    """Map a wire string onto an enum member via its value constructor.

    Raises:
        HTTPException: 400 naming the field when the string is not a
            valid value of ``enum_cls``.
    """
    try:
        return enum_cls(raw)
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not a valid {enum_cls.__name__}: {raw!r}",
        )


def build_router(bridge: OrchBridge) -> APIRouter:
    """Close one OrchBridge into a fresh APIRouter with two routes.

    Raises:
        BridgeError: when ``bridge`` is not an
            :class:`~nullcore.orch.bridge.OrchBridge`.
    """
    if not isinstance(bridge, OrchBridge):
        raise BridgeError(
            f"bridge must be an OrchBridge, got {type(bridge).__name__}"
        )
    router = APIRouter()

    @router.post("/api/task")
    def submit_task(request: TaskRequest) -> dict:
        """Submit one attempt through the bridge's fixed lifecycle."""
        # 1. Payload decoding: strict base64 only.
        try:
            request_bytes = base64.b64decode(request.request_b64, validate=True)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"request_b64 is not valid base64: {exc}",
            )
        # 2. Enum mapping: wire strings onto the M1a/M1b members.
        lifecycle_state = _map_enum(
            request.lifecycle_state, OrchState, "lifecycle_state"
        )
        tier = _map_enum(request.tier, CacheTier, "tier")
        status = _map_enum(request.status, CacheStatus, "status")
        # 3. Envelope construction: the M1d validation messages pass
        #    through unchanged so the caller sees the offending field.
        try:
            envelope = TaskEnvelope(
                task_id=request.task_id,
                stage=request.stage,
                seat=request.seat,
                attempt=request.attempt,
                config_hash=request.config_hash,
                request_bytes=request_bytes,
                lifecycle_state=lifecycle_state,
                stage_vector=request.stage_vector,
                tier=tier,
                status=status,
                degraded_flag=request.degraded_flag,
            )
        except TaskEnvelopeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        # 4. Submission through the bridge onto a fresh stage vector;
        #    only the mapped exceptions become HTTP errors.
        vector = StageVector()
        try:
            response, from_replay = bridge.submit(envelope, vector)
        except MatrixViolationError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except CacheEvictedError as exc:
            raise HTTPException(status_code=410, detail=str(exc))
        except (CaptureConflictError, CacheConflictError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except DirectorError as exc:
            if "lease held" in str(exc):
                raise HTTPException(status_code=409, detail=str(exc))
            raise HTTPException(status_code=400, detail=str(exc))
        except BridgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        # 5. Success: the response rides back as strict base64 plus the
        #    recorded stage verdicts.
        return {
            "response_b64": base64.b64encode(response).decode("ascii"),
            "from_replay": from_replay,
            "stage_vector": dict(vector.snapshot()),
            "notes": dict(vector.notes()),
        }

    @router.get("/api/seats")
    def list_seats() -> dict:
        """List the registered seat names, sorted."""
        return {"seats": list(bridge.seats())}

    return router
