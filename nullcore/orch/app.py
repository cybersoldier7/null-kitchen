from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI

from nullcore.orch.wiring import build_services

app = FastAPI(title="NULL ORCH", version="m3-mount")

_BOOT_EPOCH_MS: int = time.time_ns() // 1_000_000

# M3 state-backend ruling (Option A): the stack runs on a private
# in-memory database.  Render's filesystem is ephemeral, so a
# file-backed path would reset on every deploy anyway; the reset
# boundary is therefore the process lifetime, documented here and
# exposed via /api/status.  ADR-M3: Wave 3 swaps this single argument
# for the Option D snapshot sink (certifiable SnapshotSink core,
# LeaseManager writer election, replication-lag lamp); Option C
# (git-backed state through the cert repo) is rejected permanently.
_SERVICES: dict = build_services(":memory:")

app.include_router(_SERVICES["router"])


@app.get("/api/muster")
def muster() -> dict[str, Any]:
    epoch_ms: int = time.time_ns() // 1_000_000
    return {"status": "alive", "service": "null-orch", "epoch_ms": epoch_ms}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "stage_vector": [],
        "service": "null-orch",
        "errand": 3,
        "boot_epoch_ms": _BOOT_EPOCH_MS,
        "state_backend": "ephemeral-memory",
        "seats": list(_SERVICES["bridge"].seats()),
    }
