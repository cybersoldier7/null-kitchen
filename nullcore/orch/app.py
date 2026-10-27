from __future__ import annotations
import time
from typing import Any
from fastapi import FastAPI

   app = FastAPI(title="NULL ORCH", version="errand1")

   @app.get("/api/muster")
   def muster() -> dict[str, Any]:
       epoch_ms: int = time.time_ns() // 1_000_000
       return {"status": "alive", "service": "null-orch", "epoch_ms": epoch_ms}

   @app.get("/api/status")
   def status() -> dict[str, Any]:
       return {"stage_vector": [], "service": "null-orch", "errand": 1}
