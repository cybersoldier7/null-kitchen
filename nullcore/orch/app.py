from __future__ import annotations

   import os
   import platform
   import shutil
   import subprocess
   import sys
   import time
   from typing import Any

   from fastapi import FastAPI

   app = FastAPI(title="NULL ORCH", version="errand2-probe")


   @app.get("/api/muster")
   def muster() -> dict[str, Any]:
       epoch_ms: int = time.time_ns() // 1_000_000
       return {"status": "alive", "service": "null-orch", "epoch_ms": epoch_ms}


   @app.get("/api/status")
   def status() -> dict[str, Any]:
       return {"stage_vector": [], "service": "null-orch", "errand": 1}


   @app.get("/api/probe")
   def probe() -> dict[str, Any]:
       info: dict[str, Any] = {
           "python": sys.version.split()[0],
           "platform": platform.platform(),
           "machine": platform.machine(),
           "uid": os.getuid(),
           "gid": os.getgid(),
           "nsjail": shutil.which("nsjail"),
           "bwrap": shutil.which("bwrap"),
           "unshare": shutil.which("unshare"),
       }
       try:
           with open("/etc/os-release", encoding="utf-8") as handle:
               info["os_release"] = handle.read().splitlines()[:3]
       except OSError:
           info["os_release"] = None
       try:
           done = subprocess.run(
               ["unshare", "-Urm", "true"], capture_output=True, timeout=5
           )
           info["userns"] = "OK" if done.returncode == 0 else f"BLOCKED:{done.returncode}"
       except Exception as exc:
           info["userns"] = f"ERROR:{type(exc).__name__}"
       return info
