from __future__ import annotations
import hashlib
from typing import Final

__all__ = ["CacheKeyError", "make_cache_key"]
class CacheKeyError(Exception): pass
_FIELD_SEPARATOR: Final[str] = "\x1f"

def _require_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value: raise CacheKeyError(f"{name} must be a non-empty str")

def make_cache_key(config_hash: str, task_id: str, stage: str, seat: str, attempt: int) -> str:
    _require_non_empty_str(config_hash, "config_hash"); _require_non_empty_str(task_id, "task_id")
    _require_non_empty_str(stage, "stage"); _require_non_empty_str(seat, "seat")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0: raise CacheKeyError("attempt must be an int >= 0")
    payload = config_hash + _FIELD_SEPARATOR + task_id + _FIELD_SEPARATOR + stage + _FIELD_SEPARATOR + seat + _FIELD_SEPARATOR + str(attempt)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()