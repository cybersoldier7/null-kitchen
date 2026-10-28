from __future__ import annotations
import hashlib
from typing import Final

__all__ = ["IdempotencyKeyError", "make_idempotency_key"]
class IdempotencyKeyError(Exception): pass
_FIELD_SEPARATOR: Final[str] = "\x1f"

def _require_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value: raise IdempotencyKeyError(f"{name} must be a non-empty str")

def make_idempotency_key(task_id: str, stage: str, seat: str, attempt: int) -> str:
    _require_non_empty_str(task_id, "task_id"); _require_non_empty_str(stage, "stage"); _require_non_empty_str(seat, "seat")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0: raise IdempotencyKeyError("attempt must be an int >= 0")
    payload = task_id + _FIELD_SEPARATOR + stage + _FIELD_SEPARATOR + seat + _FIELD_SEPARATOR + str(attempt)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()