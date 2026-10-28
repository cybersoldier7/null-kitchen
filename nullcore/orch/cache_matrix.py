from __future__ import annotations
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final
from nullcore.orch.state_machine import OrchState

__all__ = ["CacheTier", "CacheStatus", "MatrixViolationError", "WRITE_MATRIX", "can_write", "write_requires_degraded_flag", "stage_passed", "verify_credential"]

class CacheTier(str, Enum): T0="T0"; T1="T1"; T2="T2"; T3="T3"
class CacheStatus(str, Enum): TERMINAL_VERIFIED="TERMINAL_VERIFIED"; PROVISIONAL="PROVISIONAL"; DEGRADED="DEGRADED"
class MatrixViolationError(Exception): pass

WRITE_MATRIX: Final[Mapping[CacheStatus, frozenset[CacheTier]]] = MappingProxyType({
    CacheStatus.TERMINAL_VERIFIED: frozenset({CacheTier.T0, CacheTier.T1, CacheTier.T2, CacheTier.T3}),
    CacheStatus.PROVISIONAL: frozenset({CacheTier.T2}),
    CacheStatus.DEGRADED: frozenset({CacheTier.T0}),
})

def can_write(status: CacheStatus, tier: CacheTier) -> bool:
    if not isinstance(status, CacheStatus) or not isinstance(tier, CacheTier): return False
    return tier in WRITE_MATRIX[status]

def write_requires_degraded_flag(status: CacheStatus, tier: CacheTier) -> bool:
    return status is CacheStatus.DEGRADED and tier is CacheTier.T0

def stage_passed(stage_vector: Mapping[str, str], stage_id: str) -> bool:
    if stage_vector is None: return False
    return stage_vector.get(stage_id) == "PASSED"

def verify_credential(lifecycle_state: OrchState, stage_vector: Mapping[str, str], claimed: CacheStatus) -> CacheStatus:
    if not isinstance(lifecycle_state, OrchState): raise MatrixViolationError(f"lifecycle_state must be OrchState, got {type(lifecycle_state).__name__}")
    if lifecycle_state in (OrchState.RECALLED, OrchState.UPGRADE_FAILED): raise MatrixViolationError("never write cache")
    if claimed is CacheStatus.TERMINAL_VERIFIED:
        if lifecycle_state is not OrchState.VERIFIED: raise MatrixViolationError("requires VERIFIED")
        for stage_id in ("S5", "S6", "S7"):
            if not stage_passed(stage_vector, stage_id): raise MatrixViolationError(f"requires {stage_id} PASSED")
    elif claimed is CacheStatus.PROVISIONAL:
        if lifecycle_state not in (OrchState.PROVISIONAL, OrchState.GATING_ASYNC, OrchState.VERIFIED): raise MatrixViolationError("requires >= PROVISIONAL")
    elif claimed is not CacheStatus.DEGRADED:
        raise MatrixViolationError("invalid claimed status")
    return claimed