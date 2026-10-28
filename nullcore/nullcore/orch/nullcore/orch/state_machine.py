from __future__ import annotations
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final

__all__ = ["OrchState", "IllegalStateTransition", "LEGAL_TRANSITIONS", "can_transition", "transition"]

class OrchState(str, Enum):
    DRAFTING = "DRAFTING"; PROVISIONAL = "PROVISIONAL"; GATING_ASYNC = "GATING_ASYNC"
    VERIFIED = "VERIFIED"; RECALLED = "RECALLED"; UPGRADE_FAILED = "UPGRADE_FAILED"

class IllegalStateTransition(Exception):
    def __init__(self, current: OrchState, desired: OrchState) -> None:
        self.current = current
        self.desired = desired
        super().__init__(f"illegal state transition: {current.value} -> {desired.value}")

LEGAL_TRANSITIONS: Final[Mapping[OrchState, frozenset[OrchState]]] = MappingProxyType({
    OrchState.DRAFTING: frozenset({OrchState.PROVISIONAL}),
    OrchState.PROVISIONAL: frozenset({OrchState.GATING_ASYNC}),
    OrchState.GATING_ASYNC: frozenset({OrchState.VERIFIED, OrchState.RECALLED, OrchState.UPGRADE_FAILED}),
    OrchState.VERIFIED: frozenset(),
    OrchState.RECALLED: frozenset(),
    OrchState.UPGRADE_FAILED: frozenset({OrchState.RECALLED}),
})

def can_transition(current: OrchState, desired: OrchState) -> bool:
    if not isinstance(current, OrchState) or not isinstance(desired, OrchState): return False
    return desired in LEGAL_TRANSITIONS[current]

def transition(current: OrchState, desired: OrchState) -> OrchState:
    if not isinstance(current, OrchState): raise TypeError("current must be OrchState")
    if not isinstance(desired, OrchState): raise TypeError("desired must be OrchState")
    if desired not in LEGAL_TRANSITIONS[current]: raise IllegalStateTransition(current, desired)
    return desired