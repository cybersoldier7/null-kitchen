"""Task envelope and stage vector for the NULL Orchestrator Core.

A :class:`TaskEnvelope` is the immutable, fully validated value object
that names exactly one attempt of one task stage: the call identity
(``task_id``, ``stage``, ``seat``, ``attempt``), the config binding
(``config_hash``), the payload (``request_bytes``), and the credential
material (``lifecycle_state``, ``stage_vector``, ``tier``, ``status``,
``degraded_flag``) that the M1c director gates on.  Every field is
checked in ``__post_init__``, so an envelope that exists at all is an
envelope whose identity and credential fields the director will accept
at its input-validation step.

A :class:`StageVector` is the mutable, insertion-ordered recorder a
caller threads through a bridge submit: every stage verdict is stored
as ``PASSED`` or ``FAILED`` with a note, and ``snapshot`` / ``notes``
hand out immutable point-in-time copies.  The module is pure: no I/O,
no clock, no global state, and the only imports are the stdlib plus
the frozen M1a/M1b enums.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from nullcore.orch.cache_matrix import CacheStatus, CacheTier
from nullcore.orch.state_machine import OrchState

__all__ = ["TaskEnvelopeError", "TaskEnvelope", "StageVector"]


class TaskEnvelopeError(Exception):
    """Raised when a task envelope or stage vector input is invalid."""


def _require_non_empty_str(value: object, name: str) -> None:
    """Raise TaskEnvelopeError unless value is a non-empty str."""
    if not isinstance(value, str):
        raise TaskEnvelopeError(
            f"{name} must be a str, got {type(value).__name__}"
        )
    if not value:
        raise TaskEnvelopeError(f"{name} must be a non-empty str")


def _require_attempt(value: object, name: str) -> None:
    """Raise TaskEnvelopeError unless value is an int >= 0 (bool rejected)."""
    if isinstance(value, bool):
        raise TaskEnvelopeError(f"{name} must be an int, got bool")
    if not isinstance(value, int):
        raise TaskEnvelopeError(
            f"{name} must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise TaskEnvelopeError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class TaskEnvelope:
    """Immutable, fully validated description of one task-stage attempt."""

    task_id: str
    stage: str
    seat: str
    attempt: int
    config_hash: str
    request_bytes: bytes
    lifecycle_state: OrchState
    stage_vector: Mapping[str, str]
    tier: CacheTier
    status: CacheStatus
    degraded_flag: bool

    def __post_init__(self) -> None:
        """Validate every field, then freeze the stage vector.

        Raises:
            TaskEnvelopeError: naming the offending field, when any
                string field is not a non-empty str, when ``attempt``
                is not an int >= 0 (bool rejected), when
                ``request_bytes`` is not bytes, when ``lifecycle_state``,
                ``tier``, or ``status`` is not a true member of its
                enum, when ``degraded_flag`` is not a bool, or when
                ``stage_vector`` is not a mapping of non-empty str keys
                to str values.
        """
        _require_non_empty_str(self.task_id, "task_id")
        _require_non_empty_str(self.stage, "stage")
        _require_non_empty_str(self.seat, "seat")
        _require_attempt(self.attempt, "attempt")
        _require_non_empty_str(self.config_hash, "config_hash")
        if not isinstance(self.request_bytes, bytes):
            raise TaskEnvelopeError(
                "request_bytes must be bytes, "
                f"got {type(self.request_bytes).__name__}"
            )
        if not isinstance(self.lifecycle_state, OrchState):
            raise TaskEnvelopeError(
                "lifecycle_state must be an OrchState member, "
                f"got {type(self.lifecycle_state).__name__}"
            )
        if not isinstance(self.stage_vector, Mapping):
            raise TaskEnvelopeError(
                "stage_vector must be a Mapping, "
                f"got {type(self.stage_vector).__name__}"
            )
        for key, value in self.stage_vector.items():
            if not isinstance(key, str) or not key:
                raise TaskEnvelopeError(
                    f"stage_vector keys must be non-empty str, got {key!r}"
                )
            if not isinstance(value, str):
                raise TaskEnvelopeError(
                    f"stage_vector[{key!r}] must be a str, "
                    f"got {type(value).__name__}"
                )
        if not isinstance(self.tier, CacheTier):
            raise TaskEnvelopeError(
                f"tier must be a CacheTier member, got {type(self.tier).__name__}"
            )
        if not isinstance(self.status, CacheStatus):
            raise TaskEnvelopeError(
                "status must be a CacheStatus member, "
                f"got {type(self.status).__name__}"
            )
        if not isinstance(self.degraded_flag, bool):
            raise TaskEnvelopeError(
                "degraded_flag must be a bool, "
                f"got {type(self.degraded_flag).__name__}"
            )
        # Freeze the credential evidence: the caller's mapping is
        # copied and wrapped read-only, so later mutation of the
        # original cannot drift an already-validated envelope.
        object.__setattr__(
            self, "stage_vector", MappingProxyType(dict(self.stage_vector))
        )

    def with_attempt(self, attempt: int) -> TaskEnvelope:
        """Return a validated copy of this envelope with a new attempt.

        Every other field is carried over unchanged, and the copy is
        re-validated end to end, so the returned envelope satisfies
        the same construction guarantees as a freshly built one.

        Raises:
            TaskEnvelopeError: when ``attempt`` is not an int >= 0
                (bool rejected).
        """
        _require_attempt(attempt, "attempt")
        return replace(self, attempt=attempt)


class StageVector:
    """Insertion-ordered recorder of per-stage verdicts and notes."""

    def __init__(self) -> None:
        """Start empty; both views stay safe when nothing is recorded."""
        self._results: dict[str, str] = {}
        self._notes: dict[str, str] = {}

    def record(self, stage_id: str, passed: bool, note: str = "") -> None:
        """Store PASSED or FAILED for ``stage_id``, keeping record order.

        Recording the same ``stage_id`` again overwrites both its
        verdict and its note in place without changing its position.

        Raises:
            TaskEnvelopeError: when ``stage_id`` is not a non-empty
                str, when ``passed`` is not a bool, or when ``note``
                is not a str.
        """
        _require_non_empty_str(stage_id, "stage_id")
        if not isinstance(passed, bool):
            raise TaskEnvelopeError(
                f"passed must be a bool, got {type(passed).__name__}"
            )
        if not isinstance(note, str):
            raise TaskEnvelopeError(
                f"note must be a str, got {type(note).__name__}"
            )
        self._results[stage_id] = "PASSED" if passed else "FAILED"
        self._notes[stage_id] = note

    def snapshot(self) -> Mapping[str, str]:
        """Return an immutable copy of the verdicts, in record order."""
        return MappingProxyType(dict(self._results))

    def notes(self) -> Mapping[str, str]:
        """Return an immutable copy of the notes, in record order."""
        return MappingProxyType(dict(self._notes))
