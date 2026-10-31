"""Framework-free bridge from task envelopes to the ORCH director.

:class:`OrchBridge` is the single public entry point callers need: it
owns a registry of seat executors, validates submitted envelopes, and
delegates the entire attempt lifecycle to the M1c
:class:`~nullcore.orch.director.OrchDirector`.  The bridge adds no
policy of its own -- no retries, no I/O, no clock, no network -- it
only routes the call to the executor registered for the envelope's
seat, records the ``S6`` verdict on the caller's
:class:`~nullcore.orch.task_envelope.StageVector`, and propagates
every exception from the director unchanged after recording.

The recording contract is fixed:

* a malformed ``envelope`` or ``vector`` argument fails fast with
  :class:`BridgeError` and nothing is recorded;
* an unknown seat fails fast with :class:`BridgeError` naming the
  seat, and the stage vector is left untouched;
* any exception raised by the director (credential denial, lease
  contention, executor failure, cache gate, ...) records ``S6``
  ``FAILED`` with the exception type name as the note, then re-raises
  the original exception unchanged;
* a successful attempt records ``S6`` ``PASSED`` with the note
  ``"replay"`` or ``"executed"`` depending on how the director served
  the response, then returns the director's result tuple unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from nullcore.orch.director import OrchDirector
from nullcore.orch.task_envelope import StageVector, TaskEnvelope

__all__ = ["BridgeError", "OrchBridge"]


class BridgeError(Exception):
    """Raised when bridge inputs or the executor registry are invalid."""


class OrchBridge:
    """Route validated task envelopes to the director's fixed lifecycle."""

    def __init__(
        self,
        director: OrchDirector,
        executors: Mapping[str, Callable[[bytes], bytes]],
    ) -> None:
        """Validate the director and the executor registry, then freeze it.

        The registry is copied into a private read-only mapping, so
        later mutation of the caller's mapping cannot change which
        executor a submit will use.

        Raises:
            BridgeError: when ``director`` is not an
                :class:`~nullcore.orch.director.OrchDirector`, when
                ``executors`` is not a mapping, when any key is not a
                non-empty str, or when any value is not callable.
        """
        if not isinstance(director, OrchDirector):
            raise BridgeError(
                "director must be an OrchDirector, "
                f"got {type(director).__name__}"
            )
        if not isinstance(executors, Mapping):
            raise BridgeError(
                f"executors must be a Mapping, got {type(executors).__name__}"
            )
        registry: dict[str, Callable[[bytes], bytes]] = {}
        for seat, executor in executors.items():
            if not isinstance(seat, str) or not seat:
                raise BridgeError(
                    "executor keys must be non-empty str seat names, "
                    f"got {seat!r}"
                )
            if not callable(executor):
                raise BridgeError(
                    f"executor for seat {seat!r} must be callable, "
                    f"got {type(executor).__name__}"
                )
            registry[seat] = executor
        self._director = director
        self._executors: Mapping[str, Callable[[bytes], bytes]] = (
            MappingProxyType(registry)
        )

    def seats(self) -> tuple[str, ...]:
        """Return the registered seat names as a sorted tuple."""
        return tuple(sorted(self._executors))

    def submit(
        self, envelope: TaskEnvelope, vector: StageVector
    ) -> tuple[bytes, bool]:
        """Run one envelope through the director and record the S6 verdict.

        Returns the director's ``(response_bytes, from_replay)``
        tuple unchanged on success.

        Raises:
            BridgeError: when ``envelope`` is not a
                :class:`~nullcore.orch.task_envelope.TaskEnvelope` or
                ``vector`` is not a
                :class:`~nullcore.orch.task_envelope.StageVector`, or
                when no executor is registered for
                ``envelope.seat``; in every one of these cases the
                vector is left untouched.
            Exception: whatever the director raises, propagated
                unchanged after ``S6`` is recorded ``FAILED`` with the
                exception type name as the note.
        """
        if not isinstance(envelope, TaskEnvelope):
            raise BridgeError(
                "envelope must be a TaskEnvelope, "
                f"got {type(envelope).__name__}"
            )
        if not isinstance(vector, StageVector):
            raise BridgeError(
                f"vector must be a StageVector, got {type(vector).__name__}"
            )
        executor = self._executors.get(envelope.seat)
        if executor is None:
            raise BridgeError(
                f"no executor registered for seat {envelope.seat!r}"
            )
        try:
            response, from_replay = self._director.run_attempt(
                envelope.task_id,
                envelope.stage,
                envelope.seat,
                envelope.attempt,
                envelope.lifecycle_state,
                envelope.stage_vector,
                envelope.tier,
                envelope.status,
                envelope.degraded_flag,
                envelope.config_hash,
                envelope.request_bytes,
                executor,
            )
        except Exception as exc:
            vector.record("S6", False, type(exc).__name__)
            raise
        vector.record("S6", True, "replay" if from_replay else "executed")
        return response, from_replay
