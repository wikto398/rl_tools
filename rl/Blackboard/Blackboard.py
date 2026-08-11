from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlackboardEvent:
    """A single metric record stored on the blackboard.

    ``id`` is a monotonically increasing global sequence number, so cursors
    can point at an event id without being invalidated when older events are
    purged.
    """

    id: int
    step: int
    kind: str  # "scalar" | "histogram"
    key: str
    value: Any


class Blackboard:
    """Shared event log + key-value store for inter-callback communication.

    Producers append metric records via ``record`` / ``record_histogram`` and
    share arbitrary state via ``set`` / ``get``. Sink callbacks register a
    cursor and drain only the events they have not seen yet.

    Auto-clearing: once *every* registered cursor has passed an event id, the
    event is no longer needed by anyone and is purged. Purging runs lazily in
    batches of ``CLEAR_BATCH_SIZE`` events so the log stays bounded without
    churning on every record.
    """

    CLEAR_BATCH_SIZE = 4096

    def __init__(self) -> None:
        self._events: deque[BlackboardEvent] = deque()
        self._next_id = 0
        self._cursors: dict[str, int] = {}
        self._last_purged = 0
        self._data: dict[str, Any] = {}

    # -- Producers ---------------------------------------------------------

    def record(self, key: str, value: float, step: int) -> None:
        self._events.append(
            BlackboardEvent(
                id=self._next_id,
                step=int(step),
                kind="scalar",
                key=key,
                value=float(value),
            )
        )
        self._next_id += 1

    def record_histogram(self, key: str, values: Any, step: int) -> None:
        self._events.append(
            BlackboardEvent(
                id=self._next_id,
                step=int(step),
                kind="histogram",
                key=key,
                value=values,
            )
        )
        self._next_id += 1

    # -- Shared state (behavior-tree style) --------------------------------

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # -- Consumers ---------------------------------------------------------

    def register_cursor(self, name: str) -> None:
        """Register a sink cursor at the current end of the log.

        Late-registered cursors only see events recorded after registration.
        """
        if name in self._cursors:
            raise ValueError(f"Cursor {name!r} is already registered")
        self._cursors[name] = self._next_id

    def drain(self, name: str) -> list[BlackboardEvent]:
        """Return new events for a cursor and advance it to the log end."""
        cursor = self._cursors.get(name)
        if cursor is None:
            raise KeyError(
                f"Cursor {name!r} is not registered. "
                "Sinks must call blackboard.register_cursor(name) first."
            )
        if cursor < self._last_purged:
            cursor = self._last_purged
        events = [e for e in self._events if e.id >= cursor]
        self._cursors[name] = self._next_id
        self._maybe_clear()
        return events

    def unregister_cursor(self, name: str) -> None:
        self._cursors.pop(name, None)

    def event_count(self) -> int:
        return len(self._events)

    # -- Auto-clear --------------------------------------------------------

    def _maybe_clear(self) -> None:
        if not self._cursors:
            return
        min_cursor = min(self._cursors.values())
        if min_cursor - self._last_purged < self.CLEAR_BATCH_SIZE:
            return
        while self._events and self._events[0].id < min_cursor:
            self._events.popleft()
        self._last_purged = min_cursor

    # -- Lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self._events.clear()
        self._cursors.clear()
        self._last_purged = 0
        self._data.clear()
        self._next_id = 0
