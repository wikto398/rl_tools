from __future__ import annotations

import logging
import select
import sys
import threading

from rl_tools.rl.Callback.Callback import NoOpCallback

logger = logging.getLogger("Evolution")


class StopEvolutionCallback(NoOpCallback):
    """Can request a graceful stop of the evolution at a generation boundary.

    Mirrors ``StopTrainingCallback``: ``request_stop`` sets a stop flag on the
    orchestrator, which ``evolve()`` checks between generations. Optionally
    listens on stdin (mirroring ``KeyStopCallback``) so a configured key can
    trigger the stop from a foreground terminal; the listener is auto-disabled
    with a log line when stdin is not a TTY.
    """

    def __init__(self, key: str = "p") -> None:
        super().__init__()
        self.key = key.strip().lower()
        self._stopped = False
        self.orchestrator = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def request_stop(self, reason: str | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.orchestrator is not None:
            self.orchestrator._stop_requested = True
        message = (
            "StopEvolutionCallback: stopping evolution after the current generation"
        )
        if reason:
            message = f"{message} ({reason})"
        logger.info(message)

    def start(self) -> None:
        if not sys.stdin.isatty():
            logger.info("StopEvolutionCallback: stdin is not a TTY — key stop disabled")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen,
            name="StopEvolutionCallback",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"StopEvolutionCallback: type '{self.key}' + Enter to stop evolution"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listen(self) -> None:
        while not self._stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                break
            if line.strip().lower() == self.key:
                self.request_stop(f"key '{self.key}'")
                break
