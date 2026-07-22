from __future__ import annotations

import select
import sys
import threading

from rl_tools.rl.Callback.StopTrainingCallback import StopTrainingCallback


class KeyStopCallback(StopTrainingCallback):
    """Stop training when a configured key is entered on stdin."""

    def __init__(self, key: str = "q") -> None:
        super().__init__()
        if not key:
            raise ValueError("key must be a non-empty string")
        self.key = key.strip().lower()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def on_train_start(self) -> None:
        if not sys.stdin.isatty():
            if self.agent is not None:
                self.agent.info(
                    "KeyStopCallback: stdin is not a TTY — key stop disabled"
                )
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen,
            name="KeyStopCallback",
            daemon=True,
        )
        self._thread.start()
        if self.agent is not None:
            self.agent.info(
                f"KeyStopCallback: type '{self.key}' + Enter to stop training"
            )

    def on_train_end(self) -> None:
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
