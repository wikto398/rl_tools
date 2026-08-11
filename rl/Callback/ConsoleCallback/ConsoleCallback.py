from __future__ import annotations

from typing import Any

from rl_tools.rl.Callback.SinkCallback import SinkCallback


class ConsoleCallback(SinkCallback):
    """Print blackboard scalars to the agent logger at INFO level."""

    def _write_scalar(self, step: int, key: str, value: float) -> None:
        if self.agent is None:
            return
        self.agent.info(f"Step {step}: {key} = {value}")

    def _write_histogram(self, step: int, key: str, values: Any) -> None:
        pass
