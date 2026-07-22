from __future__ import annotations

from tensordict import TensorDict

from rl_tools.rl.Callback.StopTrainingCallback import StopTrainingCallback


class MaxStepsStopCallback(StopTrainingCallback):
    """Stop training after a fixed number of environment steps."""

    def __init__(self, max_steps: int) -> None:
        super().__init__()
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.max_steps = max_steps
        self._steps = 0

    def on_rollout_end(self, rollout: TensorDict) -> None:
        self._steps += int(rollout["rewards"].numel())
        if self._steps >= self.max_steps:
            self.request_stop(f"reached {self._steps}/{self.max_steps} steps")
