from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback


class FitnessCallback(Callback):
    """Export the final eval fitness of an evolutionary child.

    On ``on_train_end`` reads the blackboard's ``eval/latest`` dict (published
    by ``EvalCallback``) and writes it to ``out_path`` as JSON. The parent
    process (the evolution orchestrator) reads these files after each child
    process exits, because the blackboard itself dies with the child.
    """

    def __init__(self, out_path: str):
        super().__init__()
        self.out_path = out_path

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        if self.agent is None:
            return
        blackboard = self.agent.blackboard
        latest = blackboard.get("eval/latest", {}) or {}
        fitness = {
            "win_rate": float(latest.get("win_rate", 0.0)),
            "mean_win_turns": latest.get("mean_win_turns"),
            "mean_loss_turns": latest.get("mean_loss_turns"),
            "mean_return": latest.get("mean_return"),
            "mean_length": latest.get("mean_length"),
            "buildings_started": float(latest.get("buildings_started", 0.0)),
            "buildings_completed": float(latest.get("buildings_completed", 0.0)),
            "eval_step": blackboard.get("eval/latest_step"),
            "global_step": int(getattr(self.agent, "global_step", 0)),
            "stopped_early": bool(getattr(self.agent, "_stop_requested", False)),
            "errored": int(getattr(self.agent, "global_step", 0)) == 0,
        }
        try:
            with open(self.out_path, "w") as f:
                json.dump(fitness, f, indent=2)
        except OSError as e:
            if self.agent is not None:
                self.agent.error(
                    f"FitnessCallback: failed to write {self.out_path}: {e}"
                )

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self, rollout: TensorDict) -> None:
        pass

    def on_step(
        self,
        *,
        actions: Any,
        rewards: Sequence[float],
        dones: Sequence[bool],
        infos: Sequence[dict],
    ) -> bool:
        return True

    def on_update_start(self, rollout: TensorDict) -> None:
        pass

    def on_update_end(self, update_info: dict) -> None:
        pass
