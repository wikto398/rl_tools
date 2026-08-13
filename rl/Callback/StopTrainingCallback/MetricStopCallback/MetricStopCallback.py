from __future__ import annotations


from rl_tools.rl.Callback.StopTrainingCallback import StopTrainingCallback


class MetricStopCallback(StopTrainingCallback):
    """Stop training when a metric stays on the wrong side of a threshold.

    Reads the metric from the blackboard's ``eval/latest`` dict (published by
    ``EvalCallback``) and stops once the metric has been bad for ``patience``
    consecutive evals. If the metric recovers, the counter resets so improving
    runs are not killed. Fires on every new eval until stopped.
    """

    def __init__(
        self,
        metric: str = "win_rate",
        *,
        threshold: float = 0.05,
        patience: int = 3,
        goal: str = "below",
    ) -> None:
        super().__init__()
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if goal not in ("below", "above"):
            raise ValueError(f"goal must be 'below' or 'above', got {goal!r}")
        self.metric = metric
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.goal = goal
        self._bad_count = 0
        self._last_eval_step = -1

    def on_update_end(self, update_info: dict) -> None:
        if self.agent is None:
            return
        blackboard = self.agent.blackboard
        eval_step = blackboard.get("eval/latest_step", -1)
        if eval_step <= self._last_eval_step:
            return
        self._last_eval_step = int(eval_step)
        value = blackboard.get("eval/latest", {}).get(self.metric)
        if value is None:
            return

        if self.goal == "below":
            bad = float(value) < self.threshold
        else:
            bad = float(value) > self.threshold

        if bad:
            self._bad_count += 1
        else:
            self._bad_count = 0
            return

        if self._bad_count >= self.patience:
            self.request_stop(
                f"{self.metric} {self.goal} {self.threshold} for "
                f"{self.patience} consecutive evals"
            )
