from __future__ import annotations


from rl_tools.rl.Callback.StopTrainingCallback import StopTrainingCallback


class GateStopCallback(StopTrainingCallback):
    """One-shot go/no-go gate evaluated at a specific step.

    Once ``global_step`` reaches ``check_step``, reads the metric from the
    blackboard's ``eval/latest`` dict. If the metric is on the wrong side of
    ``threshold`` the run is stopped; otherwise training continues untouched.
    Either way the callback detaches itself and never fires again.
    """

    def __init__(
        self,
        check_step: int,
        metric: str = "win_rate",
        *,
        threshold: float = 0.05,
        goal: str = "below",
    ) -> None:
        super().__init__()
        if check_step <= 0:
            raise ValueError(f"check_step must be positive, got {check_step}")
        if goal not in ("below", "above"):
            raise ValueError(f"goal must be 'below' or 'above', got {goal!r}")
        self.check_step = int(check_step)
        self.metric = metric
        self.threshold = float(threshold)
        self.goal = goal
        self._fired = False

    def on_update_end(self, update_info: dict) -> None:
        if self.agent is None or self._fired:
            return
        if self.agent.global_step < self.check_step:
            return
        self._fired = True
        try:
            value = self.agent.blackboard.get("eval/latest", {}).get(self.metric)
            if value is None:
                return
            if self.goal == "below":
                bad = float(value) < self.threshold
            else:
                bad = float(value) > self.threshold
            if bad:
                self.request_stop(
                    f"{self.metric} {self.goal} {self.threshold} at step "
                    f"{self.check_step}"
                )
        finally:
            self.detach()
