import wandb
from typing import Callable


class WandbWrapper:
    def __init__(
        self,
        project: str,
        *,
        rl_agent_params: dict | None = None,
        optimizer_params: dict | None = None,
        game_params: dict | None = None,
        **kwargs,
    ):
        self.project = project
        self.rl_agent_params = rl_agent_params or {}
        self.optimizer_params = optimizer_params or {}
        self.game_params = game_params or {}
        self.kwargs = kwargs

    @property
    def config(self) -> dict:
        return {**self.rl_agent_params, **self.optimizer_params, **self.game_params}

    def init(self) -> wandb.Run:
        return wandb.init(project=self.project, config=self.config, **self.kwargs)

    def sweep(self, sweep_config: dict, train_fn: Callable) -> str:
        sweep_id = wandb.sweep(sweep_config, project=self.project)
        wandb.agent(sweep_id, function=train_fn)
        return sweep_id

    def log(self, metrics: dict, step: int | None = None):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
