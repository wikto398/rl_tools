import torch
import numpy as np
from abc import ABC, abstractmethod

from rl.Environment.Environment import Environment
from game_engine.ObservationNormalizer.ObservationNormalizer import (
    ObservationNormalizer,
)
from game_engine.RewardNormalizer.RewardNormalizer import RewardNormalizer


class RLAgent(ABC):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        envs: list[Environment] | Environment,
        reward_normalizer: RewardNormalizer | None = None,
        observation_normalizer: ObservationNormalizer | None = None,
        device: torch.device | None = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        *args,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = network.to(self.device)
        self.gamma = gamma
        self.lam = lam
        self.observation_normalizer = observation_normalizer
        self.reward_normalizer = reward_normalizer
        self.envs = envs if isinstance(envs, list) else [envs]
        self.args = args
        self.kwargs = kwargs

        self._initialize_obs()

    @abstractmethod
    def train(self, iterations: int, steps: int, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> tuple:
        pass

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def set_eval_mode(self, eval_mode: bool):
        if eval_mode:
            self.network.eval()
        else:
            self.network.train()

    def _initialize_obs(self):
        obs_list = [env.reset() for env in self.envs]
        self.obs = {
            "observation": np.array([o["observation"] for o in obs_list]),
            "action_mask": np.array(
                [o.get("action_mask", np.zeros(0)) for o in obs_list]
            ),
        }
