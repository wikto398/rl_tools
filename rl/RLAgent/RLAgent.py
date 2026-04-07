import logging

import torch
import wandb
import numpy as np
from abc import ABC, abstractmethod
from tensordict import TensorDict

from rl_tools.rl.Environment import Environment
from rl_tools.game_engine.ObservationNormalizer import (
    ObservationNormalizer,
)
from rl_tools.game_engine.RewardNormalizer import RewardNormalizer


class RLAgent(ABC):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        envs: list[Environment] | Environment,
        reward_normalizer: RewardNormalizer | None = None,
        observation_normalizer: ObservationNormalizer | None = None,
        device: torch.device | None = None,
        wandb: wandb.Run | None = None,
        logger: logging.Logger | None = None,
        *args,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 5,
        batch_size: int = 64,
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.observation_normalizer = observation_normalizer
        self.reward_normalizer = reward_normalizer
        self.envs = envs if isinstance(envs, list) else [envs]
        self.args = args
        self.kwargs = kwargs

        self.global_step = 0
        self.wandb = wandb
        self.logger = logger or logging.getLogger(__name__)

        self._initialize_obs()

    def log(self, key: str, value: float | np.floating, step: int):
        if self.wandb:
            self.wandb.log({key: value}, step=step)

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    @abstractmethod
    def train(self, iterations: int, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_action(self, obs: TensorDict) -> TensorDict:
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
        self.obs = [self.split_observation(obs) for obs in obs_list]

    def split_observation(self, obs: dict) -> TensorDict:
        """Split the observation into separate components if necessary."""
        observation = TensorDict(obs["observation"]).to(torch.float32)
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            action_mask = TensorDict(action_mask).to(torch.bool)
        return TensorDict({"observation": observation, "action_mask": action_mask})

    def normalize_observation(self, obs: TensorDict) -> TensorDict:
        if self.observation_normalizer:
            obs = self.observation_normalizer.normalize(obs)
        return obs

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.reward_normalizer:
            reward = self.reward_normalizer.normalize(reward)
        return reward
