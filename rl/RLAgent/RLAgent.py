from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from rl_tools.game_engine.ObservationNormalizer import (
    ObservationNormalizer,
)
from rl_tools.game_engine.RewardNormalizer import RewardNormalizer
from rl_tools.rl.Blackboard import Blackboard
from rl_tools.rl.Callback import Callback, NoOpCallback
from rl_tools.rl.Environment import Environment


class RLAgent(ABC):
    def __init__(
        self,
        network: torch.nn.Module,
        envs: list[Environment] | Environment,
        optimizer: torch.optim.Optimizer | None = None,
        reward_normalizer: RewardNormalizer | None = None,
        observation_normalizer: ObservationNormalizer | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        callback: Callback | None = None,
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
        self.update_steps = 0
        self._stop_requested = False
        self.blackboard = Blackboard()
        self.logger = logger or logging.getLogger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.envs))
        self.callback = callback or NoOpCallback()
        self.callback.setup(self)

        self._initialize_obs()

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

    def get_action(self, obs: TensorDict) -> TensorDict:
        with torch.no_grad():
            obs = obs.to(self.device)
            action_mask = obs.get("action_mask", None)
            obs = self.normalize_observation(obs["observation"])
            result = self.network(obs, action_mask)
        return result

    def save(self, path: str):
        payload = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            "global_step": self.global_step,
            "update_steps": self.update_steps,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "observation_normalizer": (
                self.observation_normalizer.state_dict()
                if self.observation_normalizer is not None
                else None
            ),
            "reward_normalizer": (
                self.reward_normalizer.state_dict()
                if self.reward_normalizer is not None
                else None
            ),
            "entropy_coef": (
                self._entropy_coef if hasattr(self, "_entropy_coef") else None
            ),
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path_obj)

    def load(
        self,
        path: str,
        *,
        load_optimizer: bool = True,
        load_rng: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "network" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'network' key: {path}")
        self.network.load_state_dict(checkpoint["network"])

        if load_optimizer and "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "global_step" in checkpoint:
            self.global_step = int(checkpoint["global_step"])
        if "update_steps" in checkpoint:
            self.update_steps = int(checkpoint["update_steps"])

        if load_rng:
            if "torch_rng" in checkpoint and checkpoint["torch_rng"] is not None:
                torch.set_rng_state(checkpoint["torch_rng"])
            if (
                "cuda_rng" in checkpoint
                and checkpoint["cuda_rng"] is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            if "numpy_rng" in checkpoint and checkpoint["numpy_rng"] is not None:
                np.random.set_state(checkpoint["numpy_rng"])
            if "python_rng" in checkpoint and checkpoint["python_rng"] is not None:
                random.setstate(checkpoint["python_rng"])

        obs_state = checkpoint.get("observation_normalizer")
        if obs_state is not None and self.observation_normalizer is not None:
            self.observation_normalizer.load_state_dict(obs_state)

        reward_state = checkpoint.get("reward_normalizer")
        if reward_state is not None and self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(reward_state)

        if (
            hasattr(self, "_entropy_coef")
            and checkpoint.get("entropy_coef") is not None
        ):
            self._entropy_coef = float(checkpoint["entropy_coef"])

        self.info(
            f"Loaded checkpoint from {path} "
            f"(global_step={self.global_step}, update_steps={self.update_steps})"
        )

    def set_eval_mode(self, eval_mode: bool, deterministic: bool | None = None):
        if eval_mode:
            self.network.eval()
            if deterministic is not None:
                self.network.deterministic = deterministic
        else:
            self.network.train()
            if hasattr(self.network, "deterministic"):
                self.network.deterministic = False
        if self.observation_normalizer is not None and hasattr(
            self.observation_normalizer, "training"
        ):
            self.observation_normalizer.training = not eval_mode
        if self.reward_normalizer is not None and hasattr(
            self.reward_normalizer, "training"
        ):
            self.reward_normalizer.training = not eval_mode

    def _initialize_obs(self):
        obs_list = [env.reset() for env in self.envs]
        self.obs = [self.split_observation(obs) for obs in obs_list]

    def split_observation(self, obs: dict) -> TensorDict:
        observation_dict = obs["observation"]
        action_mask_dict = obs.get("action_mask", None)

        observation = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in observation_dict.items()
        }

        if action_mask_dict is not None:
            action_mask = {
                k: torch.tensor(v, dtype=torch.bool)
                for k, v in action_mask_dict.items()
            }
        else:
            action_mask = None

        return TensorDict(
            {
                "observation": TensorDict(observation, batch_size=[]),
                "action_mask": TensorDict(action_mask, batch_size=[])
                if action_mask
                else None,
            },
            batch_size=[],
        )

    def normalize_observation(self, obs: TensorDict) -> TensorDict:
        if self.observation_normalizer:
            obs = self.observation_normalizer.normalize(obs)
        return obs

    def normalize_reward(
        self,
        reward: np.ndarray,
        dones: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.reward_normalizer:
            reward = self.reward_normalizer.normalize(reward, dones)
        return reward

    def _step_env(self, env, action):
        o, r, d, info = env.step(action)

        if d:
            o = env.reset()

        return o, r, d, info
