from __future__ import annotations

from typing import Literal

import numpy as np

from rl_tools.game_engine.GameEnvConnector import GameEnvConnector

SeedMode = Literal["train", "eval", "none"]


class Environment:
    """Godot-backed env with deterministic advancing map-seed sequences.

    Train (n parallel, index i, episode e):  seed_base + i + e * n
    Eval  (n parallel, index j, episode e):  -(1 + j + e * n)
    """

    def __init__(
        self,
        env_connector: GameEnvConnector,
        *,
        seed_mode: SeedMode = "none",
        seed_base: int | None = None,
        env_index: int = 0,
        n_parallel: int = 1,
    ):
        if env_index < 0:
            raise ValueError(f"env_index must be >= 0, got {env_index}")
        if n_parallel < 1:
            raise ValueError(f"n_parallel must be >= 1, got {n_parallel}")
        if seed_mode == "train" and seed_base is None:
            seed_mode = "none"
        self.env_connector = env_connector
        self.seed_mode: SeedMode = seed_mode
        self.seed_base = seed_base
        self.env_index = env_index
        self.n_parallel = n_parallel
        self.episode_index = 0

    def next_seed(self) -> int | None:
        if self.seed_mode == "none":
            return None
        e = self.episode_index
        i = self.env_index
        n = self.n_parallel
        if self.seed_mode == "train":
            assert self.seed_base is not None
            return int(self.seed_base) + i + e * n
        if self.seed_mode == "eval":
            return -(1 + i + e * n)
        return None

    def reset(self, *, restart_sequence: bool = False) -> dict[str, np.ndarray]:
        if restart_sequence:
            self.episode_index = 0
        seed = self.next_seed()
        if seed is not None:
            self.episode_index += 1
        obs = self.env_connector.reset(seed=seed)
        return obs

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        obs, reward, done, info = self.env_connector.step(action.tolist())
        return obs, reward, done, info
