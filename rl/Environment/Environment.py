import numpy as np
from rl_tools.game_engine.GameEnvConnector import GameEnvConnector


class Environment:
    def __init__(
        self,
        env_connector: GameEnvConnector,
    ):
        self.env_connector = env_connector

    def reset(self) -> dict[str, np.ndarray]:
        obs = self.env_connector.reset()
        return obs

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        obs, reward, done, info = self.env_connector.step(action)
        return obs, reward, done, info
