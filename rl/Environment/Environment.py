from game_engine.GameEnvConnector.GameEnvConnector import GameEnvConnector
from game_engine.ObservationNormalizer.ObservationNormalizer import (
    ObservationNormalizer,
)
from game_engine.RewardNormalizer.RewardNormalizer import RewardNormalizer


class Environment:
    def __init__(
        self,
        env_connector: GameEnvConnector,
        observation_normalizer: ObservationNormalizer | None = None,
        reward_normalizer: RewardNormalizer | None = None,
    ):
        self.env_connector = env_connector
        self.observation_normalizer = observation_normalizer
        self.reward_normalizer = reward_normalizer

    def reset(self) -> dict:
        obs = self.env_connector.reset()
        if self.observation_normalizer:
            obs = self.observation_normalizer.normalize(obs)
        return obs

    def step(self, action) -> tuple[dict, float, bool, dict]:
        obs, reward, done, info = self.env_connector.step(action)
        if self.observation_normalizer:
            obs = self.observation_normalizer.normalize(obs)
        if self.reward_normalizer:
            reward = self.reward_normalizer.normalize(reward)
        return obs, reward, done, info
