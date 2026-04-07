import numpy as np
from tensordict import TensorDict
import torch

from abc import abstractmethod

from rl_tools.rl.RLAgent import RLAgent
from rl_tools.game_engine.ObservationNormalizer import (
    ObservationNormalizer,
)
from rl_tools.game_engine.RewardNormalizer import RewardNormalizer
from rl_tools.rl.Environment import Environment


class PolicyGradientAgent(RLAgent):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        envs: list[Environment] | Environment,
        reward_normalizer: RewardNormalizer | None = None,
        observation_normalizer: ObservationNormalizer | None = None,
        device: torch.device | None = None,
        *args,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 5,
        batch_size: int = 64,
        rollout_size: int = 2048,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            network=network,
            optimizer=optimizer,
            envs=envs,
            reward_normalizer=reward_normalizer,
            observation_normalizer=observation_normalizer,
            device=device,
            gamma=gamma,
            lam=lam,
            epochs=epochs,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.rollout_size = rollout_size

    def train(self, iterations: int, *args, **kwargs):
        for iteration in range(iterations):
            rollout_buffer = self.collect_rollouts()
            advantages, returns = self.compute_gae(rollout_buffer=rollout_buffer)
            self.update(advantages, returns, rollout_buffer, *args, **kwargs)

    @abstractmethod
    def update(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        rollout_buffer: TensorDict,
        *args,
        **kwargs,
    ):
        pass

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple:
        action_probs, values = self.network(obs, action_mask)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy

    def get_action(self, obs: TensorDict) -> TensorDict:
        self.logger.debug(f"Getting action for observation: {obs}")
        with torch.no_grad():
            observation = obs["observation"].to(self.device)
            action_mask = obs.get("action_mask", None)
            if action_mask is not None:
                action_mask = action_mask.to(self.device)
            result = self.network(observation, action_mask)
        self.logger.debug(
            f"Action: {result['action']}, Log Prob: {result['log_prob']}, Value: {result['value']}"
        )
        return result

    def compute_gae(self, rollout_buffer: TensorDict) -> tuple[np.ndarray, np.ndarray]:
        rewards = rollout_buffer["rewards"]
        values = rollout_buffer["values"]
        dones = rollout_buffer["dones"]

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_advantage = 0
        last_return = values[-1]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            advantages[t] = delta + self.gamma * self.lam * last_advantage * mask
            returns[t] = rewards[t] + self.gamma * last_return * mask

            last_advantage = advantages[t]
            last_return = returns[t]

        return advantages, returns

    def collect_rollouts(self) -> TensorDict:
        observations = []
        action_masks = []
        actions_list = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        for _ in range(self.rollout_size):
            actions, log_probs, values = [], [], []
            for obs in self.obs:
                forward_result = self.get_action(obs)
                actions.append(forward_result["action"])
                log_probs.append(forward_result["log_prob"])
                values.append(forward_result["value"])

            next_obs, rewards, dones = [], [], []
            for i, env in enumerate(self.envs):
                o, r, d, _ = env.step(actions[i])
                if d:
                    o = env.reset()
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)
            observations.append([obs["observation"] for obs in self.obs])
            action_masks.append([obs["action_mask"] for obs in self.obs])
            actions_list.append(actions)
            log_probs.append(log_probs)
            rewards.append(np.array(rewards))
            dones.append(np.array(dones))
            values.append(values)

            self.obs = [self.split_observation(obs) for obs in next_obs]

        return TensorDict(
            {
                "observations": np.array(observations),
                "action_masks": np.array(action_masks),
                "actions": np.array(actions_list),
                "log_probs": np.array(log_probs),
                "rewards": np.array(rewards),
                "dones": np.array(dones),
                "values": np.array(values),
            }
        )

    def _flatten_rollout(self, array: np.ndarray) -> np.ndarray:
        """Flatten [N_STEPS, N_ENVS, ...] to [N_STEPS * N_ENVS, ...]"""
        return array.reshape(-1, *array.shape[2:])

    def _prepare_rollout(
        self, advantages: np.ndarray, returns: np.ndarray, rollout_buffer: TensorDict
    ) -> TensorDict:
        """Convert rollout buffer to flattened tensors ready for update"""
        observations = self._flatten_rollout(rollout_buffer["observations"])
        action_masks = self._flatten_rollout(rollout_buffer["action_masks"])
        actions = self._flatten_rollout(rollout_buffer["actions"])
        log_probs = self._flatten_rollout(rollout_buffer["log_probs"])
        advantages = self._flatten_rollout(advantages)
        returns = self._flatten_rollout(returns)

        return TensorDict(
            {
                "observations": torch.tensor(observations, dtype=torch.float32),
                "action_masks": torch.tensor(action_masks, dtype=torch.float32)
                if action_masks is not None
                else None,
                "actions": torch.tensor(actions, dtype=torch.int64),
                "old_log_probs": torch.tensor(log_probs, dtype=torch.float32),
                "advantages": torch.tensor(advantages, dtype=torch.float32),
                "returns": torch.tensor(returns, dtype=torch.float32),
            }
        )

    def normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
