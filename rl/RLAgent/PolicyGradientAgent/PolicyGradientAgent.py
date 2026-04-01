import numpy as np
import torch

from dataclasses import dataclass, field

from rl.RLAgent.RLAgent import RLAgent
from rl_tools.game_engine.ObservationNormalizer.ObservationNormalizer import (
    ObservationNormalizer,
)
from rl_tools.game_engine.RewardNormalizer.RewardNormalizer import RewardNormalizer
from rl_tools.rl.Environment.Environment import Environment


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    dones: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)


class PolicyGradientAgent(RLAgent):
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
        super().__init__(
            network=network,
            optimizer=optimizer,
            envs=envs,
            reward_normalizer=reward_normalizer,
            observation_normalizer=observation_normalizer,
            device=device,
            gamma=gamma,
            lam=lam,
            *args,
            **kwargs,
        )
        self.rollout_buffer: RolloutBuffer = RolloutBuffer()

    def compute_gae(
        self, rewards, values, dones, next_value
    ) -> tuple[np.ndarray, np.ndarray]:
        gamma = self.gamma
        lam = self.lam
        advantages: np.ndarray = np.zeros_like(rewards)
        gae = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
                next_done = 0
            else:
                next_val = values[step + 1]
                next_done = dones[step + 1]

            delta = rewards[step] + gamma * next_val * (1 - next_done) - values[step]
            gae = delta + gamma * lam * (1 - next_done) * gae
            advantages[step] = gae

        returns: np.ndarray = advantages + values
        return advantages, returns

    def collect_rollouts(self, num_steps: int):
        self.rollout_buffer = RolloutBuffer()

        for _ in range(num_steps):
            obs_tensor = torch.tensor(self.obs["observation"], dtype=torch.float32).to(
                self.device
            )
            mask_tensor = torch.tensor(self.obs["action_mask"], dtype=torch.bool).to(
                self.device
            )

            with torch.no_grad():
                action_probs, value = self.network(obs_tensor, mask_tensor)

            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)

            next_obs, rewards, dones = [], [], []
            for i, env in enumerate(self.envs):
                o, r, d, _ = env.step(actions[i].item())
                if d:
                    o = env.reset()
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)

            self.rollout_buffer.observations.append(self.obs["observation"])
            self.rollout_buffer.action_masks.append(self.obs["action_mask"])
            self.rollout_buffer.actions.append(actions.cpu().numpy())
            self.rollout_buffer.log_probs.append(log_probs.cpu().numpy())
            self.rollout_buffer.rewards.append(np.array(rewards))
            self.rollout_buffer.dones.append(np.array(dones))
            self.rollout_buffer.values.append(value.squeeze().cpu().numpy())

            self.obs = {
                "observation": np.array([o["observation"] for o in next_obs]),
                "action_mask": np.array([o["action_mask"] for o in next_obs]),
            }
