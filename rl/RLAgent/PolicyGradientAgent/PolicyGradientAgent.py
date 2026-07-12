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
            self.global_step += self.rollout_size * len(self.envs)
            self.update_steps += 1
            self.logger.debug(f"Collected rollout with {len(rollout_buffer)} steps")
            rollout_buffer = self.compute_gae(rollout_buffer=rollout_buffer)
            self.logger.debug("Computed GAE advantages and returns for rollout")
            self.update(rollout_buffer, *args, **kwargs)

    @abstractmethod
    def update(
        self,
        rollout_buffer: TensorDict,
        *args,
        **kwargs,
    ):
        pass

    def evaluate(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        action_mask: TensorDict | None = None,
    ):
        obs = obs.to(self.device)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)

        result = self.network.evaluate(
            obs,
            actions,
            action_mask,
        )

        return (
            result["log_probs"],
            result["value"],
            result["entropy"],
        )

    def get_action(self, obs: TensorDict) -> TensorDict:
        with torch.no_grad():
            observation = obs["observation"].to(self.device)
            action_mask = obs.get("action_mask", None)
            if action_mask is not None:
                action_mask = action_mask.to(self.device)
            result = self.network(observation, action_mask)
        return result

    def compute_gae(self, rollout_buffer: TensorDict) -> TensorDict:
        rewards = rollout_buffer["rewards"].to(self.device)  # [T, N_ENVS]
        values = rollout_buffer["values"].to(self.device)  # [T, N_ENVS]
        dones = rollout_buffer["dones"].to(self.device)  # [T, N_ENVS]
        T = len(rewards)

        # bootstrap next value from current obs
        with torch.no_grad():
            next_td = [
                self.network(
                    obs["observation"].to(self.device),
                    obs["action_mask"].to(self.device)
                    if obs.get("action_mask") is not None
                    else None,
                )
                for obs in self.obs
            ]
            next_value = torch.stack([td["value"].squeeze(-1) for td in next_td]).to(
                self.device
            )  # [N_ENVS]

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(rewards[0])
        last_return = next_value  # bootstrap from current obs

        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()

            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            advantages[t] = delta + self.gamma * self.lam * last_advantage * mask
            returns[t] = rewards[t] + self.gamma * last_return * mask
            last_advantage = advantages[t]
            last_return = returns[t]

        return rollout_buffer.update({"advantages": advantages, "returns": returns})

    def collect_rollouts(self) -> TensorDict:
        rollout = []

        for _ in range(self.rollout_size):
            step_data = []

            # --- policy forward ---
            for obs in self.obs:
                forward_result = self.get_action(obs)

                td = TensorDict(
                    {
                        "observations": obs["observation"].squeeze(
                            0
                        ),  # already TensorDict
                        "action_masks": obs["action_mask"].squeeze(0),
                        "actions": forward_result["action"].squeeze(0),
                        "log_probs": forward_result["log_prob"].squeeze(0),
                        "values": forward_result["value"].squeeze(-1),
                    },
                    batch_size=[],
                )

                step_data.append(td)

            # --- stack envs → [N] ---
            step_td = torch.stack(step_data)

            # --- env step ---
            next_obs = []
            rewards = []
            dones = []

            for i, env in enumerate(self.envs):
                action = step_td["actions"][i].cpu().numpy()
                o, r, d, _ = env.step(action)

                if d:
                    o = env.reset()

                next_obs.append(o)
                rewards.append(r)
                dones.append(d)

            # add rewards/dones to step TensorDict
            step_td["rewards"] = torch.tensor(rewards, dtype=torch.float32)
            step_td["dones"] = torch.tensor(dones, dtype=torch.bool)

            rollout.append(step_td)

            # update obs
            self.obs = [self.split_observation(o) for o in next_obs]

        # --- stack time → [T, N] ---
        rollout_td = torch.stack(rollout)

        return rollout_td
