from tensordict import TensorDict
import torch
import numpy as np
from rl_tools.game_engine.ObservationNormalizer import (
    ObservationNormalizer,
)
from rl_tools.game_engine.RewardNormalizer import RewardNormalizer
from rl_tools.rl.Environment import Environment
from rl_tools.rl.RLAgent.PolicyGradientAgent import PolicyGradientAgent


class PPOAgent(PolicyGradientAgent):
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
        clip_epsilon: float = 0.2,
        observation_keys: list[str] | None = None,
        action_mask_keys: list[str] | None = None,
        entropy_coef_decay_steps: int = 1_000_000,
        entropy_coef_start: float = 0.7,
        entropy_coef_end: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            network=network,
            optimizer=optimizer,
            envs=envs,
            reward_normalizer=reward_normalizer,
            observation_normalizer=observation_normalizer,
            device=device,
            *args,
            gamma=gamma,
            lam=lam,
            epochs=epochs,
            batch_size=batch_size,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            rollout_size=rollout_size,
            observation_keys=observation_keys,
            action_mask_keys=action_mask_keys,
            **kwargs,
        )
        self.clip_epsilon = clip_epsilon
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_coef_decay_steps = entropy_coef_decay_steps

    def clip_gradients(self, max_norm: float):
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm)

    def update(
        self,
        rollout_buffer: TensorDict,
        *args,
        **kwargs,
    ):
        full_batch = rollout_buffer.reshape(-1)

        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []

        frac = min(self.global_step / self.entropy_coef_decay_steps, 1.0)
        current_entropy_coef = self.entropy_coef_start + frac * (
            self.entropy_coef_end - self.entropy_coef_start
        )

        for epoch in range(self.epochs):
            indices = torch.randperm(full_batch.shape[0])

            for start in range(0, full_batch.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch: TensorDict = full_batch[batch_indices]

                obs_batch = self.normalize_observation(batch["observations"])
                action_mask_batch = batch["action_masks"]
                action_batch = batch["actions"]
                old_log_probs_batch = batch["log_probs"].detach()
                advantages_batch = batch["advantages"]
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                    advantages_batch.std() + 1e-8
                )
                returns_batch = batch["returns"]

                new_log_probs, new_values, entropy = self.evaluate(
                    obs_batch, action_batch, action_mask_batch
                )

                new_values = new_values.squeeze(-1)

                log_ratio = (new_log_probs - old_log_probs_batch).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages_batch
                clipped = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages_batch
                )

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(new_values, returns_batch)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + current_entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.clip_gradients(max_norm=0.5)
                self.optimizer.step()

                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction.item())

        update_info = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "entropy_coef": float(current_entropy_coef),
        }
        self.log("loss/policy", update_info["policy_loss"])
        self.log("loss/value", update_info["value_loss"])
        self.log("loss/entropy", update_info["entropy_loss"])
        self.log("train/clip_fraction", update_info["clip_fraction"])
        self.log("train/entropy_coef", update_info["entropy_coef"])
        self.callback.on_update_end(update_info)
