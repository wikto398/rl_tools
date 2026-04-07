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

    def clip_gradients(self, max_norm: float):
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm)

    def update(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        rollout_buffer: TensorDict,
        *args,
        **kwargs,
    ):
        rollout: TensorDict = self._prepare_rollout(advantages, returns, rollout_buffer)
        for epoch in range(self.epochs):
            policy_losses = []
            value_losses = []
            entropy_losses = []
            clip_fractions = []
            for batch in rollout.batch(self.batch_size, self.device):
                obs_batch = self.normalize_observation(batch.observations)
                action_mask_batch = batch.action_masks
                action_batch = batch.actions
                old_log_probs_batch = batch.old_log_probs
                advantages_batch = self.normalize_advantages(batch.advantages)
                returns_batch = batch.returns

                new_log_probs, new_values, entropy = self.evaluate(
                    obs_batch, action_batch, action_mask_batch
                )

                ratio = torch.exp(new_log_probs - old_log_probs_batch)
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
                    + self.entropy_coef * entropy_loss
                )

                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction.item())

                loss.backward()
                self.clip_gradients(max_norm=0.5)
                self.optimizer.step()

            self.log("loss/policy", np.mean(policy_losses), self.global_step)
            self.log("loss/value", np.mean(value_losses), self.global_step)
            self.log("loss/entropy", np.mean(entropy_losses), self.global_step)
            self.log("train/clip_fraction", np.mean(clip_fractions), self.global_step)

            self.info(
                f"Step {self.global_step}, Epoch {epoch + 1}/{self.epochs} - Policy Loss: {np.mean(policy_losses):.4f}, Value Loss: {np.mean(value_losses):.4f}, Entropy Loss: {np.mean(entropy_losses):.4f}, Clip Fraction: {np.mean(clip_fractions):.4f}"
            )
            self.global_step += 1
