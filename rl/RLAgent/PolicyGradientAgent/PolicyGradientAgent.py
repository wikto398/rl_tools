from abc import abstractmethod

import numpy as np
import torch
from tensordict import TensorDict

from rl_tools.game_engine.ObservationNormalizer import (
    ObservationNormalizer,
)
from rl_tools.game_engine.RewardNormalizer import RewardNormalizer
from rl_tools.rl.Environment import Environment
from rl_tools.rl.RLAgent import RLAgent


class PolicyGradientAgent(RLAgent):
    def __init__(
        self,
        network: torch.nn.Module,
        envs: list[Environment] | Environment,
        optimizer: torch.optim.Optimizer | None = None,
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
        if self.optimizer is None:
            raise RuntimeError("optimizer is required for training")
        self._stop_requested = False
        self.callback.on_train_start()
        try:
            for iteration in range(iterations):
                if self._stop_requested:
                    break
                self.callback.on_rollout_start()
                rollout_buffer = self.collect_rollouts()
                self.global_step += self.rollout_size * len(self.envs)
                self.update_steps += 1
                self.blackboard.record(
                    "rollout/expert_eps", self.current_expert_eps(), self.global_step
                )
                self.logger.debug(f"Collected rollout with {len(rollout_buffer)} steps")
                self.callback.on_rollout_end(rollout_buffer)
                if self._stop_requested:
                    break
                if self.observation_normalizer is not None:
                    self.observation_normalizer.update(rollout_buffer["observations"])
                if self.reward_normalizer is not None and hasattr(
                    self.reward_normalizer, "scale"
                ):
                    self.blackboard.record(
                        "reward/norm_scale",
                        self.reward_normalizer.scale,
                        self.global_step,
                    )
                rollout_buffer = self.compute_gae(rollout_buffer=rollout_buffer)
                self.logger.debug("Computed GAE advantages and returns for rollout")
                self.callback.on_update_start(rollout_buffer)
                self.update(rollout_buffer, *args, **kwargs)
        finally:
            self.callback.on_train_end()

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

    def compute_gae(self, rollout_buffer: TensorDict) -> TensorDict:
        rollout_buffer = rollout_buffer.to(self.device)

        rewards = rollout_buffer["rewards"]
        values = rollout_buffer["values"]
        dones = rollout_buffer["dones"]

        T = rewards.shape[0]

        # bootstrap value
        with torch.no_grad():
            batch_obs = torch.stack(self.obs).to(self.device)

            next_obs = self.normalize_observation(batch_obs["observation"])

            next_td = self.network(
                next_obs,
                batch_obs.get("action_mask", None),
            )

            next_value = next_td["value"].squeeze(-1)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        last_advantage = torch.zeros_like(next_value)
        last_return = next_value

        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()

            next_val = next_value if t == T - 1 else values[t + 1]

            delta = rewards[t] + self.gamma * next_val * mask - values[t]

            advantages[t] = delta + self.gamma * self.lam * last_advantage * mask

            returns[t] = rewards[t] + self.gamma * last_return * mask

            last_advantage = advantages[t]
            last_return = returns[t]

        rollout_buffer.set("advantages", advantages)
        rollout_buffer.set("returns", returns)

        return rollout_buffer

    def collect_rollouts(self) -> TensorDict:
        rollout = []

        for _ in range(self.rollout_size):
            if self._stop_requested:
                break

            batch_obs = torch.stack(self.obs)
            forward_results = self.get_action(batch_obs)

            if self.current_expert_eps() > 0.0:
                forward_results = self._apply_expert_actions(batch_obs, forward_results)

            step_td = TensorDict(
                {
                    "observations": batch_obs["observation"],
                    "action_masks": batch_obs.get("action_mask", None),
                    "actions": forward_results["action"],
                    "log_probs": forward_results["log_prob"],
                    "values": forward_results["value"],
                },
                batch_size=[len(self.envs)],
            )

            next_obs = []
            rewards = []
            dones = []
            infos = []

            actions = [
                step_td["actions"][i].cpu().numpy() for i in range(len(self.envs))
            ]

            results = list(
                self.thread_pool.map(
                    self._step_env,
                    self.envs,
                    actions,
                )
            )

            for o, r, d, info in results:
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)
                infos.append(info if info is not None else {})

            step_td["rewards"] = torch.tensor(
                self.normalize_reward(
                    np.asarray(rewards, dtype=np.float32),
                    np.asarray(dones, dtype=np.bool_),
                ),
                dtype=torch.float32,
            )
            step_td["dones"] = torch.tensor(dones, dtype=torch.bool)

            rollout.append(step_td)

            self.obs = [self.split_observation(o) for o in next_obs]

            if not self.callback.on_step(
                actions=step_td["actions"],
                rewards=rewards,
                dones=dones,
                infos=infos,
            ):
                self._stop_requested = True
                break

        rollout_td = torch.stack(rollout)

        return rollout_td

    def _apply_expert_actions(
        self,
        batch_obs: TensorDict,
        forward_results: TensorDict,
    ) -> TensorDict:
        """Expert-in-the-loop override (on-policy coach).

        With probability ``current_expert_eps()`` per env, replace the agent's
        sampled action with the expert's action and recompute its log-prob
        under the current policy, so the stored ``actions``/``log_probs`` stay
        consistent for the PPO importance ratio. Recomputes in a single batched
        forward over the selected envs and skips entries where the expert
        agrees with the agent. Eval never goes through here.
        """
        eps_now = self.current_expert_eps()
        n = len(self.envs)
        use_expert = torch.rand(n) < eps_now
        if not use_expert.any():
            return forward_results

        expert_result = self.expert.action(
            batch_obs, batch_obs.get("action_mask", None)
        )
        expert_actions = expert_result["action"].to(self.device)

        idx = torch.nonzero(use_expert).flatten()
        idx_d = idx.to(self.device)
        changed = (expert_actions[idx_d] != forward_results["action"][idx_d]).any(
            dim=-1
        )
        idx = idx[changed.cpu()]
        idx_d = idx.to(self.device)
        if idx.numel() == 0:
            return forward_results

        forward_results["action"][idx_d] = expert_actions[idx_d]
        with torch.no_grad():
            obs_sel = self.normalize_observation(batch_obs["observation"][idx])
            mask_sel = batch_obs.get("action_mask", None)
            if mask_sel is not None:
                mask_sel = mask_sel[idx]
            log_prob, _, _ = self.evaluate(obs_sel, expert_actions[idx_d], mask_sel)
            forward_results["log_prob"][idx_d] = log_prob

        return forward_results
