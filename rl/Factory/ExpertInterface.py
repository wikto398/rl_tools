from __future__ import annotations

from abc import ABC, abstractmethod

from tensordict import TensorDict


class ExpertInterface(ABC):
    """Computes expert actions for the expert-in-the-loop coach.

    The expert is applied as a per-step action override inside
    ``PolicyGradientAgent.collect_rollouts`` (an "on-policy coach"): with
    probability ``expert_eps`` a training step executes the expert's action
    instead of the agent's sampled one, and the stored ``log_prob`` is
    recomputed for that action so the PPO ratio stays consistent. The expert
    never runs during evaluation.
    """

    @abstractmethod
    def action(
        self,
        obs: TensorDict,
        action_mask: TensorDict | None = None,
    ) -> TensorDict:
        """Return expert actions for a batch of raw observations.

        Args:
            obs: raw (unnormalized) observation ``TensorDict`` with
                ``observation`` (fields/global/builders) and optional
                ``action_mask`` keys, batch size ``(B,)``.
            action_mask: optional action-mask ``TensorDict`` with the same
                batch size.

        Returns:
            A ``TensorDict`` with a LongTensor ``action`` of shape ``(B, 4)``
            matching the network's action vector ``[action, builder, building,
            cell]`` (int format consumed by the environment / ActionExecutor).
        """
