from typing import Any

import numpy as np
import torch

from rl_tools.game_engine.RewardNormalizer import RewardNormalizer
from rl_tools.game_engine.RunningMeanStd import RunningMeanStd


class RunningMeanStdRewardNormalizer(RewardNormalizer):
    """RunningMeanStd reward normalization (VecNormalize-style).

    Tracks a per-env discounted-return accumulator updated with the raw
    reward; when an episode ends (``dones`` entry True) the accumulated return
    feeds a ``RunningMeanStd`` and that entry is reset. Rewards are then scaled
    by ``1 / max(1, std(ret))`` so early, small-magnitude rewards are not
    amplified before the return distribution is known, and clipped to
    ``[-clip_reward, clip_reward]``.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        clip_reward: float | None = 10.0,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.epsilon = epsilon
        self.training = True
        self.ret = np.zeros(0, dtype=np.float32)
        self.ret_rms = RunningMeanStd((1,), epsilon=epsilon)

    @property
    def scale(self) -> float:
        return 1.0 / max(1.0, float(np.sqrt(self.ret_rms.var.item() + self.epsilon)))

    def _normalize(
        self,
        reward: np.ndarray,
        dones: np.ndarray | None = None,
    ) -> np.ndarray:
        reward = np.asarray(reward, dtype=np.float32)
        n = reward.shape[0]
        if self.ret.shape[0] < n:
            self.ret = np.concatenate(
                [self.ret, np.zeros(n - self.ret.shape[0], dtype=np.float32)]
            )
        self.ret = self.ret[:n] * self.gamma + reward
        if dones is not None:
            done = np.asarray(dones, dtype=np.bool_)
            if done.any():
                if self.training:
                    ret_done = torch.from_numpy(self.ret[done]).unsqueeze(1)
                    self.ret_rms.update(ret_done)
                self.ret[done] = 0.0
        normalized = reward * self.scale
        if self.clip_reward is not None:
            normalized = np.clip(normalized, -self.clip_reward, self.clip_reward)
        return normalized.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "ret": self.ret,
            "ret_rms": self.ret_rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            self.ret = np.zeros(0, dtype=np.float32)
            self.ret_rms = RunningMeanStd((1,), epsilon=self.epsilon)
            return
        self.ret = np.asarray(state["ret"], dtype=np.float32)
        self.ret_rms = RunningMeanStd((1,), epsilon=self.epsilon)
        self.ret_rms.load_state_dict(state["ret_rms"])
