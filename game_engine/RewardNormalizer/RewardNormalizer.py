from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RewardNormalizer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def normalize(
        self,
        reward: np.ndarray,
        dones: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize the reward if necessary.

        ``dones`` marks episode boundaries (one entry per env in the same order
        as ``reward``); normalizers may use it to update episodic statistics.
        """
        return self._normalize(reward, dones)

    @abstractmethod
    def _normalize(
        self,
        reward: np.ndarray,
        dones: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize the reward if necessary."""
        pass

    def state_dict(self) -> Any:
        return None

    def load_state_dict(self, state: Any) -> None:
        pass
