from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RewardNormalizer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def normalize(self, reward: np.ndarray) -> np.ndarray:
        """Normalize the reward if necessary."""
        return self._normalize(reward)

    @abstractmethod
    def _normalize(self, reward: np.ndarray) -> np.ndarray:
        """Normalize the reward if necessary."""
        pass

    def state_dict(self) -> Any:
        return None

    def load_state_dict(self, state: Any) -> None:
        pass
