from abc import ABC, abstractmethod


class RewardNormalizer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def normalize(self, reward: float) -> float:
        """Normalize the reward if necessary."""
        return self._normalize(reward)

    @abstractmethod
    def _normalize(self, reward: float) -> float:
        """Normalize the reward if necessary."""
        pass
