from abc import ABC, abstractmethod
from tensordict import TensorDict


class ObservationNormalizer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def normalize(self, observation: TensorDict) -> TensorDict:
        """Normalize the observation data if necessary."""
        return self._normalize(observation)

    @abstractmethod
    def _normalize(self, observation: TensorDict) -> TensorDict:
        """Normalize the observation data if necessary."""
        pass
