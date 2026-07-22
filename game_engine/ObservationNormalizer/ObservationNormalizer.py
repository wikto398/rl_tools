from abc import ABC, abstractmethod
from typing import Any

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

    def state_dict(self) -> Any:
        return None

    def load_state_dict(self, state: Any) -> None:
        pass
