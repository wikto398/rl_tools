from abc import ABC, abstractmethod


class ObservationNormalizer(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def normalize(self, observation: dict) -> dict:
        """Normalize the observation data if necessary."""
        return self._normalize(observation)

    @abstractmethod
    def _normalize(self, observation: dict) -> dict:
        """Normalize the observation data if necessary."""
        pass
