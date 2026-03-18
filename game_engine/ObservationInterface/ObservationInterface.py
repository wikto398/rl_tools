from abc import ABC, abstractmethod
import logging

class ObservationInterface(ABC):
    def __init__(self, logger: logging.Logger | None = None, **kwargs):
        super().__init__()
        self._observation = None
        self._logger = logger if logger else logging.getLogger(__name__)

    def get_observation(self):
        """Retrieve the current observation from the environment."""
        try:
            data = self._get_observation()
        except KeyboardInterrupt:
            if self._logger:
                self._logger.error("Observation receiving interrupted by user.")
            exit(0)
        return data

    @abstractmethod
    def _get_observation(self) -> bytes | None:
        """Retrieve the current observation from the environment."""
        pass

    @abstractmethod
    def parse_observation(self, raw_observation: bytes) -> dict | bytes | list:
        """Parse the raw observation data if necessary."""
        pass
