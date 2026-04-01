from abc import ABC, abstractmethod
import logging


class ObservationInterface(ABC):
    def __init__(self, logger: logging.Logger | None = None, **kwargs):
        super().__init__()
        self._observation = None
        self._logger = logger if logger else logging.getLogger(__name__)

    def get_observation(self) -> dict | None:
        """Retrieve the current observation from the environment."""
        try:
            data = self._get_observation()
        except KeyboardInterrupt:
            if self._logger:
                self._logger.error("Observation receiving interrupted by user.")
            exit(0)
        return self.parse_observation(data)

    def get_raw_message(self) -> bytes | None:
        """Get the raw message from the environment without parsing."""
        try:
            return self._get_observation()
        except KeyboardInterrupt:
            if self._logger:
                self._logger.error("Observation receiving interrupted by user.")
            exit(0)

    @abstractmethod
    def _get_observation(self) -> bytes | None:
        """Retrieve the current observation from the environment."""
        pass

    @abstractmethod
    def parse_observation(self, raw_observation: bytes | None) -> dict | None:
        """Parse the raw observation data if necessary."""
        pass
