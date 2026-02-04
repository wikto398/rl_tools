from abc import ABC, abstractmethod

class ObservationInterface(ABC):
    def __init__(self):
        super().__init__()
        self._observation = None

    def get_observation(self):
        """Retrieve the current observation from the environment."""
        try:
            data = self._get_observation()
        except KeyboardInterrupt:
            print("Observation receiving interrupted by user.")
            exit(0)
        return data

    @abstractmethod
    def _get_observation(self):
        """Retrieve the current observation from the environment."""
        pass

    @property
    def normalized_observation(self):
        """Normalize the given observation."""
        pass

    @property
    def observation(self):
        """Get the current observation."""
        return self._observation

    @observation.setter
    def observation(self, value):
        """Set the current observation."""
        self._observation = value