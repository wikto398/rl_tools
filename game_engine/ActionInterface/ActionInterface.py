from abc import ABC, abstractmethod
import logging


class ActionInterface(ABC):
    def __init__(self, logger: logging.Logger | None = None, **kwargs):
        super().__init__()
        self._logger = logger if logger else logging.getLogger(__name__)

    def send_action(self, action):
        """Send the given action to the environment."""
        self._logger.info(f"Sending action: {action}")
        try:
            self._send_action(action)
        except KeyboardInterrupt:
            self._logger.error("Action sending interrupted by user.")
            exit(0)

    @abstractmethod
    def _send_action(self, action):
        """Internal method to handle action sending logic."""
        pass

    def close(self) -> None:
        """Release any resources (e.g. sockets) held by this interface."""
        pass

    @property
    def action(self):
        """Get the current action."""
        return self._action

    @action.setter
    def action(self, value):
        """Set the current action."""
        self._action = value
