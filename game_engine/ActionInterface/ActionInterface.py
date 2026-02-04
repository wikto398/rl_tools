from abc import ABC, abstractmethod

class ActionInterface(ABC):
    def __init__(self):
        super().__init__()

    def send_action(self, action):
        """Send the given action to the environment."""
        print(f"Sending action: {action}")
        try:
            self._send_action(action)
        except KeyboardInterrupt:
            print("Action sending interrupted by user.")
            exit(0)
    
    @abstractmethod
    def _send_action(self, action):
        """Internal method to handle action sending logic."""
        pass

    @property
    def action(self):
        """Get the current action."""
        return self._action

    @action.setter
    def action(self, value):
        """Set the current action."""
        self._action = value