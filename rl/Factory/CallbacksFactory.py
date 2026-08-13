from __future__ import annotations

from abc import ABC, abstractmethod


class CallbacksFactory(ABC):
    """Builds the game's metric callbacks."""

    @abstractmethod
    def build(self):
        """Return a list of game-specific callback instances."""
