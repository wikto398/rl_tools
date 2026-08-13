from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class NetworkFactory(ABC):
    """Builds the game's policy/value network.

    Game-specific construction lives behind this interface so the generic
    ``Trainer`` in ``rl_tools`` never imports the game package.
    """

    @property
    @abstractmethod
    def building_names(self) -> Sequence[str]:
        """Names of all buildings in agent-index order."""

    @abstractmethod
    def build(self):
        """Return a fresh network instance."""
