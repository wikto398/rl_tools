from __future__ import annotations

from abc import ABC, abstractmethod

from rl_tools.rl.Factory import ExpertInterface


class ExpertFactory(ABC):
    """Builds the game's expert-in-the-loop coach (optional).

    Game-specific construction lives behind this interface so the generic
    ``Trainer`` in ``rl_tools`` never imports the game package.
    """

    @abstractmethod
    def build(self, args) -> ExpertInterface:
        """Return an ``ExpertInterface`` instance (or ``None`` if disabled)."""
