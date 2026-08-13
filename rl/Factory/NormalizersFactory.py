from __future__ import annotations

from abc import ABC, abstractmethod


class NormalizersFactory(ABC):
    """Builds observation/reward normalizers for a training run."""

    @abstractmethod
    def build(self, args):
        """Return ``(observation_normalizer, reward_normalizer)``."""
