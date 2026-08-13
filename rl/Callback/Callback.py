from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensordict import TensorDict

if TYPE_CHECKING:
    from rl_tools.rl.RLAgent import RLAgent


class Callback(ABC):
    """SB3-style training hook interface. Subclasses must implement all hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.agent: RLAgent | None = None

    def setup(self, agent: RLAgent) -> None:
        """One-time init when the agent attaches this callback. Optional override."""
        self.agent = agent

    def detach(self) -> None:
        """Remove this callback from the agent's callback container.

        Only effective when the agent holds a ``CallbackList``; a no-op
        otherwise. Safe to call from inside any hook (the container iterates
        over snapshots).
        """
        if self.agent is None:
            return
        from rl_tools.rl.Callback.CallbackList import CallbackList

        container = self.agent.callback
        if isinstance(container, CallbackList):
            container.remove(self)

    @abstractmethod
    def on_train_start(self) -> None: ...

    @abstractmethod
    def on_train_end(self) -> None: ...

    @abstractmethod
    def on_rollout_start(self) -> None: ...

    @abstractmethod
    def on_rollout_end(self, rollout: TensorDict) -> None: ...

    @abstractmethod
    def on_step(
        self,
        *,
        actions: Any,
        rewards: Sequence[float],
        dones: Sequence[bool],
        infos: Sequence[dict],
    ) -> bool:
        """Called after each env step across the vector. Return False to stop training."""
        ...

    @abstractmethod
    def on_update_start(self, rollout: TensorDict) -> None: ...

    @abstractmethod
    def on_update_end(self, update_info: dict) -> None: ...


class NoOpCallback(Callback):
    """Default callback that performs no work."""

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self, rollout: TensorDict) -> None:
        pass

    def on_step(
        self,
        *,
        actions: Any,
        rewards: Sequence[float],
        dones: Sequence[bool],
        infos: Sequence[dict],
    ) -> bool:
        return True

    def on_update_start(self, rollout: TensorDict) -> None:
        pass

    def on_update_end(self, update_info: dict) -> None:
        pass
