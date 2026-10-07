"""The evolutionary-algorithm policy.

``EvolutionOrchestrator`` is the scheduler: it owns worker slots, the launcher,
logging, snapshots and stop handling. *What* to run next, *when* the population
is done, and *how* to reproduce are decided here by an ``EvolutionStrategy``.

Concrete strategies live in sibling packages (``GenerationalStrategy/``,
``PBTStrategy/``); adding one is implementing this ABC + a config dataclass.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from rl_tools.rl.Evolution.GenomeSpace import GenomeSpace
from rl_tools.rl.Evolution.strategy.Individual import Individual
from rl_tools.rl.Evolution.strategy.Population import default_fitness_key


@dataclass
class Stats:
    """Run-level counters handed to strategies."""

    completed: int = 0
    completed_steps: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.started_at


class EvolutionHooks(Protocol):
    """Engine callbacks a strategy may invoke (implemented by the orchestrator).

    ``phase_*`` are generic lifecycle events; the generational strategy uses them
    for its generation boundaries (mapping to the engine's generation callbacks),
    PBT simply never emits them.
    """

    def phase_start(self, phase: int, budget: int, genomes: list[dict]) -> None: ...

    def phase_end(self, phase: int, ranked: list[tuple[dict, dict]]) -> bool:
        """Report a finished phase; return False to stop instead of evolving."""

    def phase_snapshot(
        self,
        phase: int,
        ranked: list[tuple[dict, dict]],
        next_population: list[dict],
    ) -> None: ...

    def stop_requested(self) -> bool: ...


class EvolutionStrategy(ABC):
    #: Short alias for ``evolution.strategy``.
    name: ClassVar[str] = "abstract"
    #: Whether ``resume()`` is implemented.
    supports_resume: ClassVar[bool] = False

    def __init__(
        self,
        space: GenomeSpace,
        cfg: dict | None = None,
        fitness_key=default_fitness_key,
    ) -> None:
        self.space = space
        self.cfg = cfg or {}
        self.fitness_key = fitness_key
        self.hooks: EvolutionHooks | None = None

    def bind(self, hooks: EvolutionHooks) -> None:
        self.hooks = hooks

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict, space: GenomeSpace) -> EvolutionStrategy:
        """Build the strategy from the merged ``evolution:`` config."""

    @abstractmethod
    def initial_population(self, rng: random.Random) -> list[Individual]:
        """Individuals to seed the population with (may be empty)."""

    @abstractmethod
    def next_pending(self, pop, active: int, rng: random.Random) -> Individual | None:
        """The next individual to launch now, or ``None`` to wait."""

    @abstractmethod
    def on_child_finished(
        self, ind: Individual, fitness: dict, pop, stats: Stats, rng: random.Random
    ) -> None:
        """Record a finished child and advance the algorithm if a phase ended."""

    def max_workers(self) -> int:
        """Upper bound on concurrent children (e.g. population size)."""
        return 10**9

    def should_stop(self, pop, stats: Stats) -> tuple[bool, str]:
        return False, ""

    def resume(self, state: dict, rng: random.Random) -> list[Individual]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support --resume_evolution"
        )
