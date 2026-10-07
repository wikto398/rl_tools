"""Population container shared by all evolution strategies.

Ranks individuals with a lexicographic fitness key, supports tournament
selection and elitist eviction. Algorithm-agnostic: the generational strategy
replaces the whole population each phase, while PBT adds/evicts members
(``μ+λ``).
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from rl_tools.rl.Evolution.strategy.Individual import Individual

#: Lexicographic ranking: win_rate, then fewer turns to win, then buildings.
WORST_KEY: tuple[float, float, float] = (-1.0, 0.0, 0.0)


def default_fitness_key(fitness: dict | None) -> tuple[float, float, float]:
    """Score an individual's fitness dict (higher is better).

    A child that errored before its first rollout (no steps) ranks worst, not
    as an ordinary 0-win child.
    """
    if not fitness:
        return WORST_KEY
    if fitness.get("errored") or int(fitness.get("global_step", 1) or 0) == 0:
        return WORST_KEY
    win_rate = float(fitness.get("win_rate", 0.0) or 0.0)
    mean_win_turns = fitness.get("mean_win_turns")
    f2 = -float(mean_win_turns) if mean_win_turns is not None else 0.0
    f3 = float(fitness.get("buildings_completed", 0.0) or 0.0)
    return (win_rate, f2, f3)


class Population:
    def __init__(
        self,
        members: list[Individual] | None = None,
        fitness_key=default_fitness_key,
    ) -> None:
        self.members: list[Individual] = list(members or [])
        self.fitness_key = fitness_key

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.members)

    def add(self, individual: Individual) -> None:
        self.members.append(individual)

    def replace(self, members: list[Individual]) -> None:
        self.members = list(members)

    def evaluated(self) -> list[Individual]:
        return [m for m in self.members if m.evaluated]

    def ranked(self) -> list[Individual]:
        """Evaluated members, best first."""
        return sorted(
            self.evaluated(),
            key=lambda m: self.fitness_key(m.fitness),
            reverse=True,
        )

    @property
    def best(self) -> Individual | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def key(self, individual: Individual) -> tuple[float, float, float]:
        return self.fitness_key(individual.fitness)

    def tournament(self, k: int, rng: random.Random) -> Individual | None:
        pool = self.evaluated()
        if not pool:
            return None
        k = max(1, min(int(k), len(pool)))
        return max(rng.sample(pool, k), key=self.key)

    def evict_worst(self, protect: int = 1) -> Individual | None:
        """Drop the worst evaluated member, never touching the top ``protect``."""
        if len(self.members) <= protect:
            return None
        ascending = sorted(self.members, key=self.key)
        victim = ascending[0]
        self.members.remove(victim)
        return victim
