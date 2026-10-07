"""PBT-lite: steady-state evolution with checkpoint warm starts.

Each free worker slot launches an ``epoch_steps`` child that is warm-started
from a selected donor's checkpoint (exploit) and given a mutated genome
(explore). No generational barrier, so no slot ever waits on a straggler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rl_tools.rl.Evolution.GenomeSpace import GenomeSpace
from rl_tools.rl.Evolution.strategy.EvolutionStrategy import (
    EvolutionStrategy,
    Stats,
)
from rl_tools.rl.Evolution.strategy.Individual import Individual
from rl_tools.rl.Evolution.strategy.Population import Population, default_fitness_key


@dataclass
class PBTConfig:
    population: int = 4
    epoch_steps: int = 50000
    #: probability a new child warm-starts from its donor's checkpoint
    exploit_fraction: float = 0.5
    #: per-gene mutation probability for the new genome
    mutate_rate: float = 1.0
    #: multiplier on each gene's own ``mutate_scale``
    mutate_scale: float = 1.2
    tournament_k: int = 3
    elite_count: int = 1

    @classmethod
    def from_cfg(cls, cfg: dict) -> PBTConfig:
        pbt = cfg.get("pbt") or {}
        workers = int(cfg.get("workers", cls.population) or cls.population)
        return cls(
            population=int(pbt.get("population", workers)),
            epoch_steps=int(pbt.get("epoch_steps", cls.epoch_steps)),
            exploit_fraction=float(pbt.get("exploit_fraction", cls.exploit_fraction)),
            mutate_rate=float(pbt.get("mutate_rate", cls.mutate_rate)),
            mutate_scale=float(pbt.get("mutate_scale", cls.mutate_scale)),
            tournament_k=int(pbt.get("tournament_k", cls.tournament_k)),
            elite_count=int(pbt.get("elite_count", cls.elite_count)),
        )


class PBTStrategy(EvolutionStrategy):
    name = "pbt"
    supports_resume = False

    def __init__(
        self,
        space: GenomeSpace,
        config: PBTConfig,
        fitness_key=default_fitness_key,
    ) -> None:
        super().__init__(space, {}, fitness_key)
        self.config = config
        self._counter = 0

    @classmethod
    def from_config(cls, cfg: dict, space: GenomeSpace) -> PBTStrategy:
        return cls(space, PBTConfig.from_cfg(cfg))

    def max_workers(self) -> int:
        return self.config.population

    def initial_population(self, rng: random.Random) -> list[Individual]:
        # Members are created on demand as slots free; nothing is pre-seeded.
        return []

    def next_pending(
        self, pop: Population, active: int, rng: random.Random
    ) -> Individual | None:
        pool = pop.evaluated()
        if not pool:
            # Bootstrap: keep sampling fresh genomes until something is evaluated.
            return self._new(rng, self.space.sample(rng), None)

        donor = pop.tournament(self.config.tournament_k, rng)
        genome = self.space.mutate(
            donor.genome,
            rng,
            rate=self.config.mutate_rate,
            scale=self.config.mutate_scale,
        )
        checkpoint = None
        if donor.checkpoint and rng.random() < self.config.exploit_fraction:
            checkpoint = donor.checkpoint
        return self._new(rng, genome, checkpoint)

    def _new(
        self, rng: random.Random, genome: dict, checkpoint: str | None
    ) -> Individual:
        slot = self._counter
        self._counter += 1
        return Individual(
            genome=genome,
            budget=self.config.epoch_steps,
            checkpoint=checkpoint,
            phase="pbt",
            slot=slot,
            phase_index=slot,
            seed_offset=slot,
        )

    def on_child_finished(
        self,
        ind: Individual,
        fitness: dict,
        pop: Population,
        stats: Stats,
        rng: random.Random,
    ) -> None:
        ind.fitness = fitness
        pop.add(ind)
        pop.evict_worst(protect=self.config.elite_count)
        stats.completed += 1
        stats.completed_steps += ind.budget

    def should_stop(self, pop: Population, stats: Stats) -> tuple[bool, str]:
        if self.hooks and self.hooks.stop_requested():
            return True, "stop requested"
        return False, ""
