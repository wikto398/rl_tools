"""Generational GA: evaluate a fixed population per generation, then evolve."""

from __future__ import annotations

import random
from dataclasses import dataclass

from rl_tools.rl.Evolution.BudgetSchedule import BudgetSchedule
from rl_tools.rl.Evolution.GenomeSpace import GenomeSpace
from rl_tools.rl.Evolution.strategy.EvolutionStrategy import (
    EvolutionStrategy,
    Stats,
)
from rl_tools.rl.Evolution.strategy.Individual import Individual
from rl_tools.rl.Evolution.strategy.Population import Population, default_fitness_key


@dataclass
class GenerationalConfig:
    population: int = 8
    generations: int = 30
    steps_start: int = 50000
    steps_end: int = 750000
    steps_curve: str = "geometric"
    elite_count: int = 1
    tournament_k: int = 3
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8

    @classmethod
    def from_cfg(cls, cfg: dict) -> GenerationalConfig:
        return cls(
            population=int(cfg.get("population", cls.population)),
            generations=int(cfg.get("generations", cls.generations)),
            steps_start=int(cfg.get("steps_start", cls.steps_start)),
            steps_end=int(cfg.get("steps_end", cls.steps_end)),
            steps_curve=str(cfg.get("steps_curve", cls.steps_curve)),
            elite_count=int(cfg.get("elite_count", cls.elite_count)),
            tournament_k=int(cfg.get("tournament_k", cls.tournament_k)),
            mutation_rate=float(cfg.get("mutation_rate", cls.mutation_rate)),
            crossover_rate=float(cfg.get("crossover_rate", cls.crossover_rate)),
        )


class GenerationalStrategy(EvolutionStrategy):
    """One generation at a time: ``population`` children, a barrier, then
    elitism + tournament selection + crossover/mutation."""

    name = "generational"
    supports_resume = True

    def __init__(
        self,
        space: GenomeSpace,
        config: GenerationalConfig,
        fitness_key=default_fitness_key,
    ) -> None:
        super().__init__(space, {}, fitness_key)
        self.config = config
        self.budget_schedule = BudgetSchedule(
            start=config.steps_start,
            end=config.steps_end,
            curve=config.steps_curve,
        )
        self._generation = 0
        self._budget = 0
        self._queue: list[Individual] = []
        self._evaluated = 0
        self._expected = config.population
        self._exhausted = False

    @classmethod
    def from_config(cls, cfg: dict, space: GenomeSpace) -> GenerationalStrategy:
        return cls(space, GenerationalConfig.from_cfg(cfg))

    def max_workers(self) -> int:
        return self.config.population

    # --- lifecycle ---

    def initial_population(self, rng: random.Random) -> list[Individual]:
        return self._start_generation(0, None, rng)

    def resume(self, state: dict, rng: random.Random) -> list[Individual]:
        generation = int(state["next_generation"])
        genomes = list(state["next_population"])
        return self._start_generation(generation, genomes, rng)

    def _start_generation(
        self,
        generation: int,
        genomes: list[dict] | None,
        rng: random.Random,
    ) -> list[Individual]:
        self._generation = generation
        self._budget = self.budget_schedule(generation, self.config.generations)
        if genomes is None:
            genomes = [self.space.sample(rng) for _ in range(self.config.population)]
        individuals = self._individuals(generation, genomes)
        self._queue = list(individuals)
        self._evaluated = 0
        self._expected = len(individuals)
        self._exhausted = False
        if self.hooks:
            self.hooks.phase_start(generation, self._budget, genomes)
        return individuals

    def _individuals(self, generation: int, genomes: list[dict]) -> list[Individual]:
        return [
            Individual(
                genome=genome,
                budget=self._budget,
                phase=f"gen_{generation}",
                slot=i,
                phase_index=generation,
                seed_offset=generation * self.config.population + i,
            )
            for i, genome in enumerate(genomes)
        ]

    # --- scheduling ---

    def next_pending(self, pop, active: int, rng: random.Random) -> Individual | None:
        # A queued individual is only ever launched once; no new work is created
        # until ``on_child_finished`` advances the generation.
        return self._queue.pop(0) if self._queue else None

    def on_child_finished(
        self,
        ind: Individual,
        fitness: dict,
        pop: Population,
        stats: Stats,
        rng: random.Random,
    ) -> None:
        ind.fitness = fitness
        self._evaluated += 1
        stats.completed += 1
        stats.completed_steps += ind.budget
        if self._evaluated < self._expected:
            return

        # Generation complete: rank, report, evolve, report, start next.
        ranked = pop.ranked()
        pairs = [(m.genome, m.fitness) for m in ranked]
        if self.hooks and not self.hooks.phase_end(self._generation, pairs):
            self._exhausted = True
            return

        next_genomes = self._evolve(ranked, rng)
        if self.hooks:
            self.hooks.phase_snapshot(self._generation, pairs, next_genomes)
            if self.hooks.stop_requested():
                self._exhausted = True
                return
        if self._generation + 1 >= self.config.generations:
            self._exhausted = True
            return
        individuals = self._start_generation(self._generation + 1, next_genomes, rng)
        pop.replace(individuals)

    def should_stop(self, pop, stats: Stats) -> tuple[bool, str]:
        return (True, "generations complete") if self._exhausted else (False, "")

    # --- selection / variation ---

    def _evolve(self, ranked: list[Individual], rng: random.Random) -> list[dict]:
        population = self.config.population
        elite_count = min(self.config.elite_count, population)
        next_gen = [m.genome for m in ranked[:elite_count]]
        while len(next_gen) < population:
            a = self._tournament(ranked, rng)
            b = self._tournament(ranked, rng)
            child = self.space.crossover(a, b, rng, rate=self.config.crossover_rate)
            child = self.space.mutate(child, rng, rate=self.config.mutation_rate)
            next_gen.append(child)
        return next_gen

    def _tournament(self, ranked: list[Individual], rng: random.Random) -> dict:
        k = min(self.config.tournament_k, len(ranked))
        contenders = rng.sample(ranked, k)
        best = max(contenders, key=lambda m: self.fitness_key(m.fitness))
        return best.genome
