"""Evolution strategies (algorithm policies) for ``EvolutionOrchestrator``.

Concrete strategies are imported lazily by :func:`build_strategy` so importing
the shared types never drags in a specific algorithm.
"""

from __future__ import annotations

from rl_tools.rl.Evolution._specs import resolve_spec
from rl_tools.rl.Evolution.GenomeSpace import GenomeSpace
from rl_tools.rl.Evolution.strategy.EvolutionStrategy import (
    EvolutionHooks,
    EvolutionStrategy,
    Stats,
)
from rl_tools.rl.Evolution.strategy.Individual import Individual
from rl_tools.rl.Evolution.strategy.Population import Population, default_fitness_key

#: Short ``evolution.strategy`` alias -> concrete class spec.
ALIASES = {
    "generational": (
        "rl_tools.rl.Evolution.strategy.GenerationalStrategy.GenerationalStrategy"
    ),
    "pbt": "rl_tools.rl.Evolution.strategy.PBTStrategy.PBTStrategy",
}


def resolve_strategy_class(spec: str):
    target = ALIASES.get(str(spec).strip().lower(), str(spec).strip())
    return resolve_spec(target)


def build_strategy(spec: str, cfg: dict, space: GenomeSpace) -> EvolutionStrategy:
    """Resolve ``spec`` (alias or ``module:Class``) and construct it from config."""
    return resolve_strategy_class(spec).from_config(cfg, space)


__all__ = [
    "ALIASES",
    "EvolutionHooks",
    "EvolutionStrategy",
    "Individual",
    "Population",
    "Stats",
    "build_strategy",
    "default_fitness_key",
    "resolve_strategy_class",
]
