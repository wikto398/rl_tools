from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class GeneSpec:
    """Definition of one evolvable gene.

    ``type`` is one of:
      - ``float``: uniform in [min, max], gaussian mutation
      - ``log_float``: log-uniform in [min, max] (min/max are actual values)
      - ``int``: uniform integer in [min, max]
      - ``choice``: one of ``values``
      - ``list_ints``: fixed-length list of ints in [min, max], element-wise
        mutation by ``step`` (e.g. milestone threshold lists)
    """

    type: str
    min: float | None = None
    max: float | None = None
    values: list[Any] | None = None
    length: int | None = None
    step: int = 1
    mutate_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.type == "choice":
            if not self.values:
                raise ValueError("choice genes require values")
        elif self.type == "list_ints":
            if not self.length:
                raise ValueError("list_ints genes require length")
        elif self.type in ("float", "log_float", "int"):
            if self.min is None or self.max is None:
                raise ValueError(f"{self.type} genes require min and max")
        else:
            raise ValueError(f"unknown gene type: {self.type}")


class GenomeSpace:
    """Sample / mutate / crossover a genome from a set of ``GeneSpec``s."""

    def __init__(self, genes: dict[str, GeneSpec]):
        self.genes = genes

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {key: self._sample_one(spec, rng) for key, spec in self.genes.items()}

    def mutate(
        self, genome: dict[str, Any], rng: random.Random, rate: float = 0.2
    ) -> dict[str, Any]:
        out = dict(genome)
        for key, spec in self.genes.items():
            if key in out and rng.random() < rate:
                out[key] = self._mutate_one(spec, out[key], rng)
        return out

    def crossover(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        rng: random.Random,
        rate: float = 0.8,
    ) -> dict[str, Any]:
        """Uniform per-gene crossover: with prob ``rate`` take the gene from ``b``."""
        out = {}
        for key in self.genes:
            take_b = rng.random() < rate and key in b
            out[key] = b.get(key) if take_b else a.get(key)
        return out

    # --- per-type sampling / mutation ---

    def _sample_one(self, spec: GeneSpec, rng: random.Random) -> Any:
        if spec.type == "float":
            return rng.uniform(spec.min, spec.max)
        if spec.type == "log_float":
            return self._log_uniform(float(spec.min), float(spec.max), rng)
        if spec.type == "int":
            return rng.randint(int(spec.min), int(spec.max))
        if spec.type == "choice":
            return rng.choice(spec.values)
        if spec.type == "list_ints":
            return [
                rng.randint(int(spec.min), int(spec.max)) for _ in range(spec.length)
            ]
        raise ValueError(f"unknown gene type: {spec.type}")

    @staticmethod
    def _log_uniform(lo: float, hi: float, rng: random.Random) -> float:
        import math

        if lo <= 0 or hi <= 0:
            raise ValueError("log_float bounds must be positive")
        return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi))

    def _mutate_one(self, spec: GeneSpec, value: Any, rng: random.Random) -> Any:
        if spec.type == "float":
            span = (spec.max - spec.min) * spec.mutate_scale
            return float(min(spec.max, max(spec.min, value + rng.gauss(0.0, span))))
        if spec.type == "log_float":
            import math

            lo, hi = math.log10(spec.min), math.log10(spec.max)
            cur = math.log10(max(value, 1e-12))
            span = (hi - lo) * spec.mutate_scale
            return 10.0 ** min(hi, max(lo, cur + rng.gauss(0.0, span)))
        if spec.type == "int":
            span = max(1, int((spec.max - spec.min) * spec.mutate_scale))
            return int(min(spec.max, max(spec.min, value + rng.randint(-span, span))))
        if spec.type == "choice":
            others = [v for v in spec.values if v != value] or spec.values
            return rng.choice(others)
        if spec.type == "list_ints":
            out = list(value)
            idx = rng.randrange(len(out))
            out[idx] = int(
                min(spec.max, max(spec.min, out[idx] + rng.choice((-1, 1)) * spec.step))
            )
            return out
        raise ValueError(f"unknown gene type: {spec.type}")
