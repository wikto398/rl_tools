"""One individual (a candidate genome) in an evolutionary population."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Individual:
    """A genome plus everything the orchestrator learns about it.

    The engine fills in ``child_dir``/``job_name``/``checkpoint`` when it
    launches the run; the strategy owns ``genome``, ``budget`` and the naming
    fields (``phase``, ``slot``, ``phase_index``, ``seed_offset``).
    """

    genome: dict
    budget: int
    #: ``phase`` is the child-directory bucket (``gen_3`` / ``pbt``); ``slot`` is
    #: the index within it; ``phase_index`` names the launcher/container.
    phase: str = "gen_0"
    slot: int = 0
    phase_index: int = 0
    seed_offset: int = 0
    #: fitness dict once evaluated (``None`` while pending).
    fitness: dict | None = None
    #: after launch: the checkpoint this run will *produce*
    #: (``<child_dir>/checkpoints/final.pt``). Before launch it may hold the
    #: donor checkpoint this run *started from* (PBT warm start).
    checkpoint: str | None = None
    child_dir: str | None = None
    job_name: str | None = None
    global_step: int = 0
    stopped_early: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def evaluated(self) -> bool:
        return self.fitness is not None
