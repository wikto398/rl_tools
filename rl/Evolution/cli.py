"""Project-agnostic evolution CLI.

Games usually ship a one-line wrapper (e.g. ``torch_files/evolution.py``) that
passes their specs, but the framework can also be driven directly:

    python -m rl_tools.rl.Evolution \
        --space torch_files.Evolution.StrategyEvolutionSpace:StrategyEvolutionSpace \
        --child_runner torch_files.Evolution.StrategyChildRunner:StrategyChildRunner \
        --config torch_files/evolution.yaml --executor docker \
        --image strategy-resource:dev
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rl_tools.rl.Evolution.BudgetSchedule import BudgetSchedule
from rl_tools.rl.Evolution.EvolutionOrchestrator import EvolutionOrchestrator
from rl_tools.rl.Evolution._specs import resolve_spec
from rl_tools.rl.RLArgsParser import RLArgsParser

# CLI flag -> `evolution:` config key.
_CLI_TO_CFG = {
    "executor": "executor",
    "image": "image",
    "docker_network": "network",
    "docker_cpus": "cpus",
    "docker_memory": "memory",
    "docker_shm": "shm_size",
    "docker_extra_args": "extra_docker_args",
}


def load_evo_cfg(config_path: str | None) -> dict:
    """Read the ``evolution:`` section of a YAML config (empty if absent)."""
    if not config_path:
        return {}
    data = yaml.safe_load(Path(config_path).read_text())
    if isinstance(data, dict):
        return data.get("evolution", {}) or {}
    return {}


def _merge_cli(args, evo_cfg: dict) -> dict:
    """Overlay evolution-related CLI flags on top of the YAML section."""
    cfg = dict(evo_cfg)
    for attr, key in _CLI_TO_CFG.items():
        value = getattr(args, attr, None)
        if value is not None:
            cfg[key] = value
    if getattr(args, "no_docker_gpu", False):
        cfg["gpu"] = False
    return cfg


def run_evolution(
    space_spec: str | None = None,
    child_runner_spec: str | None = None,
    argv: list[str] | None = None,
) -> dict:
    args = RLArgsParser.parse_args(argv)
    evo_cfg = _merge_cli(args, load_evo_cfg(getattr(args, "config", None)))

    space_spec = space_spec or getattr(args, "space", None) or evo_cfg.get("space")
    child_runner_spec = (
        child_runner_spec
        or getattr(args, "child_runner", None)
        or evo_cfg.get("child_runner")
    )
    if not space_spec:
        raise SystemExit(
            "No genome space configured: pass --space module:Class or set "
            "evolution.space in the config."
        )
    if not child_runner_spec:
        raise SystemExit(
            "No child runner configured: pass --child_runner module:Class or set "
            "evolution.child_runner in the config."
        )

    space = resolve_spec(space_spec).genome_space()
    budget = BudgetSchedule(
        start=evo_cfg.get("steps_start", 50000),
        end=evo_cfg.get("steps_end", 750000),
        curve=evo_cfg.get("steps_curve", "geometric"),
    )
    resume_dir = getattr(args, "resume_evolution", None)
    orchestrator = EvolutionOrchestrator(
        args,
        space,
        budget,
        child_runner_spec,
        evo_cfg=evo_cfg,
        root_dir=resume_dir,
        resume=bool(resume_dir),
    )
    result = orchestrator.evolve()
    print(f"Evolution finished. Root: {orchestrator.root_dir}")
    print("Best genome:", result.get("best_genome"))
    print("Best fitness:", result.get("best_fitness"))
    return result


def main(argv: list[str] | None = None) -> dict:
    return run_evolution(argv=argv)
