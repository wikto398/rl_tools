"""Serializable description of one evolutionary child.

The orchestrator already hands a child everything it needs as three plain
values — ``base_args`` (the parsed CLI namespace), ``genome`` and
``child_config``. ``multiprocessing`` pickles them; the Docker launcher instead
writes them to ``job.json`` and runs ``child_main`` inside the container. Both
paths funnel through :func:`run_child_job`, so a child behaves identically
whether it runs in-process or in a container.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively convert a value into something ``json.dump`` accepts.

    Handles the bits of the args namespace that are not JSON-native (enums such
    as ``game_engine_type``, ``Path`` objects, tuples/sets).
    """
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def encode_job(
    runner_spec: str,
    base_args: argparse.Namespace | dict,
    genome: dict,
    child_config: dict,
) -> dict:
    """Build the JSON-safe child job description."""
    base = vars(base_args) if isinstance(base_args, argparse.Namespace) else base_args
    return {
        "runner_spec": runner_spec,
        "base_args": jsonable(base),
        "genome": jsonable(genome),
        "child_config": jsonable(child_config),
    }


def write_job(path: str, job: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(job, f, indent=2)
    return path


def read_job(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_child_job(job: dict) -> None:
    """Run one child from a job description (the actual work).

    Imports the child-runner class lazily so this module stays torch-free for
    callers that only build/launch jobs (the orchestrator parent process).
    """
    from rl_tools.rl.Evolution._specs import resolve_spec

    cls = resolve_spec(job["runner_spec"])
    cls.run(job["base_args"], job["genome"], job["child_config"])


def spawn_worker(job: dict) -> None:
    """``multiprocessing.Process`` target (must be a module-level function)."""
    run_child_job(job)
