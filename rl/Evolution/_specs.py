"""Resolve ``module:Class`` spec strings to objects.

Project-agnostic entrypoints (e.g. ``python -m rl_tools.rl.Evolution``) take the
game-specific classes as specs so the framework never imports the game:

    --space torch_files.Evolution.StrategyEvolutionSpace:StrategyEvolutionSpace
    --child_runner torch_files.Evolution.StrategyChildRunner:StrategyChildRunner
"""

from __future__ import annotations

import importlib
from typing import Any


def split_spec(spec: str) -> tuple[str, str]:
    """Split ``"pkg.module:Attr"`` (or legacy ``"pkg.module.Attr"``)."""
    if not spec or not isinstance(spec, str):
        raise ValueError(f"spec must be a non-empty string, got {spec!r}")
    if ":" in spec:
        module_name, _, attr = spec.partition(":")
    else:
        module_name, _, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"spec must be 'module:Attr' or 'module.Attr', got {spec!r}")
    return module_name, attr


def resolve_spec(spec: str) -> Any:
    """Import and return the attribute named by ``spec``."""
    module_name, attr = split_spec(spec)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise AttributeError(f"{module_name!r} has no attribute {attr!r}") from e
