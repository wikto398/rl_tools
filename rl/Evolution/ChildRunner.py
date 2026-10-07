from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rl_tools.game_engine.ObservationInterface.UDPObservation import UDPObservation
from rl_tools.rl.Evolution.FitnessCallback import FitnessCallback
from rl_tools.rl.RLInitializer.RLIntializer import RLInitializer

if TYPE_CHECKING:
    from rl_tools.rl.Trainer import Trainer


@dataclass
class ChildSpec:
    """Decoded genome for one child: how each gene maps onto a run."""

    agent_overrides: dict = field(default_factory=dict)
    engine_args: dict = field(default_factory=dict)
    expert_eps: float = 0.0
    expert_eps_decay_steps: int = 0


class ChildRunner:
    """Generic skeleton for running one evolutionary child (a single Trainer).

    Subclass and override ``decode_genome`` (which keys map to agent
    overrides / engine args / expert) and ``build_trainer`` (game factories).
    All methods are classmethods so ``ConcreteChildRunner.run`` is picklable and
    can be a ``multiprocessing.Process`` target under ``spawn``.
    """

    observation_class = UDPObservation

    @classmethod
    def _configure(cls) -> None:
        """Per-child environment hook (e.g. CPU thread limits). Override in
        subclasses; default is a no-op."""

    @classmethod
    def run(cls, base_args: dict, genome: dict, child_config: dict) -> None:
        cls._configure()
        spec = cls.decode_genome(genome)
        child_args = cls.build_child_args(base_args, spec, child_config)
        trainer = cls.build_trainer(child_args)
        initializer = RLInitializer(
            child_args,
            log_path=child_config["child_dir"],
            observation_class=cls.observation_class,
        )
        callbacks = cls.build_callbacks(child_config)
        trainer.run(
            overrides=spec.agent_overrides,
            initializer=initializer,
            extra_callbacks=callbacks,
        )

    @classmethod
    def build_child_args(
        cls, base_args: dict, spec: ChildSpec, child_config: dict
    ) -> argparse.Namespace:
        """Build the per-child args namespace from the base parsed args."""
        args = argparse.Namespace(**base_args)
        base_engine = list(getattr(args, "engine_args", None) or [])
        genome_engine = [f"{k}={v}" for k, v in spec.engine_args.items()]
        args.engine_args = base_engine + genome_engine
        args.instances = child_config["instances"]
        args.eval_instances = child_config.get("eval_instances", 1)
        args.eval_episodes = child_config.get("eval_episodes", 10)
        args.eval_every_timesteps = child_config.get("eval_every", 10000)
        args.max_steps = child_config["max_steps"]
        args.seed = child_config.get("seed", getattr(args, "seed", None))
        args.port_offset = child_config.get("port_offset", 0)
        # PBT-lite warm start: continue from a donor child's checkpoint.
        checkpoint = child_config.get("checkpoint")
        if checkpoint:
            args.checkpoint = checkpoint
            args.no_load_optimizer = bool(child_config.get("no_load_optimizer", False))
            args.no_load_rng = bool(child_config.get("no_load_rng", True))
        args.expert_eps = spec.expert_eps
        args.expert_eps_decay_steps = spec.expert_eps_decay_steps
        args.torch_compile = False
        args.wandb_name = child_config.get("wandb_name")
        args.wandb_group = child_config.get("wandb_group")
        args.wandb_extra_tags = child_config.get("wandb_extra_tags")
        args.sweep_config = None
        args.gate_step = None
        args.gates_config = None
        args.stop_metric = None
        return args

    @classmethod
    def build_callbacks(cls, child_config: dict) -> list:
        """Game-agnostic child callbacks (fitness export). Game-specific
        stopping callbacks are added by subclasses."""
        return [
            FitnessCallback(os.path.join(child_config["child_dir"], "fitness.json"))
        ]

    @classmethod
    def decode_genome(cls, genome: dict) -> ChildSpec:
        raise NotImplementedError

    @classmethod
    def build_trainer(cls, child_args: argparse.Namespace) -> Trainer:
        raise NotImplementedError
