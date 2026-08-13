from __future__ import annotations

import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.tensorboard.writer import SummaryWriter
from rl_tools.rl.Callback import CallbackList
from rl_tools.rl.Callback.ConsoleCallback import ConsoleCallback
from rl_tools.rl.Callback.EvalCallback import EvalCallback
from rl_tools.rl.Callback.NetworkSaveCallback import NetworkSaveCallback
from rl_tools.rl.Callback.StopTrainingCallback.GateStopCallback import GateStopCallback
from rl_tools.rl.Callback.StopTrainingCallback.KeyStopCallback import KeyStopCallback
from rl_tools.rl.Callback.StopTrainingCallback.MetricStopCallback import (
    MetricStopCallback,
)
from rl_tools.rl.Callback.TensorboardCallback import TensorboardCallback
from rl_tools.rl.Callback.TimingCallback import TimingCallback
from rl_tools.rl.Callback.WandbCallback import WandbCallback
from rl_tools.rl.Factory import CallbacksFactory, NetworkFactory, NormalizersFactory
from rl_tools.rl.RLAgent.PolicyGradientAgent.PPOAgent import PPOAgent
from rl_tools.rl.RLInitializer import RLInitializer
from rl_tools.rl.WandbWrapper import WandbWrapper

import wandb as wandb_lib


def _parse_gate_spec(args) -> list[dict]:
    """Collect gate specs from --gate_step and repeatable --gate flags.

    Returns a list of ``GateStopCallback``-compatible kwargs. The legacy
    ``--gate_step/--gate_metric/--gate_threshold/--gate_goal`` flags form one
    spec; each ``--gate "step,metric,threshold[,goal]"`` forms another.
    """
    specs: list[dict] = []
    if args.gate_step:
        specs.append(
            {
                "check_step": int(args.gate_step),
                "metric": args.gate_metric,
                "threshold": float(args.gate_threshold),
                "goal": args.gate_goal,
            }
        )
    for raw in args.gate or []:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(
                f"--gate expects 'step,metric,threshold[,goal]', got {raw!r}"
            )
        spec = {
            "check_step": int(parts[0]),
            "metric": parts[1],
            "threshold": float(parts[2]),
        }
        if len(parts) == 4:
            goal = parts[3]
            if goal not in ("below", "above"):
                raise ValueError(
                    f"--gate goal must be 'below' or 'above', got {goal!r}"
                )
            spec["goal"] = goal
        specs.append(spec)
    return specs


def _load_gates_config(path: str | None) -> list[dict]:
    """Load gate specs from a YAML file with a ``gates:`` list."""
    if not path:
        return []
    from pathlib import Path

    data = yaml.safe_load(Path(path).read_text())
    gates = data.get("gates", []) if isinstance(data, dict) else []
    print(f"[config] loaded gates config {path}")
    print(f"[config]   gates: {gates}")
    specs: list[dict] = []
    for entry in gates:
        spec = {
            "check_step": int(entry["step"]),
            "metric": entry.get("metric", "win_rate"),
            "threshold": float(entry.get("threshold", 0.05)),
            "goal": entry.get("goal", "below"),
        }
        if spec["goal"] not in ("below", "above"):
            raise ValueError(
                f"gate goal must be 'below' or 'above', got {spec['goal']!r}"
            )
        specs.append(spec)
    return specs


def _seed_rng(seed: int | None) -> None:
    """Seed Python / NumPy / PyTorch RNGs for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Generic training / sweep orchestrator.

    Game-specific construction (network, callbacks, normalizers) is provided
    via singleton factories, so this class lives in ``rl_tools`` without
    importing the game package.
    """

    DEFAULT_LR = 3e-4
    DEFAULT_ROLLOUT_SIZE = 256

    @staticmethod
    def load_sweep_config(path: str) -> dict:
        """Load a W&B sweep config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        print(f"[config] loaded sweep config {path}")
        print(f"[config]   {data}")
        return data

    def __init__(
        self,
        args,
        *,
        network_factory: NetworkFactory,
        callbacks_factory: CallbacksFactory,
        normalizers_factory: NormalizersFactory,
    ) -> None:
        _seed_rng(getattr(args, "seed", None))
        self.args = args
        self.network_factory = network_factory
        self.callbacks_factory = callbacks_factory
        self.normalizers_factory = normalizers_factory
        self.initializer = RLInitializer(args)

    def run(
        self,
        overrides: dict | None = None,
        wandb_run=None,
        initializer: RLInitializer | None = None,
    ) -> None:
        args = self.args
        initializer = initializer or self.initializer
        base = getattr(args, "hyperparams", None) or {}
        params = {**base, **(overrides or {})}
        lr = params.pop("lr", self.DEFAULT_LR)
        tensorboard_proc = None

        try:
            connectors = initializer.start_instances(
                n=args.instances,
                id_offset=0,
                render=args.render,
                role="train",
            )

            from rl_tools.rl.Environment.Environment import Environment

            envs = [
                Environment(
                    connector,
                    seed_mode="train",
                    seed_base=args.seed,
                    env_index=i,
                    n_parallel=args.instances,
                )
                for i, connector in enumerate(connectors)
            ]

            network = self.network_factory.build()
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
            tensorboard_writer = None
            if not args.no_tensorboard:
                log_dir = f"{initializer.log_path}/tensorboard"
                tensorboard_writer = SummaryWriter(log_dir=log_dir)
                if args.tensorboard_port > 0:
                    tensorboard_proc = subprocess.Popen(
                        [
                            "tensorboard",
                            "--logdir",
                            log_dir,
                            "--port",
                            str(args.tensorboard_port),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    initializer.main_logger.info(
                        f"TensorBoard serving at http://localhost:{args.tensorboard_port}"
                    )
            elif args.tensorboard_port > 0:
                initializer.main_logger.warning(
                    "--tensorboard_port is ignored because --no_tensorboard is set"
                )
            if wandb_run is None and args.wandb_project:
                wandb_run = self._init_wandb()
            if wandb_run is not None and params:
                wandb_run.config.update(params)

            callbacks = [
                *self.callbacks_factory.build(),
                TimingCallback(),
            ]
            if args.eval_every_timesteps:
                if args.eval_instances < 1:
                    raise ValueError(
                        "--eval_instances must be >= 1 when eval is enabled"
                    )
                if args.eval_episodes < 1:
                    raise ValueError(
                        "--eval_episodes must be >= 1 when eval is enabled"
                    )
                eval_connectors = initializer.start_instances(
                    n=args.eval_instances,
                    id_offset=args.instances,
                    render=args.render or args.render_eval,
                    role="eval",
                )
                eval_envs = [
                    Environment(
                        connector,
                        seed_mode="eval",
                        env_index=j,
                        n_parallel=args.eval_instances,
                    )
                    for j, connector in enumerate(eval_connectors)
                ]
                callbacks.append(
                    EvalCallback(
                        envs=eval_envs,
                        every_timesteps=args.eval_every_timesteps,
                        n_episodes=args.eval_episodes,
                        deterministic=args.deterministic_eval,
                    )
                )
            # Metric-based stopping reads eval/latest from the blackboard, so it
            # must come after EvalCallback within the same update cycle.
            if args.stop_metric:
                callbacks.append(
                    MetricStopCallback(
                        metric=args.stop_metric,
                        threshold=args.stop_threshold,
                        patience=args.stop_patience,
                    )
                )
            if args.gate_step:
                callbacks.append(
                    GateStopCallback(
                        check_step=args.gate_step,
                        metric=args.gate_metric,
                        threshold=args.gate_threshold,
                        goal=args.gate_goal,
                    )
                )
            for spec in _parse_gate_spec(args):
                callbacks.append(GateStopCallback(**spec))
            gates_specs = _load_gates_config(args.gates_config)
            if gates_specs:
                initializer.main_logger.info(
                    "[config] loaded gates config %s: %s",
                    args.gates_config,
                    gates_specs,
                )
            for spec in gates_specs:
                callbacks.append(GateStopCallback(**spec))
            callbacks.extend(
                [
                    KeyStopCallback(key="q"),
                    NetworkSaveCallback(
                        save_path=f"{initializer.log_path}/checkpoints/final.pt",
                        save_every_updates=args.save_every_updates,
                    ),
                ]
            )
            # Sinks consume the agent blackboard. Order matters: EvalCallback runs
            # its eval inside on_update_end, so sinks must come after it to pick up
            # eval metrics in the same update cycle.
            sinks = [ConsoleCallback()]
            if not args.no_tensorboard:
                sinks.append(TensorboardCallback(tensorboard_writer))
            if wandb_run is not None:
                sinks.append(WandbCallback(wandb_run))
            callbacks.extend(sinks)
            callback = CallbackList(callbacks)

            obs_normalizer, rew_normalizer = self.normalizers_factory.build(args)
            rollout_size = params.pop("rollout_size", self.DEFAULT_ROLLOUT_SIZE)
            if args.max_steps:
                iterations = math.ceil(args.max_steps / (rollout_size * len(envs)))
            else:
                iterations = args.iterations
            agent = PPOAgent(
                network=network,
                optimizer=optimizer,
                envs=envs,
                rollout_size=rollout_size,
                callback=callback,
                observation_normalizer=obs_normalizer,
                reward_normalizer=rew_normalizer,
                **params,
            )
            if args.checkpoint:
                agent.load(
                    args.checkpoint,
                    load_optimizer=not args.no_load_optimizer,
                    load_rng=not args.no_load_rng,
                )
            agent.train(iterations=iterations)

        except KeyboardInterrupt:
            initializer.main_logger.info("Interrupted — shutting down...")
        except Exception as e:
            initializer.main_logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            if tensorboard_proc is not None:
                tensorboard_proc.terminate()
            initializer.stop_instances()

    def sweep(self, sweep_config: dict) -> str:
        if not self.args.wandb_project:
            raise ValueError("--wandb_project is required for sweeps")
        if self.args.wandb_mode in ("offline", "disabled"):
            raise ValueError(
                f"Sweeps require the W&B backend; --wandb_mode={self.args.wandb_mode} "
                "is not supported (use --wandb_mode online or a self-hosted server)"
            )
        self.initializer.main_logger.info(
            "[config] loaded sweep config %s: %s",
            self.args.sweep_config,
            sweep_config,
        )
        wrapper = WandbWrapper(project=self.args.wandb_project)

        trial_index = 0

        def _run() -> None:
            nonlocal trial_index
            _seed_rng(self.args.seed)
            trial_dir = os.path.join(
                self.initializer.log_path, "trials", f"trial_{trial_index}"
            )
            trial_index += 1
            trial = RLInitializer(self.args, log_path=trial_dir)
            run = wandb_lib.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                mode=self.args.wandb_mode,
                dir=trial_dir,
            )
            overrides = {
                k: v
                for k, v in run.config.items()
                if isinstance(v, (int, float, str, bool))
            }
            self.run(overrides, wandb_run=run, initializer=trial)

        sweep_id = wrapper.sweep(
            sweep_config,
            _run,
            count=self.args.sweep_count,
            entity=self.args.wandb_entity,
        )
        return sweep_id

    def _init_wandb(self):
        args = self.args
        wandb_dir = os.path.abspath(self.initializer.log_path)
        os.makedirs(wandb_dir, exist_ok=True)
        tags = (
            [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        )
        wrapper = WandbWrapper(
            project=args.wandb_project,
            rl_agent_params={},
            optimizer_params={"lr": self.DEFAULT_LR},
            game_params={
                "instances": args.instances,
                "seed": args.seed,
                "iterations": args.iterations,
                "max_steps": args.max_steps,
                "checkpoint": args.checkpoint,
                "eval_every_timesteps": args.eval_every_timesteps,
                "eval_instances": args.eval_instances,
                "eval_episodes": args.eval_episodes,
                "building_names": list(self.network_factory.building_names),
                "no_obs_norm": args.no_obs_norm,
                "no_reward_norm": args.no_reward_norm,
            },
            dir=wandb_dir,
            entity=args.wandb_entity,
            name=args.wandb_name or os.path.basename(self.initializer.log_path),
            tags=tags,
            mode=args.wandb_mode,
        )
        wandb_run = wrapper.init()
        self.initializer.main_logger.info(
            f"W&B run '{wandb_run.name}' ({args.wandb_mode}) id={wandb_run.id}, "
            f"dir={wandb_dir}"
        )
        return wandb_run
