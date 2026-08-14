import argparse
from pathlib import Path

import yaml

from rl_tools.game_engine.HeadlessGameEngine import HeadlessGameEngine
from rl_tools.utils.config import CONFIG


class RLArgsParser:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description="RL Tools - A framework for training reinforcement learning agents with custom game environments."
        )
        parser.add_argument(
            "--log_level",
            type=str,
            choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
            help="Set the logging level for the application",
            default="INFO",
        )
        parser.add_argument(
            "--instances",
            type=int,
            help="Number of instances to run",
            default=CONFIG.INSTANCES,
        )
        parser.add_argument(
            "--log_to_file", action="store_true", help="Enable logging to a file"
        )
        parser.add_argument(
            "-g",
            "--game_engine_type",
            type=HeadlessGameEngine.GameEngineType,
            help="Type of game engine to use (e.g., 'Godot', 'Unity', 'Unreal')",
            default=HeadlessGameEngine.GameEngineType.GODOT,
        )
        parser.add_argument(
            "-k",
            "--kill_existing",
            action="store_true",
            help="Kill existing game engine instances before starting new ones",
        )
        parser.add_argument(
            "--render",
            action="store_true",
            help="Enable rendering for training (and eval) Godot instances",
        )
        parser.add_argument(
            "--render_eval",
            action="store_true",
            help="Enable rendering for eval Godot instances only",
        )
        parser.add_argument(
            "--eval_every_timesteps",
            type=int,
            default=None,
            help="Run evaluation every N global timesteps (disabled if omitted)",
        )
        parser.add_argument(
            "--eval_instances",
            type=int,
            default=1,
            help="Number of dedicated Godot instances used for evaluation",
        )
        parser.add_argument(
            "--eval_episodes",
            type=int,
            default=5,
            help="Number of full episodes to average per evaluation",
        )
        parser.add_argument(
            "--deterministic_eval",
            action="store_true",
            default=True,
            help="Evaluate using argmax (greedy) actions (default)",
        )
        parser.add_argument(
            "--stochastic_eval",
            action="store_false",
            dest="deterministic_eval",
            help="Evaluate using sampled actions (override --deterministic_eval)",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=1000,
            help="Number of training iterations to run (additional iterations when resuming)",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="Stop each run after ~N global steps (overrides --iterations). "
            "Derived as ceil(max_steps / (rollout_size * instances)).",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to a checkpoint .pt file to load before training",
        )
        parser.add_argument(
            "--save_every_updates",
            type=int,
            default=None,
            help="If set, also write a latest checkpoint every N completed updates",
        )
        parser.add_argument(
            "--no_load_optimizer",
            action="store_true",
            help="When loading --checkpoint, skip optimizer state",
        )
        parser.add_argument(
            "--no_load_rng",
            action="store_true",
            help="When loading --checkpoint, skip RNG states",
        )
        parser.add_argument(
            "--torch_compile",
            action="store_true",
            help=(
                "Wrap the network in torch.compile (requires a CUDA-capable "
                "device). Off by default: enables op fusion for lower forward "
                "latency but changes FP rounding (runs stay seed-reproducible "
                "only within the same compile mode)."
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Train map seed base: env i episode e uses seed+i+e*n_train. "
                "Eval uses negative decreasing seeds -(1+j+e*n_eval)."
            ),
        )
        parser.add_argument(
            "--expert_eps",
            type=float,
            default=0.0,
            help=(
                "Fraction of training steps where a scripted expert overrides the "
                "agent's action (expert-in-the-loop on-policy coach). All envs stay "
                "active; only the action is replaced. 0 disables the expert."
            ),
        )
        parser.add_argument(
            "--expert_eps_decay_steps",
            type=int,
            default=0,
            help=(
                "Linearly decay --expert_eps to 0 over this many global steps "
                "(0 = no decay). After decay the expert is inert and every env "
                "is pure agent training."
            ),
        )
        parser.add_argument(
            "--tensorboard_port",
            type=int,
            default=0,
            help="Port to serve TensorBoard on (0 = disabled)",
        )
        parser.add_argument(
            "--no_obs_norm",
            action="store_true",
            help="Disable RunningMeanStd observation normalization (on by default)",
        )
        parser.add_argument(
            "--no_reward_norm",
            action="store_true",
            help="Disable RunningMeanStd reward normalization (on by default)",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Only print ERROR logs to console and files; info/metrics go to TensorBoard only",
        )
        parser.add_argument(
            "--no_tensorboard",
            action="store_true",
            help="Disable the TensorBoard SummaryWriter (and --tensorboard_port server)",
        )
        parser.add_argument(
            "--csv_metrics",
            action="store_true",
            help="Dump all blackboard scalars to <log_path>/metrics.csv (long format: step,key,value)",
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            default=None,
            help="W&B project name; passing this enables W&B logging",
        )
        parser.add_argument(
            "--wandb_entity",
            type=str,
            default=None,
            help="W&B entity/team to scope the run to",
        )
        parser.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="W&B run name (default: the logs/<timestamp> directory basename)",
        )
        parser.add_argument(
            "--wandb_tags",
            type=str,
            default=None,
            help="Comma-separated W&B run tags",
        )
        parser.add_argument(
            "--wandb_group",
            type=str,
            default=None,
            help="W&B run group; runs sharing a group are grouped together in the dashboard",
        )
        parser.add_argument(
            "--wandb_mode",
            type=str,
            choices=["offline", "online", "disabled"],
            default="offline",
            help="W&B run mode (offline writes locally to logs/<ts>/wandb, sync later with `wandb sync`)",
        )
        parser.add_argument(
            "--sweep_config",
            type=str,
            default=None,
            help="Path to a YAML W&B sweep search space; presence enables sweep mode",
        )
        parser.add_argument(
            "--sweep_count",
            type=int,
            default=None,
            help="Maximum number of sweep trials to run (default: until stopped)",
        )
        parser.add_argument(
            "--sweep_entity",
            type=str,
            default=None,
            help="W&B entity to scope the sweep to (default: --wandb_entity)",
        )
        parser.add_argument(
            "--stop_metric",
            type=str,
            default=None,
            help="Stop training when this eval metric stays bad for --stop_patience consecutive evals (e.g. win_rate)",
        )
        parser.add_argument(
            "--stop_threshold",
            type=float,
            default=0.05,
            help="Threshold for --stop_metric pruning",
        )
        parser.add_argument(
            "--stop_patience",
            type=int,
            default=3,
            help="Consecutive bad evals before stopping (--stop_metric)",
        )
        parser.add_argument(
            "--gate_step",
            type=int,
            default=None,
            help="One-shot go/no-go step: stop the run if the metric is bad at this step",
        )
        parser.add_argument(
            "--gate_metric",
            type=str,
            default="win_rate",
            help="Metric evaluated at --gate_step",
        )
        parser.add_argument(
            "--gate_threshold",
            type=float,
            default=0.05,
            help="Threshold for --gate_step check",
        )
        parser.add_argument(
            "--gate_goal",
            type=str,
            choices=["below", "above"],
            default="below",
            help="Stop when metric is below/above the gate threshold",
        )
        parser.add_argument(
            "--gate",
            action="append",
            default=None,
            help="One-shot gate as 'step,metric,threshold[,goal]'. Repeatable for "
            "multiple gates (goal defaults to 'below').",
        )
        parser.add_argument(
            "--gates_config",
            type=str,
            default=None,
            help="Path to a YAML file listing multiple gates under a 'gates:' key",
        )
        parser.add_argument(
            "--engine_args",
            action="append",
            default=None,
            help="Extra key=value args forwarded verbatim to the game engine process. "
            "Repeatable (e.g. --engine_args my_flag=1 --engine_args other=2).",
        )
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to a YAML config file with 'hyperparams' and 'cli' sections "
            "(non-sweep runs only; explicit CLI flags override the file).",
        )
        args = parser.parse_args()
        if args.config:
            _apply_config(args, parser, args.config)
        return args


# Hyperparameter knobs forwarded to PPOAgent (mirrors the sweepable set + extras).
CONFIG_HYPERPARAM_KEYS = {
    "lr",
    "gamma",
    "lam",
    "epochs",
    "batch_size",
    "rollout_size",
    "entropy_coef_start",
    "entropy_coef_end",
    "entropy_coef_decay_steps",
    "entropy_target",
    "entropy_adapt_lr",
    "adaptive_entropy",
    "clip_epsilon",
    "value_loss_coef",
}


def _apply_config(args, parser, config_path: str) -> None:
    """Merge a YAML config into parsed args.

    ``cli`` keys are applied only to existing argparse dests and only when the
    CLI did not explicitly set them (CLI wins). ``hyperparams`` keys are stored
    on ``args.hyperparams`` for the Trainer. Unknown keys are ignored.
    """
    try:
        data = yaml.safe_load(Path(config_path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load config {config_path!r}: {e}") from e

    defaults = parser.parse_args([])

    cli = data.get("cli")
    effective_cli: dict = {}
    if isinstance(cli, dict):
        for key, value in cli.items():
            if key not in vars(args):
                print(f"[config] ignoring unknown cli key: {key}")
                continue
            if key == "config":
                continue
            # Only apply if the CLI left the value at its default.
            if getattr(args, key) == getattr(defaults, key):
                setattr(args, key, value)
                effective_cli[key] = value

    hyperparams = data.get("hyperparams")
    if isinstance(hyperparams, dict):
        unknown = set(hyperparams) - CONFIG_HYPERPARAM_KEYS
        for key in unknown:
            print(f"[config] ignoring unknown hyperparam key: {key}")
        args.hyperparams = {
            k: v for k, v in hyperparams.items() if k in CONFIG_HYPERPARAM_KEYS
        }
    else:
        args.hyperparams = {}

    # ``engine_args`` forwards game-side key=value args to Godot (reward
    # coefficients etc.). Config entries come first so explicit CLI
    # ``--engine_args`` still win (last-wins overwrite in both the engine
    # connector and Godot's ArgsParser).
    engine_args = data.get("engine_args")
    effective_engine_args: dict = {}
    if isinstance(engine_args, dict):
        config_entries = [f"{k}={v}" for k, v in engine_args.items()]
        existing = list(getattr(args, "engine_args", None) or [])
        args.engine_args = config_entries + existing
        effective_engine_args = dict(engine_args)

    print(f"[config] loaded {config_path}")
    print(f"[config]   cli:         {cli}")
    print(f"[config]   hyperparams: {hyperparams}")
    print(f"[config]   engine_args: {engine_args}")
    print(f"[config]   effective cli:         {effective_cli}")
    print(f"[config]   effective hyperparams: {args.hyperparams}")
    print(f"[config]   effective engine_args: {args.engine_args}")
    args.config_report = {
        "path": config_path,
        "cli": cli,
        "hyperparams": hyperparams,
        "engine_args": engine_args,
        "effective_cli": effective_cli,
        "effective_hyperparams": args.hyperparams,
        "effective_engine_args": effective_engine_args,
    }
