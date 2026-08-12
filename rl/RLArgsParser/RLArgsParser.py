import argparse
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
            "--seed",
            type=int,
            default=None,
            help=(
                "Train map seed base: env i episode e uses seed+i+e*n_train. "
                "Eval uses negative decreasing seeds -(1+j+e*n_eval)."
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
            "--wandb_mode",
            type=str,
            choices=["offline", "online", "disabled"],
            default="offline",
            help="W&B run mode (offline writes locally to logs/<ts>/wandb, sync later with `wandb sync`)",
        )
        return parser.parse_args()
