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
            help="Enable rendering of the game environment (if supported by the game engine)",
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
        return parser.parse_args()
