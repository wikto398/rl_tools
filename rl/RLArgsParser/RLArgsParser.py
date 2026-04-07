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
        return parser.parse_args()
