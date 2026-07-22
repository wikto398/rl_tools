import os
import enum

from abc import ABC, abstractmethod
import subprocess


class HeadlessGameEngine(ABC):
    class GameEngineType(enum.Enum):
        GODOT = "godot"

    def __init__(
        self,
        instance_id: int,
        run_args: list | None = None,
        run_kwargs: dict | None = None,
        project_path: str | None = None,
        log_path: str | None = None,
        log_file_name: str | None = None,
        **kwargs,
    ):
        self.instance_id = instance_id
        self.run_args = run_args
        self.run_kwargs = run_kwargs
        self.project_path = project_path if project_path else "."
        self.log_path = log_path if log_path else "logs"
        self.log_file_name = (
            log_file_name if log_file_name else f"instance_{self.instance_id}.log"
        )
        self.kwargs = kwargs
        self.process: subprocess.Popen
        self.start()

    @abstractmethod
    def start(self):
        """Start the headless game engine."""
        pass

    @staticmethod
    def kill_existing_instances(game_engine_type: GameEngineType):
        """Kill existing instances of the headless game engine."""
        if game_engine_type == HeadlessGameEngine.GameEngineType.GODOT:
            os.system("pkill -f 'godot.*--headless'")
