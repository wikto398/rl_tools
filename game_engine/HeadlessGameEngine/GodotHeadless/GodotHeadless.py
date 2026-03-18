import subprocess
from os import path

from game_engine.HeadlessGameEngine.HeadlessGameEngine import HeadlessGameEngine

class GodotHeadless(HeadlessGameEngine):
    def __init__(self, instance_id: int, run_args: list | None = None, run_kwargs: dict | None = None, project_path: str | None = None, log_path: str | None = None, **kwargs):
        super().__init__(instance_id=instance_id, run_args=run_args, run_kwargs=run_kwargs, project_path=project_path, log_path=log_path, **kwargs)

    def start(self):
        print("Starting Godot headless game engine...")
        command = ["godot", "--path", self.project_path, "--headless"]
        if self.run_args:
            command.extend(self.run_args)
        if self.run_kwargs:
            for key, value in self.run_kwargs.items():
                command.append(f"--{key}={value}")
        try:
            print("Godot headless game engine command:", " ".join(command))
            with open(path.join(self.log_path, self.log_file_name), "w") as log_file:
                self.process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            # self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Godot headless game engine started successfully.")
        except Exception as e:
            print(f"Failed to start Godot headless game engine: {e}")
            exit(1)
