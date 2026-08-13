import logging
from subprocess import TimeoutExpired

from rl_tools.game_engine.ActionInterface import ActionInterface
from rl_tools.game_engine.ObservationInterface import ObservationInterface
from rl_tools.game_engine.HeadlessGameEngine.HeadlessGameEngineFactory import (
    HeadlessGameEngineFactory,
)
from rl_tools.game_engine.HeadlessGameEngine import HeadlessGameEngine

MAX_RETRIES = 5


class GameEnvConnector:
    def __init__(
        self,
        instance_id: int,
        action_interface: ActionInterface,
        observation_interface: ObservationInterface,
        game_engine_type: HeadlessGameEngine.GameEngineType | None = None,
        game_engine_args: list | None = None,
        game_engine_kwargs: dict | None = None,
        logger: logging.Logger | None = None,
        log_path: str | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger(f"Instance-{instance_id}")
        self.instance_id = instance_id
        self.action_interface = action_interface
        self.observation_interface = observation_interface
        self._retries = 0
        self.game_engine: HeadlessGameEngine = self._start_game_engine(
            game_engine_type,
            game_engine_args,
            game_engine_kwargs,
            log_path=log_path,
            **kwargs,
        )

    @staticmethod
    def retry(func):
        def wrapper(self, *args, **kwargs):
            for _ in range(MAX_RETRIES):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Error during '{func.__name__}': {e}. Retrying..."
                    )
            self.logger.error(
                f"Maximum retries reached for '{func.__name__}'. Exiting."
            )
            exit(1)

        return wrapper

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Terminate the game engine process and close all interfaces.

        Idempotent: safe to call multiple times. Must be called so UDP
        sockets are released and the port can be reused by a later instance
        (e.g. the next sweep trial in the same process).
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.game_engine and self.game_engine.process:
            self.logger.info("Terminating game engine process...")
            self.game_engine.process.terminate()
            try:
                self.game_engine.process.wait(timeout=5)
                self.logger.info("Game engine process terminated successfully.")
            except TimeoutExpired:
                self.logger.warning(
                    "Game engine process did not terminate in time. Killing it."
                )
                self.game_engine.process.kill()
        self.action_interface.close()
        self.observation_interface.close()

    def connect(self):
        self._test_connection()
        self._notify_start()

    def _test_connection(self) -> bool:
        self.logger.info("Testing connection...")
        observation = self.observation_interface.get_raw_message()
        if not observation == b"ENV_READY":
            return False

        self._send_action(b"TRAINER_READY")
        observation = self.observation_interface.get_raw_message()
        if not observation == b"TRAINER_READY_ACK":
            return False
        self.logger.info("Connection test successful.")
        return True

    def _notify_start(self):
        self.logger.info("Notifying environment of readiness...")
        ready_action = b"START_TRAINING"
        self._send_action(ready_action)

    def step(self, action) -> tuple[dict, float, bool, dict]:
        self._send_action(action)
        observation = self._get_observation()
        obs, reward, done, info = self._split_observation(observation)
        return obs, reward, done, info

    def reset(self, seed: int | None = None) -> dict:
        if seed is None:
            self.logger.info("Resetting environment...")
        else:
            self.logger.info(f"Resetting environment with seed={seed}...")
        self._reset_environment(seed=seed)
        observation = self._get_observation()
        obs, _, _, _ = self._split_observation(observation)
        return obs

    def _start_game_engine(
        self,
        game_engine_type: HeadlessGameEngine.GameEngineType | None = None,
        game_engine_args: list | None = None,
        game_engine_kwargs: dict | None = None,
        log_path: str | None = None,
        **kwargs,
    ) -> HeadlessGameEngine:
        self.logger.info("Starting game engine...")
        if game_engine_kwargs is None:
            game_engine_kwargs = {}
        return HeadlessGameEngineFactory().create(
            game_engine_type,
            instance_id=self.instance_id,
            run_args=game_engine_args,
            run_kwargs=game_engine_kwargs,
            log_path=log_path,
            **kwargs,
        )

    def _split_observation(self, observation: dict) -> tuple:
        """Parse the raw observation data if necessary."""
        obs = {
            "observation": observation.get("observation", {}),
            "action_mask": observation.get("action_mask", {}),
        }
        reward = observation.get("reward", 0.0)
        done = observation.get("done", False)
        info = observation.get("info", {})
        return obs, reward, done, info

    @retry
    def _get_observation(self) -> dict:
        """Retrieve the current observation from the environment."""
        observation = self.observation_interface.get_observation()
        if observation is None:
            raise ConnectionError("Failed to retrieve observation.")
        return observation

    def _send_action(self, action):
        """Send an action to the environment."""
        self.action_interface.send_action(action)

    @retry
    def _reset_environment(self, seed: int | None = None):
        """Reset the environment. Optional seed becomes RESET:<seed> for map RNG."""
        if seed is None:
            self.action_interface.send_action(b"RESET")
        else:
            self.action_interface.send_action(f"RESET:{int(seed)}".encode())
        reset_ack = self.observation_interface.get_raw_message()
        self.logger.debug(f"Received message after reset: {reset_ack}")
        if not reset_ack == b"RESET_ACK":
            raise ConnectionError("Did not receive RESET_ACK from environment.")
        self.logger.info("Environment reset successfully.")
