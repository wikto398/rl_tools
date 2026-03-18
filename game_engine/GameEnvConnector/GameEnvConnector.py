import logging

from game_engine.ActionInterface.ActionInterface import ActionInterface
from game_engine.ObservationInterface.ObservationInterface import ObservationInterface
from game_engine.HeadlessGameEngine.HeadlessGameEngineFactory.HeadlessGameEngineFactory import HeadlessGameEngineFactory
from game_engine.HeadlessGameEngine.HeadlessGameEngine import HeadlessGameEngine
from time import sleep

MAX_RETRIES = 5

class GameEnvConnector:
    def __init__(self, instance_id: int, action_interface: ActionInterface, observation_interface: ObservationInterface, game_engine_type: HeadlessGameEngine.GameEngineType | None = None, game_engine_args: list | None = None, game_engine_kwargs: dict | None = None, logger: logging.Logger | None = None, log_path: str | None = None, **kwargs):
        self.logger = logger or logging.getLogger(f"Instance-{instance_id}")
        self.instance_id = instance_id
        self.action_interface = action_interface
        self.observation_interface = observation_interface
        self._action = None
        self._retries = 0
        self.game_engine: HeadlessGameEngine = self.start_game_engine(game_engine_type, game_engine_args, game_engine_kwargs, log_path=log_path, **kwargs)

    def start(self):
        self.test_connection()
        self.notify_start()
        while True:
            self.train_step()

    def test_connection(self) -> bool:
        self.logger.info("Testing connection...")
        observation = self.observation_interface.get_observation()
        if not observation == b"ENV_READY":
            return False

        self.action_interface.send_action(b"TRAINER_READY")
        observation = self.observation_interface.get_observation()
        if not observation == b"TRAINER_READY_ACK":
            return False
        self.logger.info("Connection test successful.")
        return True

    def notify_start(self):
        self.logger.info("Notifying environment of readiness...")
        ready_action = b"START_TRAINING"
        self.action_interface.send_action(ready_action)

    def train_step(self):
        observation = self.observation_interface.get_observation()
        observation = self.observation_interface.parse_observation(observation) if observation is not None else None
        self.logger.info(f"Current observation: {observation}")
        # Here you would typically process the observation and decide on an action
        if observation is None:
            self.logger.warning("Connection might be lost. Attempting to resend previous action...")
            self._retries += 1
            if self._retries > MAX_RETRIES:
                self.logger.error("Maximum retries reached. Exiting.")
                exit(1)
        else:
            self._retries = 0
            self._action = self.decide_action(observation)
        self.action_interface.send_action(self._action)

    def decide_action(self, observation):
        sleep(1)
        return [0, 0, 0]

    def start_game_engine(self, game_engine_type: HeadlessGameEngine.GameEngineType | None = None, game_engine_args: list | None = None, game_engine_kwargs: dict | None = None, log_path: str | None = None, **kwargs) -> HeadlessGameEngine:
        self.logger.info("Starting game engine...")
        if game_engine_kwargs is None:
            game_engine_kwargs = {}
        return HeadlessGameEngineFactory().create(game_engine_type, instance_id=self.instance_id, run_args=game_engine_args, run_kwargs=game_engine_kwargs, log_path=log_path, **kwargs)
