from game_engine.ActionInterface.ActionInterface import ActionInterface
from game_engine.ObservationInterface.ObservationInterface import ObservationInterface
from time import sleep

MAX_RETRIES = 5

class GameEnvConnector:
    current_id: int = 0
    def __init__(self, action_interface: ActionInterface, observation_interface: ObservationInterface):
        self.id = GameEnvConnector.current_id
        GameEnvConnector.current_id += 1
        self.action_interface = action_interface
        self.observation_interface = observation_interface
        self.game_engine_process = None
        self._action = None
        self._retries = 0

    def start(self):
        self.test_connection()
        self.notify_start()
        while True:
            self.train_step()

    def test_connection(self):
        print("Testing connection...")
        test_action = b"TRAINER_READY"
        self.action_interface.send_action(test_action)
        observation = self.observation_interface.get_observation()
        if observation == b"ENV_READY":
            print("Connection test successful.")
        else:
            print("Connection test failed.")
            exit(1)
    
    def notify_start(self):
        print("Notifying environment of readiness...")
        ready_action = b"START_TRAINING"
        self.action_interface.send_action(ready_action)

    def train_step(self):
        observation = self.observation_interface.get_observation()
        print(f"Current observation: {observation}")
        # Here you would typically process the observation and decide on an action
        if observation is None:
            print("Connection might be lost. Attempting to resend previous action...")
            self._retries += 1
            if self._retries > MAX_RETRIES:
                print("Maximum retries reached. Exiting.")
                exit(1)
        else:
            self._retries = 0
            self._action = self.decide_action(observation)
        self.action_interface.send_action(self._action)


    def decide_action(self, observation):
        sleep(1)
        return b"Sample action based on observation"
    