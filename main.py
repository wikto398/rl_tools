import os
import subprocess

from game_engine.ObservationInterface.UDPObservation.UDPObservation import UDPObservation
from dotenv import load_dotenv
from game_engine.ActionInterface.UDPAction.UDPAction import UDPAction
from game_engine.GameEnvConnector.GameEnvConnector import GameEnvConnector

def main():
    load_dotenv(dotenv_path="../.env")
    udp_observer = UDPObservation(ip=os.getenv("PYTHON_HOST"), port=int(os.getenv("OBSERVATION_RECEIVER_PORT")))
    
    udp_action_sender = UDPAction(ip=os.getenv("GODOT_HOST"), port=int(os.getenv("ACTION_RECEIVER_PORT")))
    env_connector = GameEnvConnector(action_interface=udp_action_sender, observation_interface=udp_observer)

    env_connector.start()

if __name__ == "__main__":
    main()