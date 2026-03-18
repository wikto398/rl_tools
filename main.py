import multiprocessing as mp
import argparse
import logging
import os

from datetime import datetime
from types import SimpleNamespace
from game_engine.ObservationInterface.UDPObservation.UDPObservation import (
    UDPObservation,
)
from game_engine.ActionInterface.UDPAction.UDPAction import UDPAction
from game_engine.GameEnvConnector.GameEnvConnector import GameEnvConnector

CONFIG = SimpleNamespace(
    PYTHON_HOST="127.0.0.1",
    GODOT_HOST="127.0.0.1",
    OBSERVATION_RECEIVER_PORT=5000,
    ACTION_RECEIVER_PORT=5500,
    INSTANCES=2,
)

def parse_args():
    parser = argparse.ArgumentParser(description="RL Tools - A framework for training reinforcement learning agents with custom game environments.")
    parser.add_argument("--log_level", type=str, choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level for the application", default="INFO")
    parser.add_argument("--instances", type=int, help="Number of instances to run", default=CONFIG.INSTANCES)
    parser.add_argument("--log_to_file", action="store_true", help="Enable logging to a file")
    return parser.parse_args()


def setup_instance_logger(instance_id: int) -> logging.Logger:
    logger = logging.getLogger(f"Instance-{instance_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent log messages from being propagated to the root logger

    # Avoid adding duplicate handlers on re-init
    if logger.handlers:
        logger.handlers.clear()

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"logs/{timestamp}/game_connector_instance_{instance_id}.log"

    formatter = logging.Formatter(
        fmt="[%(levelname)s] | %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — per instance
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger

def start_instance(instance_id: int = 0, args = None, log_path = None) -> GameEnvConnector:
    logger = setup_instance_logger(instance_id)
    logger.info(f"Starting instance {instance_id} with args: {args}")
    observation_port = CONFIG.OBSERVATION_RECEIVER_PORT + instance_id
    action_port = CONFIG.ACTION_RECEIVER_PORT + instance_id
    udp_observer = UDPObservation(
        ip=CONFIG.PYTHON_HOST, port=observation_port, logger=logger
    )

    udp_action_sender = UDPAction(
        ip=CONFIG.GODOT_HOST, port=action_port, logger=logger
    )
    game_engine_kwargs = {
        "action_receiver_port": action_port,
        "observation_receiver_port": observation_port,
    }
    for key, value in vars(args).items():
        game_engine_kwargs[key] = value
    game_envc_connector = GameEnvConnector(
        instance_id=instance_id,
        action_interface=udp_action_sender,
        observation_interface=udp_observer,
        game_engine_type=None,
        game_engine_args=None,
        game_engine_kwargs=game_engine_kwargs,
        logger=logger,
        log_path=log_path,
    )
    game_envc_connector.start()
    return game_envc_connector

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"logs/{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    main_logger = logging.getLogger("Main")
    file_handler = logging.FileHandler(f"{log_path}/main.log")
    file_handler.setFormatter(logging.Formatter(fmt="[%(levelname)s] | %(asctime)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    main_logger.addHandler(file_handler)
    main_logger.info("RL Tools started with configuration: %s", vars(args))
    main_logger.info("Starting RL Tools with %d instances", args.instances)
    if args.instances < 1:
        main_logger.error("Number of instances must be at least 1.")
        exit(1)
    else:
        CONFIG.INSTANCES = args.instances
    instances = []
    for i in range(CONFIG.INSTANCES):
        p = mp.Process(target=start_instance, kwargs={"instance_id": i, "args": args, "log_path": log_path})
        p.start()
        instances.append(p)

    for instance in instances:
        instance.join()


if __name__ == "__main__":
    main()
