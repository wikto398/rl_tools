import multiprocessing as mp
import argparse
import importlib
import logging
import os

from datetime import datetime
from types import SimpleNamespace
from rl_tools.game_engine.ObservationInterface.UDPObservation import (
    UDPObservation,
)
from rl_tools.game_engine.ActionInterface.UDPAction import UDPAction
from rl_tools.game_engine.GameEnvConnector import GameEnvConnector
from rl_tools.game_engine.HeadlessGameEngine import HeadlessGameEngine

CONFIG = SimpleNamespace(
    PYTHON_HOST="127.0.0.1",
    GODOT_HOST="127.0.0.1",
    OBSERVATION_RECEIVER_PORT=5000,
    ACTION_RECEIVER_PORT=5500,
    INSTANCES=2,
)


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
        "--observation_class",
        type=str,
        default=None,
        help=(
            "Dotted path of the observation class to use, e.g. "
            "torch_files.StrategyUDPObservation.StrategyUDPObservation "
            "(default: the generic rl_tools UDPObservation / MessagePack)"
        ),
    )
    return parser.parse_args()


def resolve_observation_class(spec: str | None):
    if not spec:
        return UDPObservation
    module_path, _, attr = spec.rpartition(".")
    if not module_path or not attr:
        raise argparse.ArgumentTypeError(
            f"invalid --observation_class {spec!r}: expected 'module.path.Attr'"
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def setup_instance_logger(
    instance_id: int, log_path: str, *, role: str = "train"
) -> logging.Logger:
    logger = logging.getLogger(f"Instance-{instance_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = (
        False  # Prevent log messages from being propagated to the root logger
    )

    # Avoid adding duplicate handlers on re-init
    if logger.handlers:
        logger.handlers.clear()

    connector_dir = os.path.join(log_path, role, "game_connector")
    os.makedirs(connector_dir, exist_ok=True)
    log_file = os.path.join(connector_dir, f"instance_{instance_id}.log")

    formatter = logging.Formatter(
        fmt="[%(levelname)s] | %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — per instance
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def start_instance(
    instance_id: int = 0,
    args: argparse.Namespace | None = None,
    log_path=None,
    role: str = "train",
    observation_class=UDPObservation,
) -> GameEnvConnector:
    run_log_path = log_path or "logs"
    logger = setup_instance_logger(instance_id, run_log_path, role=role)
    logger.info(f"Starting instance {instance_id} with args: {args}")
    observation_port = CONFIG.OBSERVATION_RECEIVER_PORT + instance_id
    action_port = CONFIG.ACTION_RECEIVER_PORT + instance_id
    udp_observer = observation_class(
        ip=CONFIG.PYTHON_HOST, port=observation_port, logger=logger
    )

    udp_action_sender = UDPAction(ip=CONFIG.GODOT_HOST, port=action_port, logger=logger)
    game_engine_kwargs = {
        "action_receiver_port": action_port,
        "observation_receiver_port": observation_port,
    }
    for key, value in vars(args).items():
        if key == "observation_class":
            continue
        game_engine_kwargs[key] = value
    engine_log_dir = os.path.join(run_log_path, role, "headless_game_engine")
    os.makedirs(engine_log_dir, exist_ok=True)
    game_env_connector = GameEnvConnector(
        instance_id=instance_id,
        action_interface=udp_action_sender,
        observation_interface=udp_observer,
        game_engine_type=(
            HeadlessGameEngine.GameEngineType(args.game_engine_type) if args else None
        ),
        game_engine_args=None,
        game_engine_kwargs=game_engine_kwargs,
        logger=logger,
        log_path=engine_log_dir,
    )
    game_env_connector.connect()
    for step in range(10):
        logger.info(f"Instance {instance_id} - Step {step}")
        obs, reward, done, info = game_env_connector.step([1, 1, 1])
        logger.info(
            f"Instance {instance_id} - Received observation: {obs}, reward: {reward}, done: {done}, info: {info}"
        )
    game_env_connector.reset()
    for step in range(10):
        logger.info(f"Instance {instance_id} - Step {step}")
        obs, reward, done, info = game_env_connector.step([1, 1, 1])
        logger.info(
            f"Instance {instance_id} - Received observation: {obs}, reward: {reward}, done: {done}, info: {info}"
        )
    return game_env_connector


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"logs/{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main_logger = logging.getLogger("Main")
    file_handler = logging.FileHandler(f"{log_path}/main.log")
    file_handler.setFormatter(
        logging.Formatter(
            fmt="[%(levelname)s] | %(asctime)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    main_logger.addHandler(file_handler)
    main_logger.info("RL Tools started with configuration: %s", vars(args))
    main_logger.info("Starting RL Tools with %d instances", args.instances)
    if args.kill_existing:
        main_logger.info("Killing existing game engine instances...")
        HeadlessGameEngine.kill_existing_instances(args.game_engine_type)
    if args.instances < 1:
        main_logger.error("Number of instances must be at least 1.")
        exit(1)
    else:
        CONFIG.INSTANCES = args.instances
    instances = []
    observation_class = resolve_observation_class(args.observation_class)
    for i in range(CONFIG.INSTANCES):
        p = mp.Process(
            target=start_instance,
            kwargs={
                "instance_id": i,
                "args": args,
                "log_path": log_path,
                "role": "train",
                "observation_class": observation_class,
            },
        )
        p.start()
        instances.append(p)

    for instance in instances:
        instance.join()


if __name__ == "__main__":
    main()
