import argparse
import logging
import os

from datetime import datetime
from rl_tools.game_engine.ObservationInterface.UDPObservation import UDPObservation
from rl_tools.rl.RLArgsParser import RLArgsParser
from rl_tools.utils.config import CONFIG
from rl_tools.game_engine.ActionInterface.UDPAction import UDPAction
from rl_tools.game_engine.GameEnvConnector import GameEnvConnector
from rl_tools.game_engine.HeadlessGameEngine import HeadlessGameEngine

# Args forwarded to the game engine process. Everything else in the CLI
# namespace is Trainer-only and stays out of the engine's command line.
GODOT_ARG_ALLOWLIST = {"log_level", "log_to_file", "seed", "render"}


class RLInitializer:
    def __init__(
        self,
        args: argparse.Namespace,
        log_path: str | None = None,
        observation_class=UDPObservation,
    ):
        self.args = args
        self.observation_class = observation_class
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = log_path or f"logs/{self.timestamp}"
        os.makedirs(self.log_path, exist_ok=True)
        self.quiet = bool(getattr(args, "quiet", False))
        console_level = (
            logging.ERROR
            if self.quiet
            else getattr(logging, args.log_level)
            if hasattr(logging, args.log_level)
            else logging.DEBUG
        )
        logging.basicConfig(
            level=console_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.main_logger = logging.getLogger("Main")
        self._file_handler = None
        for handler in list(self.main_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                self.main_logger.removeHandler(handler)
                handler.close()
        self._file_handler = logging.FileHandler(f"{self.log_path}/main.log")
        self._file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(levelname)s] | %(asctime)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.main_logger.addHandler(self._file_handler)
        self.main_logger.info("RL Tools started with configuration: %s", vars(args))
        report = getattr(args, "config_report", None)
        if report:
            self.main_logger.info("[config] loaded %s", report["path"])
            self.main_logger.info("[config]   cli:         %s", report["cli"])
            self.main_logger.info("[config]   hyperparams: %s", report["hyperparams"])
            self.main_logger.info(
                "[config]   effective cli:         %s", report["effective_cli"]
            )
            self.main_logger.info(
                "[config]   effective hyperparams: %s",
                report["effective_hyperparams"],
            )
        self.main_logger.info("Starting RL Tools with %d instances", args.instances)
        if args.kill_existing:
            self.main_logger.info("Killing existing game engine instances...")
            HeadlessGameEngine.kill_existing_instances(args.game_engine_type)
        if args.instances < 1:
            self.main_logger.error("Number of instances must be at least 1.")
            exit(1)
        else:
            CONFIG.INSTANCES = args.instances
        self.connectors: list[GameEnvConnector] = []

    def start_instances(
        self,
        n: int | None = None,
        *,
        id_offset: int = 0,
        render: bool | None = None,
        role: str = "train",
    ) -> list[GameEnvConnector]:
        """Start n instances and return the newly started connectors."""
        count = self.args.instances if n is None else n
        if count < 1:
            raise ValueError(f"n must be at least 1, got {count}")
        if role not in ("train", "eval"):
            raise ValueError(f"role must be 'train' or 'eval', got {role!r}")
        started: list[GameEnvConnector] = []
        for i in range(count):
            instance_id = id_offset + i
            self.main_logger.info(f"Starting {role} instance {instance_id}...")
            connector = start_instance(
                instance_id=instance_id,
                local_index=i,
                args=self.args,
                log_path=self.log_path,
                render=render,
                role=role,
                observation_class=self.observation_class,
            )
            self.connectors.append(connector)
            started.append(connector)
            self.main_logger.info(f"Instance {instance_id} connected and ready")
        return started

    def stop_instances(self):
        """Disconnect and close all instances cleanly.

        Closes each connector (terminating its game engine process and its
        UDP sockets) so ports can be reused by later instances — e.g. the
        next sweep trial running in the same process.
        """
        for i, connector in enumerate(self.connectors):
            try:
                connector.close()
                self.main_logger.info(
                    f"Instance {i} closed (exit code {connector.game_engine.process.returncode})"
                )
            except Exception as e:
                self.main_logger.error(f"Error closing instance {i}: {e}")
        self.connectors.clear()
        if self._file_handler is not None:
            try:
                self.main_logger.removeHandler(self._file_handler)
                self._file_handler.close()
            except Exception:
                pass
            self._file_handler = None


def setup_instance_logger(
    instance_id: int, log_path: str, *, role: str = "train", quiet: bool = False
) -> logging.Logger:
    logger = logging.getLogger(f"Instance-{instance_id}")
    logger.setLevel(logging.ERROR if quiet else logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    connector_dir = os.path.join(log_path, role, "game_connector")
    os.makedirs(connector_dir, exist_ok=True)
    log_file = os.path.join(connector_dir, f"instance_{instance_id}.log")

    formatter = logging.Formatter(
        fmt="[%(levelname)s] | %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def start_instance(
    instance_id: int = 0,
    local_index: int = 0,
    args: argparse.Namespace | None = None,
    log_path: str = "logs",
    render: bool | None = None,
    role: str = "train",
    observation_class=UDPObservation,
) -> GameEnvConnector:
    """Start a single instance, connect it, and return the connector."""
    quiet = bool(args is not None and getattr(args, "quiet", False))
    logger = setup_instance_logger(instance_id, log_path, role=role, quiet=quiet)
    logger.info(f"Starting instance {instance_id} with args: {args}")

    observation_port = CONFIG.OBSERVATION_RECEIVER_PORT + instance_id
    action_port = CONFIG.ACTION_RECEIVER_PORT + instance_id

    udp_observer = observation_class(
        ip=CONFIG.PYTHON_HOST,
        port=observation_port,
        logger=logger,
    )
    udp_action_sender = UDPAction(
        ip=CONFIG.GODOT_HOST,
        port=action_port,
        logger=logger,
    )

    game_engine_kwargs = {
        "action_receiver_port": action_port,
        "observation_receiver_port": observation_port,
    }
    if args:
        for key, value in vars(args).items():
            if key in GODOT_ARG_ALLOWLIST and value is not None:
                game_engine_kwargs[key] = value
        for raw in getattr(args, "engine_args", None) or []:
            key, _, value = raw.partition("=")
            game_engine_kwargs[key.strip().lstrip("-")] = value.strip()
    if render is not None:
        game_engine_kwargs["render"] = render
    elif args is not None and hasattr(args, "render"):
        game_engine_kwargs["render"] = bool(args.render)
    if quiet:
        game_engine_kwargs["log_level"] = "ERROR"
    # Boot map = episode 0 of the role's seed stream.
    base_seed = getattr(args, "seed", None) if args is not None else None
    if role == "train" and base_seed is not None:
        game_engine_kwargs["seed"] = int(base_seed) + int(local_index)
    elif role == "eval":
        game_engine_kwargs["seed"] = -(1 + int(local_index))
    else:
        game_engine_kwargs.pop("seed", None)

    engine_log_dir = os.path.join(log_path, role, "headless_game_engine")
    os.makedirs(engine_log_dir, exist_ok=True)

    connector = GameEnvConnector(
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

    connector.connect()
    logger.info(f"Instance {instance_id} connected and ready")
    return connector


def main():
    args = RLArgsParser.parse_args()
    initializer = RLInitializer(args)

    try:
        connectors = initializer.start_instances()

        for i, connector in enumerate(connectors):
            logger = logging.getLogger(f"Instance-{i}")
            for step in range(10):
                logger.info(f"Instance {i} - Step {step}")
                obs, reward, done, info = connector.step([1, 1, 1])
                logger.info(
                    f"Instance {i} - Received observation: {obs}, reward: {reward}, done: {done}, info: {info}"
                )
            connector.reset()
            for step in range(10):
                logger.info(f"Instance {i} - Step {step}")
                obs, reward, done, info = connector.step([1, 1, 1])
                logger.info(
                    f"Instance {i} - Received observation: {obs}, reward: {reward}, done: {done}, info: {info}"
                )

    except KeyboardInterrupt:
        initializer.main_logger.info("Interrupted — shutting down...")
    except Exception as e:
        initializer.main_logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        initializer.stop_instances()


if __name__ == "__main__":
    main()
