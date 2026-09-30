# RL Tools
This directory contains tools for training and evaluating reinforcement learning agents. It handles communication between game environments and RL algorithms, allowing for seamless integration and efficient training. All base classes are designed with OOP principles in mind, making it easy to extend and customize for specific use cases. The current implementation is focused on the Godot game engine, but the architecture allows for easy adaptation to other engines as well.

## Game Engine 
### GameEnvConnector
The `GameEnvConnector` class is responsible for connecting the game environment to the RL algorithm. It manages the communication between the two, ensuring that the agent receives the necessary information from the environment and can send actions back to the environment effectively. It also starts multiple instances of the game environment for parallel training, which can significantly speed up the learning process.

### HeadlessGameEngine
Interface allowing to run the game engine in headless mode, which is essential for training RL agents without the need for a graphical interface. This can save computational resources and allow for faster training. GameEnvConnector uses the `HeadlessGameEngine` to manage multiple instances of the game environment for parallel training.

### Observation and Action Interfaces
Interfaces for communication between the game environment and the RL algorithm, that handle data exchange. For now only UDP communication is implemented, but other protocols can be added in the future. These interfaces ensure that the agent receives the necessary information from the environment and can send actions back to the environment effectively.

## RL
### RLAgent
Base class for reinforcement learning agents. It defines the structure and methods that all RL agents should implement, such as action selection and learning from experience. This class can be extended to create specific types of RL agents, such as DQN, PPO, etc. Currently working on finished PPO implementation, but other algorithms can be added in the future.
RLAgent was designed in a way to be really flexible with network architectures. Data is passed as TensorDicts, so that agent is responsible only for learning and network handles data processing. This allows to use more complicated networks, without need to modify the agent itself. 

### Environment
Wrapper for GameEnvConnector to be used during training.

## Docker
### Images
`docker/` builds two reusable, game-agnostic images, in order (see
`docker/README.md` and `docker/build.sh`):

| Image | Dockerfile | Context | Contents |
|---|---|---|---|
| `rl-base:rocm7.2.4-py3.12` | `docker/Dockerfile` | `rl_tools/` | ROCm + PyTorch, `uv`, this project's deps in `/opt/venv` |
| `rl-godot:4.7.2` | `docker/godot/Dockerfile` | `docker/godot/` | Godot runtime + its runtime libs, `FROM` the base |

```bash
docker/build.sh
```

Version knobs are build args (`ROCM_VERSION`, `UBUNTU_VERSION`, `PYTHON_VERSION`,
`TORCH_VERSION`, `GODOT_VERSION`). The venv lives at `/opt/venv` (outside any
`/app` bind-mount a game might add), and `/app` is the default working dir.
Other engines extend `rl-base` from a sibling directory; games extend
`rl-godot`.
