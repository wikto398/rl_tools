# RL Tools

Reusable tooling for training and evaluating reinforcement-learning agents against a game
environment. It handles communication between the game and the RL algorithm, the training
loop, metrics, evolutionary hyperparameter search and containers. All base classes follow
OOP principles so a game plugs in through small subclass/factory seams; the current
implementation targets Godot, but the architecture adapts to other engines.

Everything here is **game-agnostic**: `rl_tools` never imports the game.

## Layout

| Path | Role |
|------|------|
| `game_engine/` | `GameEnvConnector`, `HeadlessGameEngine` (+ `GodotHeadless`), observation/action interfaces, normalizers |
| `rl/Trainer/` | training / sweep orchestration |
| `rl/RLAgent/` | agents; `PolicyGradientAgent/PPOAgent` |
| `rl/Callback/` | callback ABC + list, eval, timing, checkpoints, stopping, metric **sinks** |
| `rl/Blackboard/` | in-process metrics bus (producers → sinks) |
| `rl/Evolution/` | evolutionary search: orchestrator engine, `strategy/` package, launchers |
| `rl/Factory/` | ABCs a game implements to plug in (network, callbacks, normalizers, expert) |
| `rl/WandbWrapper/` | W&B init/sweep helper |
| `rl/RLArgsParser/` | the shared CLI (`RLArgsParser.parse_args`) |
| `rl/RLInitializer/` | starts game instances and sets up logging |
| `utils/config.py` | `CONFIG` defaults (hosts, base ports, instances) |
| `main.py` | smoke connector — starts instances and steps them, no training |
| `docker/` | reusable `rl-base` / `rl-godot` images |

## Game Engine

### GameEnvConnector

Connects the game environment to the RL algorithm: it manages the communication, so the
agent receives observations and sends actions back, and it starts multiple game instances
for parallel training.

### HeadlessGameEngine

Interface for running the game engine headless (no GUI), which saves resources and speeds up
training. `GameEnvConnector` uses it to manage several instances. `GodotHeadless` is the
Godot implementation; other engines add a sibling subclass and a factory entry.

### Observation and Action Interfaces

Interfaces for exchanging data between the environment and the algorithm. Only UDP is
implemented today, but other transports can be added. Games with a non-messagepack protocol
provide a subclass (the game here wires a flat binary packet via
`StrategyUDPObservation`).

## RL

### RLAgent

Base class for RL agents — the structure and methods every agent implements (action
selection, learning). `PPOAgent` is the current implementation. Data is passed as
TensorDicts, so the agent only learns and the network handles data processing; this allows
richer architectures without touching the agent.

### Environment

Wrapper around `GameEnvConnector` used during training (seeding, episode bookkeeping).

### Trainer

`Trainer` owns the training loop and everything around it: builds envs/network/optimizer,
evaluates on a cadence, saves checkpoints, logs, and supports:

- **sweeps** (`--sweep_config`, W&B-driven) and **YAML configs** (`--config`),
- **early stopping** (`MetricStopCallback`) and **one-shot gates** (`GateStopCallback`),
- the **expert-in-the-loop coach** (a scripted policy overrides a decaying fraction of
  actions on-policy).

### Callbacks & Blackboard

Callbacks implement `Callback` (`on_train_start/end`, `on_rollout_*`, `on_update_*`,
`on_step`). Metrics flow through the shared **Blackboard**: producers (eval, timing, PPO
losses, game metrics) write records, and **sink callbacks** drain them —
`ConsoleCallback`, `TensorboardCallback`, `WandbCallback`, `CsvCallback` (all extend
`SinkCallback`). Add an output by subclassing `SinkCallback`.

### Factories

A game provides four factories (singletons with an `instance()` classmethod) implementing
the ABCs in `rl/Factory/`: `NetworkFactory`, `CallbacksFactory`, `NormalizersFactory`,
`ExpertFactory`. `Trainer` takes them as constructor arguments, so adding a game needs no
Trainer changes.

## Evolution

`rl_tools/rl/Evolution/` is a pluggable evolutionary hyperparameter search.

- **Engine vs strategy** — `EvolutionOrchestrator` is only the scheduler (worker slots,
  launcher, logging, snapshots, stop handling, W&B). The *algorithm* is an
  `EvolutionStrategy` chosen with `evolution.strategy` (`generational` or `pbt`) or a
  `module:Class` spec.
- **Shared types** (`strategy/`) — `EvolutionStrategy` (ABC + `Stats`), `Individual`,
  `Population` (ranking, elitism, tournament, eviction).
- **Strategies** — `GenerationalStrategy` (population per generation + barrier, elitism,
  crossover/mutation; the default) and `PBTStrategy` (steady-state PBT-lite: every free slot
  launches an `epoch_steps` child, warm-started from a tournament donor's checkpoint).
- **Launchers** — `ProcessLauncher` (local `multiprocessing`) or `DockerLauncher` (one
  `docker run` container per child). A child job is serialized to `job.json` and executed by
  `child_main`, so in-process and container children run identical code.
- **Fitness** — `Population.default_fitness_key` ranks lexicographically:
  `win_rate → -mean_win_turns → mean_end_total_resources → mean_end_population →
  mean_end_working_population → buildings_completed`. The tiebreakers are **reward-invariant
  game state** (children mutate their own reward coefficients, so raw return is not
  comparable across children).
- **Observability** — the engine logs `evolution/run/*` on every child completion (avg/best
  `win_rate`, the `mean_end_*` selection signal, `steps_per_second`, `crashed`, pool table).
- **Hard stops** — `evolution.max_steps` and `evolution.max_wall_hours` cap a run.

## Docker

`docker/` builds two reusable, game-agnostic images, in order (see `docker/README.md` and
`docker/build.sh`):

| Image | Dockerfile | Context | Contents |
|---|---|---|---|
| `rl-base:rocm7.2.4-py3.12` | `docker/Dockerfile` | `rl_tools/` | ROCm + PyTorch, `uv`, this project's deps in `/opt/venv` |
| `rl-godot:4.7.2` | `docker/godot/Dockerfile` | `docker/godot/` | Godot runtime + its runtime libs, `FROM` the base |

```bash
docker/build.sh
```

Version knobs are build args (`ROCM_VERSION`, `UBUNTU_VERSION`, `PYTHON_VERSION`,
`TORCH_VERSION`, `GODOT_VERSION`). The venv lives at `/opt/venv` (outside any `/app`
bind-mount a game might add), and `/app` is the default working dir. Other engines extend
`rl-base` from a sibling directory; games extend `rl-godot`.

## See also

- [`../AGENTS.md`](../AGENTS.md) — full operational detail and conventions
- [`../torch_files/README.md`](../torch_files/README.md) — the game's RL glue
- [`docker/README.md`](docker/README.md) — framework images
