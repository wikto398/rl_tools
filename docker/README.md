# Framework images

Two reusable images, built in order, neither containing game code:

| Image | Dockerfile | Context | Contents |
|---|---|---|---|
| `rl-base:rocm7.2.4-py3.12` | `Dockerfile` | `rl_tools/` | ROCm + PyTorch, `uv`, `rl_tools` deps in `/opt/venv`, build tooling |
| `rl-godot:4.7.2` | `godot/Dockerfile` | `godot/` | Godot runtime binary + its runtime libs, `FROM` the base |

```bash
rl_tools/docker/build.sh
```

The base needs the project metadata (`pyproject.toml`, `uv.lock`,
`.python-version`), hence its build context is the whole `rl_tools/` submodule
(`rl_tools/.dockerignore` keeps the ~13 GB `.venv` out of it). The venv lives at
`/opt/venv` — deliberately outside `/app` — so a game's code bind-mount does not
shadow it.

## Adding another engine

Copy `godot/` to a sibling directory (e.g. `unity/`), keep `FROM rl-base:...`,
and add the engine binary plus any runtime libraries it needs. Extra Python
dependencies can be added with
`uv pip install --python /opt/venv -r <requirements>`.

A game image extends the engine image directly, e.g.
`FROM rl-godot:4.7.2`; see the game repo's `docker/Dockerfile`.
