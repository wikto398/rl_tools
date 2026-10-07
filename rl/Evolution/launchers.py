"""Launching one evolutionary child: in-process or in a Docker container.

``EvolutionOrchestrator`` only talks to :class:`ChildLauncher` / :class:`Job`,
so the same generation logic drives:

* ``ProcessLauncher`` — a ``multiprocessing`` child (local dev, the default), or
* ``DockerLauncher`` — a ``docker run`` container (cloud), one per child.

The child job itself is serialized with :mod:`rl_tools.rl.Evolution.jobs`, so a
container runs the exact same code path as an in-process child.
"""

from __future__ import annotations

import abc
import grp
import multiprocessing
import os
import subprocess

from rl_tools.rl.Evolution.jobs import encode_job, spawn_worker, write_job

#: Environment variables forwarded into child containers when present.
#: (PYTHONPATH is intentionally not forwarded: the image already sets it to the
#: app root and child containers run with that workdir.)
DEFAULT_ENV_PASSTHROUGH = (
    "WANDB_API_KEY",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "HSA_OVERRIDE_GFX_VERSION",
)


def resolve_group_ids(groups) -> list[str]:
    """Map group names to numeric GIDs (``--group-add`` accepts either).

    Names resolve against the *orchestrator host*, not the image, and unknown
    groups are dropped: containers run as root by default, which already has
    device access. This avoids ``unable to find group render`` failures on
    images whose ``/etc/group`` differs from the host.
    """
    ids: list[str] = []
    for group in groups:
        name = str(group)
        if name.isdigit():
            ids.append(name)
            continue
        try:
            ids.append(str(grp.getgrnam(name).gr_gid))
        except KeyError:
            continue
    return ids


class Job(abc.ABC):
    """A running child; the orchestrator polls/waits/terminates it."""

    name: str

    @abc.abstractmethod
    def poll(self) -> bool:
        """True while the child is still running."""

    @abc.abstractmethod
    def wait(self) -> int:
        """Block until the child exits and return its exit code."""

    @property
    @abc.abstractmethod
    def exitcode(self) -> int | None: ...

    @abc.abstractmethod
    def terminate(self) -> None:
        """Best-effort stop (used when the orchestrator is interrupted)."""


class ProcessJob(Job):
    def __init__(self, name: str, process: multiprocessing.Process):
        self.name = name
        self._process = process

    def poll(self) -> bool:
        return self._process.is_alive()

    def wait(self) -> int:
        self._process.join()
        return int(self._process.exitcode if self._process.exitcode is not None else 0)

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()


class DockerJob(Job):
    def __init__(
        self, name: str, popen: subprocess.Popen, log_file, docker_bin: str = "docker"
    ):
        self.name = name
        self._popen = popen
        self._log_file = log_file
        self._docker_bin = docker_bin

    def poll(self) -> bool:
        return self._popen.poll() is None

    def wait(self) -> int:
        rc = self._popen.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        return int(rc)

    @property
    def exitcode(self) -> int | None:
        return self._popen.returncode

    def terminate(self) -> None:
        if self._popen.poll() is None:
            self._popen.terminate()
            try:
                self._popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._popen.kill()
        # Belt and braces: a container whose PID 1 is the entrypoint ignores
        # SIGTERM, so force-remove it (also cleans up after a killed client).
        subprocess.run(
            [self._docker_bin, "rm", "-f", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


class ChildLauncher(abc.ABC):
    #: Human-readable executor name (log/config reporting).
    kind = "abstract"

    @abc.abstractmethod
    def launch(
        self,
        *,
        child_index: int,
        generation: int,
        runner_spec: str,
        base_args: dict,
        genome: dict,
        child_config: dict,
    ) -> Job: ...

    def port_offset(self, child_index: int, stride: int) -> int:
        """UDP port offset for this child.

        Containers with their own network namespace are already isolated, so the
        default is 0; the process launcher (and host networking) needs a real
        offset to avoid collisions on the shared host network stack.
        """
        return child_index * stride

    def close(self) -> None:
        """Release any launcher-level resources (no-op by default)."""


class ProcessLauncher(ChildLauncher):
    """Run each child as a ``multiprocessing`` process (spawn)."""

    kind = "process"

    def launch(
        self, *, child_index, generation, runner_spec, base_args, genome, child_config
    ):
        job = encode_job(runner_spec, base_args, genome, child_config)
        name = f"evo-g{generation}-c{child_index}"
        process = multiprocessing.Process(target=spawn_worker, args=(job,), name=name)
        process.start()
        return ProcessJob(name, process)


class DockerLauncher(ChildLauncher):
    """Run each child as its own ``docker run`` container."""

    kind = "docker"

    def __init__(
        self,
        image: str,
        *,
        root_dir: str,
        gpu: bool = True,
        network: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        shm_size: str | None = None,
        env: dict | None = None,
        extra_args: list[str] | None = None,
        devices: list[str] | None = None,
        group_add: list[str] | None = None,
        security_opt: list[str] | None = None,
        ipc: str | None = "host",
        workdir: str = "/app",
        python: str = "python",
        child_module: str = "rl_tools.rl.Evolution.child_main",
        docker_bin: str = "docker",
        name_prefix: str = "evo",
    ):
        self.image = image
        self.root_dir = os.path.abspath(root_dir)
        self.gpu = gpu
        self.network = network
        self.cpus = cpus
        self.memory = memory
        self.shm_size = shm_size
        self.env = dict(env or {})
        self.extra_args = list(extra_args or [])
        self.devices = list(devices or (["/dev/kfd", "/dev/dri"] if gpu else []))
        self.group_add = (
            resolve_group_ids(group_add)
            if group_add is not None
            else resolve_group_ids(["video", "render"] if gpu else [])
        )
        self.security_opt = list(
            security_opt or (["seccomp=unconfined"] if gpu else [])
        )
        self.ipc = ipc
        self.workdir = workdir
        self.python = python
        self.child_module = child_module
        self.docker_bin = docker_bin
        self.name_prefix = name_prefix
        # Host networking shares the host UDP stack -> children need real offsets.
        self.host_network = network == "host"

    def port_offset(self, child_index: int, stride: int) -> int:
        return child_index * stride if self.host_network else 0

    def _container_name(self, generation: int, child_index: int) -> str:
        suffix = os.path.basename(self.root_dir.rstrip("/")) or "run"
        # Docker names allow [a-zA-Z0-9][a-zA-Z0-9_.-]; replace the rest.
        safe = "".join(c if c.isalnum() or c in "_.-" else "-" for c in suffix)
        return f"{self.name_prefix}-{safe}-g{generation}-c{child_index}"

    def _build_command(self, name: str, job_path: str) -> list[str]:
        cmd: list[str] = [
            self.docker_bin,
            "run",
            "--rm",
            # PID 1 (tini) forwards signals, so `docker stop`/Ctrl-C reach Python.
            "--init",
            "--name",
            name,
            "--workdir",
            self.workdir,
        ]
        # Mount the orchestrator's root_dir at the same absolute path so every
        # child_dir/log/checkpoint path resolves identically inside the container.
        cmd += ["-v", f"{self.root_dir}:{self.root_dir}"]
        if self.network:
            cmd += ["--network", self.network]
        for device in self.devices:
            cmd += ["--device", device]
        for group in self.group_add:
            cmd += ["--group-add", group]
        for opt in self.security_opt:
            cmd += ["--security-opt", opt]
        if self.ipc:
            cmd += ["--ipc", self.ipc]
        if self.cpus is not None:
            cmd += ["--cpus", str(self.cpus)]
        if self.memory:
            cmd += ["--memory", self.memory]
        if self.shm_size:
            cmd += ["--shm-size", self.shm_size]
        passthrough = {
            **{k: os.environ[k] for k in DEFAULT_ENV_PASSTHROUGH if k in os.environ},
            **self.env,
        }
        for key, value in passthrough.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += list(self.extra_args)
        cmd += [self.image, self.python, "-m", self.child_module, job_path]
        return cmd

    def launch(
        self, *, child_index, generation, runner_spec, base_args, genome, child_config
    ):
        child_dir = os.path.abspath(child_config["child_dir"])
        os.makedirs(child_dir, exist_ok=True)
        job_path = os.path.join(child_dir, "job.json")
        write_job(job_path, encode_job(runner_spec, base_args, genome, child_config))

        name = self._container_name(generation, child_index)
        command = self._build_command(name, job_path)
        log_path = os.path.join(child_dir, "container.log")
        log_file = open(log_path, "w")
        try:
            popen = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        except OSError:
            log_file.close()
            raise
        return DockerJob(name, popen, log_file, docker_bin=self.docker_bin)


def build_launcher(cfg: dict, *, root_dir: str) -> ChildLauncher:
    """Build the launcher selected by ``cfg['executor']`` (``process``/``docker``)."""
    executor = str(cfg.get("executor", "process") or "process").lower()
    if executor == "process":
        return ProcessLauncher()
    if executor == "docker":
        image = cfg.get("image")
        if not image:
            raise ValueError(
                "evolution.executor='docker' requires evolution.image "
                "(or --image) to be set"
            )
        return DockerLauncher(
            image=image,
            root_dir=root_dir,
            gpu=bool(cfg.get("gpu", True)),
            network=cfg.get("network") or None,
            cpus=cfg.get("cpus"),
            memory=cfg.get("memory"),
            shm_size=cfg.get("shm_size"),
            env=cfg.get("env") or {},
            extra_args=cfg.get("extra_docker_args") or [],
            name_prefix=str(cfg.get("container_prefix", "evo")),
        )
    raise ValueError(f"unknown executor {executor!r} (expected 'process' or 'docker')")
