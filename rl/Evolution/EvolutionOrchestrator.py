from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
import signal
import time
from datetime import datetime
from typing import ClassVar

import wandb
from rl_tools.rl.Callback.Callback import Callback
from rl_tools.rl.Callback.CallbackList import CallbackList
from rl_tools.rl.Evolution.BudgetSchedule import BudgetSchedule
from rl_tools.rl.Evolution.GenomeSpace import GenomeSpace
from rl_tools.rl.Evolution.SaveEvolutionCallback import SaveEvolutionCallback
from rl_tools.rl.Evolution.StopEvolutionCallback import StopEvolutionCallback
from rl_tools.rl.Evolution.launchers import ChildLauncher, Job, build_launcher

logger = logging.getLogger("Evolution")


class EvolutionOrchestrator:
    """Generational GA over Trainers.

    Each generation samples/evolves ``population`` genomes; children run
    ``workers`` at a time (rolling batch). How a child is executed is pluggable
    via :class:`~rl_tools.rl.Evolution.launchers.ChildLauncher`: a local
    ``multiprocessing`` process (default) or its own Docker container
    (``executor: docker``). Each child is one ``Trainer`` run (via a
    ``ChildRunner`` subclass) and reports its fitness by writing
    ``fitness.json``. Selection is lexicographic over
    ``(win_rate, -mean_win_turns, buildings_completed)`` with elitism +
    tournament, followed by crossover/mutation.
    """

    DEFAULT_EVO_CFG: ClassVar[dict] = {
        "population": 8,
        "workers": 4,
        "instances": 2,
        "eval_instances": 1,
        "eval_episodes": 10,
        "eval_every": 10000,
        "generations": 30,
        "steps_start": 50000,
        "steps_end": 750000,
        "steps_curve": "geometric",
        "stop_win_rate": 0.6,
        "elite_count": 1,
        "tournament_k": 3,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "win_rate_gate": 0.0,
        "build_gate_threshold": 0.5,
        "build_gate_patience": 3,
        "wandb_group": None,
        "port_stride": 4,
        # --- child execution backend ---
        # "process": one multiprocessing child per genome (local dev).
        # "docker":  one `docker run` container per genome (cloud).
        "executor": "process",
        "image": None,
        "container_prefix": "evo",
        "gpu": True,
        "network": None,
        "cpus": None,
        "memory": None,
        "shm_size": None,
        "env": None,
        "extra_docker_args": None,
    }

    def __init__(
        self,
        base_args: argparse.Namespace,
        space: GenomeSpace,
        budget: BudgetSchedule,
        child_runner,
        evo_cfg: dict | None = None,
        root_dir: str | None = None,
        generation_callbacks: list[Callback] | None = None,
        resume: bool = False,
        launcher: ChildLauncher | None = None,
    ) -> None:
        self.base_args = base_args
        self.base_args_dict = vars(base_args)
        self.space = space
        self.budget = budget
        if isinstance(child_runner, str):
            self.child_runner_path = child_runner
        else:
            self.child_runner_path = (
                f"{child_runner.__module__}.{child_runner.__qualname__}"
            )
        cfg = dict(self.DEFAULT_EVO_CFG)
        cfg.update(evo_cfg or {})
        self.cfg = cfg
        if cfg["workers"] < 1 or cfg["population"] < 1:
            raise ValueError("population and workers must be >= 1")
        cfg["workers"] = min(cfg["workers"], cfg["population"])
        self.seed = int(getattr(base_args, "seed", None) or cfg.get("seed") or 42)
        # Absolute so child paths match what containers see through the
        # root_dir bind mount (the DockerLauncher mounts root_dir verbatim).
        self.root_dir = os.path.abspath(root_dir or self._default_root_dir())
        os.makedirs(os.path.join(self.root_dir, "state"), exist_ok=True)
        self.resume = bool(resume)
        self._stop_requested = False
        self._active_jobs: list[tuple[Job, int]] = []
        self.launcher = launcher or build_launcher(cfg, root_dir=self.root_dir)
        self._pid_path = os.path.join(self.root_dir, "pid")
        self._setup_logging()
        self.exp_name = (
            cfg["wandb_group"]
            or getattr(base_args, "wandb_name", None)
            or os.path.basename(self.root_dir.rstrip("/"))
        )
        self.wandb_group = cfg["wandb_group"] or f"evolution-{self.exp_name}"
        callbacks = list(generation_callbacks or [])
        if not any(isinstance(cb, StopEvolutionCallback) for cb in callbacks):
            callbacks.append(StopEvolutionCallback(key="p"))
        callbacks.append(SaveEvolutionCallback(os.path.join(self.root_dir, "state")))
        self.generation_callbacks = CallbackList(callbacks)
        for cb in self.generation_callbacks.callbacks:
            cb.orchestrator = self

    def _default_root_dir(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        return f"logs/evolution-{ts}"

    def _setup_logging(self) -> None:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            fh = logging.FileHandler(os.path.join(self.root_dir, "evolution.log"))
            fh.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
            )
            logger.addHandler(fh)
            ch = logging.StreamHandler()
            logger.addHandler(ch)

    # --- main loop ---

    def evolve(self) -> dict:
        if self.launcher.kind == "process":
            multiprocessing.set_start_method("spawn", force=True)
        wandb_run = self._init_wandb_run()
        self._install_signal_handlers()
        self._write_pid()
        logger.info(
            f"executor={self.launcher.kind}"
            + (
                f" image={self.cfg.get('image')}"
                if self.launcher.kind == "docker"
                else ""
            )
        )
        stop_callbacks = [
            cb
            for cb in self.generation_callbacks.callbacks
            if isinstance(cb, StopEvolutionCallback)
        ]
        for cb in stop_callbacks:
            cb.start()
        rng = random.Random(self.seed)
        if self.resume:
            start_generation, population = self._load_resume_state()
        else:
            start_generation = 0
            population = [self.space.sample(rng) for _ in range(self.cfg["population"])]
        best_fitness = None
        ranked = None
        try:
            for generation in range(start_generation, self.cfg["generations"]):
                if self._stop_file_requested():
                    self._request_stop("stop file present")
                budget = self.budget(generation, self.cfg["generations"])
                logger.info(
                    f"=== generation {generation}/{self.cfg['generations'] - 1} "
                    f"budget={budget} ==="
                )
                self.generation_callbacks.on_generation_start(
                    generation, budget, population
                )
                fitnesses = self._run_generation(population, generation)
                ranked = self._rank(population, fitnesses)
                best = ranked[0]
                best_fitness = best[1]
                logger.info(
                    f"gen {generation}: best win_rate={best_fitness.get('win_rate')} "
                    f"mean_win_turns={best_fitness.get('mean_win_turns')} "
                    f"buildings_completed={best_fitness.get('buildings_completed')}"
                )
                logger.info(f"gen {generation}: best genome = {self._compact(best[0])}")
                if wandb_run is not None:
                    self._log_generation_wandb(wandb_run, generation, ranked, budget)
                if best_fitness.get("win_rate", 0.0) >= self.cfg["stop_win_rate"]:
                    logger.info(
                        f"Stopping: best win_rate {best_fitness.get('win_rate')} "
                        f">= {self.cfg['stop_win_rate']}"
                    )
                    break
                next_population = self._evolve(ranked, rng)
                self.generation_callbacks.on_generation_end(
                    generation, ranked, next_population
                )
                if self._stop_requested:
                    logger.info(
                        f"Stop requested — stopping after generation {generation}"
                    )
                    break
                population = next_population
            if ranked is None:
                summary = {"best_genome": None, "best_fitness": None}
            else:
                summary = {"best_genome": ranked[0][0], "best_fitness": best_fitness}
            if wandb_run is not None:
                self._finish_wandb(wandb_run, summary)
        finally:
            for cb in stop_callbacks:
                cb.stop()
            self._remove_pid()
            for job, _ in list(self._active_jobs):
                try:
                    job.terminate()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
            self._active_jobs = []
            self.launcher.close()
            if wandb_run is not None:
                wandb_run.finish()
        with open(os.path.join(self.root_dir, "best.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # --- W&B summary run ---

    def _init_wandb_run(self):
        project = getattr(self.base_args, "wandb_project", None)
        if not project:
            return None
        args = self.base_args
        tags = [
            t.strip()
            for t in (getattr(args, "wandb_tags", "") or "").split(",")
            if t.strip()
        ]
        for t in ("evolution", "orchestrator", "single"):
            if t not in tags:
                tags.append(t)
        return wandb.init(
            project=project,
            entity=getattr(args, "wandb_entity", None),
            mode=getattr(args, "wandb_mode", "offline"),
            dir=self.root_dir,
            name=f"evolution-{self.exp_name}",
            group=self.wandb_group,
            tags=tags or None,
        )

    def _log_generation_wandb(
        self, wandb_run, generation: int, ranked, budget: int
    ) -> None:
        win_rates = [float(f.get("win_rate", 0.0) or 0.0) for _, f in ranked]
        buildings = [float(f.get("buildings_completed", 0.0) or 0.0) for _, f in ranked]
        best = ranked[0][1]
        scalars = {
            "evolution/best_win_rate": max(win_rates, default=0.0),
            "evolution/mean_win_rate": sum(win_rates) / len(win_rates)
            if win_rates
            else 0.0,
            "evolution/best_buildings_completed": best.get("buildings_completed", 0.0),
            "evolution/mean_buildings_completed": sum(buildings) / len(buildings)
            if buildings
            else 0.0,
            "evolution/best_mean_win_turns": best.get("mean_win_turns"),
            "evolution/children_crashed": sum(1 for wr in win_rates if wr < 0),
            "evolution/budget": budget,
        }
        wandb_run.log(scalars, step=generation)
        table = wandb.Table(
            columns=[
                "child",
                "win_rate",
                "mean_win_turns",
                "mean_return",
                "buildings_completed",
                "buildings_started",
                "global_step",
                "stopped_early",
                "genome",
            ],
            data=[
                [
                    i,
                    fitness.get("win_rate"),
                    fitness.get("mean_win_turns"),
                    fitness.get("mean_return"),
                    fitness.get("buildings_completed"),
                    fitness.get("buildings_started"),
                    fitness.get("global_step"),
                    fitness.get("stopped_early"),
                    json.dumps(self._compact(genome)),
                ]
                for i, (genome, fitness) in enumerate(ranked)
            ],
        )
        wandb_run.log({"evolution/population": table}, step=generation)

    def _finish_wandb(self, wandb_run, summary: dict) -> None:
        best_fitness = summary.get("best_fitness") or {}
        wandb_run.summary.update(
            {
                "best_win_rate": best_fitness.get("win_rate"),
                "best_mean_win_turns": best_fitness.get("mean_win_turns"),
                "best_buildings_completed": best_fitness.get("buildings_completed"),
                "best_genome": json.dumps(summary.get("best_genome", {})),
            }
        )

    # --- running children ---

    def _run_generation(self, genomes: list[dict], generation: int) -> list[dict]:
        budget = self.budget(generation, self.cfg["generations"])
        workers = self.cfg["workers"]
        pending = list(enumerate(genomes))
        active: list[tuple[Job, int]] = []
        results: dict[int, dict] = {}
        self._active_jobs = active

        try:
            while pending or active:
                while pending and len(active) < workers:
                    child_index, genome = pending.pop(0)
                    child_config = self._child_config(child_index, generation, budget)
                    job = self.launcher.launch(
                        child_index=child_index,
                        generation=generation,
                        runner_spec=self.child_runner_path,
                        base_args=self.base_args_dict,
                        genome=genome,
                        child_config=child_config,
                    )
                    active.append((job, child_index))
                    logger.info(
                        f"  started child {child_index} as {job.name!r} "
                        f"dir={child_config['child_dir']}"
                    )

                finished = [a for a in active if not a[0].poll()]
                for job, child_index in finished:
                    exit_code = job.wait()
                    active.remove((job, child_index))
                    fitness = self._read_fitness(child_index, generation)
                    results[child_index] = fitness
                    logger.info(
                        f"  finished child {child_index} (exit={exit_code}) "
                        f"win_rate={fitness.get('win_rate')}"
                    )
                if not finished:
                    time.sleep(2.0)
        finally:
            for job, _ in active:
                try:
                    job.terminate()
                except Exception as e:  # noqa: BLE001 - best-effort cleanup
                    logger.warning(f"  failed to terminate {job.name}: {e}")
            self._active_jobs = []

        return [results[i] for i in range(len(genomes))]

    def _child_config(self, child_index: int, generation: int, budget: int) -> dict:
        child_dir = os.path.join(
            self.root_dir, "children", f"gen_{generation}", f"child_{child_index}"
        )
        return {
            "child_dir": child_dir,
            "instances": self.cfg["instances"],
            "eval_instances": self.cfg["eval_instances"],
            "eval_episodes": self.cfg["eval_episodes"],
            "eval_every": self.cfg["eval_every"],
            "max_steps": budget,
            "seed": self.seed + (generation * self.cfg["population"] + child_index),
            "port_offset": self.launcher.port_offset(
                child_index, self.cfg["port_stride"]
            ),
            "wandb_name": f"evo-{self.exp_name}-g{generation}-c{child_index}",
            "wandb_group": self.wandb_group,
            "wandb_extra_tags": f"evolution,gen_{generation},child_{child_index}",
            "win_rate_gate": self.cfg["win_rate_gate"],
            "build_gate_threshold": self.cfg["build_gate_threshold"],
            "build_gate_patience": self.cfg["build_gate_patience"],
        }

    def _read_fitness(self, child_index: int, generation: int) -> dict:
        path = os.path.join(
            self.root_dir,
            "children",
            f"gen_{generation}",
            f"child_{child_index}",
            "fitness.json",
        )
        try:
            with open(path) as f:
                fitness = json.load(f)
            return fitness
        except (OSError, json.JSONDecodeError):
            # crashed / no fitness -> worst (win_rate -1 sorts below everything)
            return {"win_rate": -1.0}

    # --- selection / evolution ---

    def _fitness_key(self, fitness: dict) -> tuple:
        # A child that errored before its first rollout (no steps) ranks worst,
        # not as an ordinary 0-win child.
        if fitness.get("errored") or int(fitness.get("global_step", 1) or 0) == 0:
            return (-1.0, 0.0, 0.0)
        win_rate = float(fitness.get("win_rate", 0.0) or 0.0)
        mean_win_turns = fitness.get("mean_win_turns")
        f2 = -float(mean_win_turns) if mean_win_turns is not None else 0.0
        f3 = float(fitness.get("buildings_completed", 0.0) or 0.0)
        return (win_rate, f2, f3)

    def _rank(self, genomes, fitnesses) -> list[tuple[dict, dict]]:
        ranked = sorted(
            zip(genomes, fitnesses),
            key=lambda gf: self._fitness_key(gf[1]),
            reverse=True,
        )
        return ranked

    def _evolve(self, ranked, rng) -> list[dict]:
        population = self.cfg["population"]
        elite_count = min(self.cfg["elite_count"], population)
        next_gen = [genome for genome, _ in ranked[:elite_count]]
        while len(next_gen) < population:
            a = self._tournament(ranked, rng)
            b = self._tournament(ranked, rng)
            child = self.space.crossover(a, b, rng, rate=self.cfg["crossover_rate"])
            child = self.space.mutate(child, rng, rate=self.cfg["mutation_rate"])
            next_gen.append(child)
        return next_gen

    def _tournament(self, ranked, rng) -> dict:
        k = min(self.cfg["tournament_k"], len(ranked))
        contenders = rng.sample(ranked, k)
        best = max(contenders, key=lambda gf: self._fitness_key(gf[1]))
        return best[0]

    # --- state ---

    # --- pause / resume ---

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame) -> None:
            self._request_stop(f"signal {signum}")

        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
        except ValueError:
            pass  # not in the main thread

    def _request_stop(self, reason: str | None = None) -> None:
        for cb in self.generation_callbacks.callbacks:
            if isinstance(cb, StopEvolutionCallback):
                cb.request_stop(reason)
                return
        self._stop_requested = True

    def _stop_file_requested(self) -> bool:
        return os.path.exists(os.path.join(self.root_dir, "stop"))

    def _write_pid(self) -> None:
        with open(self._pid_path, "w") as f:
            f.write(str(os.getpid()))

    def _remove_pid(self) -> None:
        try:
            os.remove(self._pid_path)
        except OSError:
            pass

    def _load_resume_state(self) -> tuple[int, list[dict]]:
        state_dir = os.path.join(self.root_dir, "state")
        snapshots = [
            name
            for name in os.listdir(state_dir)
            if name.startswith("gen_") and name.endswith(".json")
        ]
        snapshots.sort(key=lambda name: int(name[4:-5]))
        if not snapshots:
            raise RuntimeError(
                f"No evolution snapshots in {state_dir} — nothing to resume"
            )
        latest = snapshots[-1]
        with open(os.path.join(state_dir, latest)) as f:
            state = json.load(f)
        start_generation = int(state["next_generation"])
        next_population = list(state["next_population"])
        logger.info(
            f"Resuming evolution from generation {start_generation} (snapshot {latest})"
        )
        return start_generation, next_population

    @staticmethod
    def _compact(genome: dict) -> dict:
        return {
            k: (v if not isinstance(v, list) else v[:4] + ["..."])
            for k, v in genome.items()
        }
