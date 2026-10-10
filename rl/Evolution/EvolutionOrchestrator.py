from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
import signal
import statistics
import time
from datetime import datetime
from typing import ClassVar

import wandb
from rl_tools.rl.Callback.Callback import Callback
from rl_tools.rl.Callback.CallbackList import CallbackList
from rl_tools.rl.Evolution.SaveEvolutionCallback import SaveEvolutionCallback
from rl_tools.rl.Evolution.StopEvolutionCallback import StopEvolutionCallback
from rl_tools.rl.Evolution.launchers import ChildLauncher, Job, build_launcher
from rl_tools.rl.Evolution.strategy.EvolutionStrategy import (
    EvolutionStrategy,
    Stats,
)
from rl_tools.rl.Evolution.strategy.Individual import Individual
from rl_tools.rl.Evolution.strategy.Population import (
    WORST_KEY,
    Population,
)

logger = logging.getLogger("Evolution")


class EvolutionOrchestrator:
    """Scheduler/engine for evolutionary hyperparameter search.

    The engine owns the mechanics: worker slots, the child launcher (local
    processes or one Docker container per child), logging, snapshots, stop
    handling and W&B. *What* to run next, when a phase is done and how to
    reproduce is delegated to an injected :class:`EvolutionStrategy`
    (``generational`` or ``pbt``).

    A child is one ``Trainer`` run (via a ``ChildRunner``) that reports its
    fitness by writing ``fitness.json``.
    """

    DEFAULT_EVO_CFG: ClassVar[dict] = {
        # --- shared ---
        "workers": 4,
        "instances": 2,
        "eval_instances": 1,
        "eval_episodes": 10,
        "eval_every": 10000,
        "stop_win_rate": 0.6,
        "win_rate_gate": 0.0,
        "build_gate_threshold": 0.5,
        "build_gate_patience": 3,
        "wandb_group": None,
        "port_stride": 4,
        #: log the pool table to W&B every N completed children (0 = never)
        "wandb_table_every": 20,
        #: log a trend/progress line + write progress.json every N children
        "progress_every": 10,
        #: trend window: last N children vs the N before them
        "progress_window": 10,
        # --- strategy selection ---
        "strategy": "generational",
        # --- hard stops (any strategy; useful for cloud spend) ---
        "max_steps": None,
        "max_wall_hours": None,
        "poll_interval": 2.0,
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
        # --- generational defaults (also read by GenerationalStrategy) ---
        "population": 8,
        "generations": 30,
        "steps_start": 50000,
        "steps_end": 750000,
        "steps_curve": "geometric",
        "elite_count": 1,
        "tournament_k": 3,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
    }

    def __init__(
        self,
        base_args: argparse.Namespace,
        strategy: EvolutionStrategy,
        child_runner,
        *,
        evo_cfg: dict | None = None,
        root_dir: str | None = None,
        generation_callbacks: list[Callback] | None = None,
        resume: bool = False,
        launcher: ChildLauncher | None = None,
    ) -> None:
        self.base_args = base_args
        self.base_args_dict = vars(base_args)
        self.strategy = strategy
        if isinstance(child_runner, str):
            self.child_runner_path = child_runner
        else:
            self.child_runner_path = (
                f"{child_runner.__module__}.{child_runner.__qualname__}"
            )
        cfg = dict(self.DEFAULT_EVO_CFG)
        cfg.update(evo_cfg or {})
        self.cfg = cfg
        if cfg["workers"] < 1:
            raise ValueError("workers must be >= 1")
        self.workers = max(1, min(int(cfg["workers"]), strategy.max_workers()))
        self.seed = int(getattr(base_args, "seed", None) or cfg.get("seed") or 42)
        # Absolute so child paths match what containers see through the
        # root_dir bind mount (the DockerLauncher mounts root_dir verbatim).
        self.root_dir = os.path.abspath(root_dir or self._default_root_dir())
        os.makedirs(os.path.join(self.root_dir, "state"), exist_ok=True)
        self.resume = bool(resume)
        if self.resume and not strategy.supports_resume:
            raise ValueError(
                f"strategy '{strategy.name}' does not support --resume_evolution"
            )
        self._stop_requested = False
        self._active_jobs: dict[Job, Individual] = {}
        self._phase_budget = 0
        self._wandb_run = None
        self._run_metrics_defined = False
        self._last_table_at = 0
        self._history: list[dict] = []
        self._best_individual: Individual | None = None
        self._last_progress_at = 0
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

    # --- main loop -----------------------------------------------------------

    def evolve(self) -> dict:
        if self.launcher.kind == "process":
            multiprocessing.set_start_method("spawn", force=True)
        self.strategy.bind(self)
        self._wandb_run = self._init_wandb_run()
        self._install_signal_handlers()
        self._write_pid()
        logger.info(
            f"strategy={self.strategy.name} executor={self.launcher.kind} "
            f"workers={self.workers}"
            + (
                f" image={self.cfg.get('image')}"
                if self.launcher.kind == "docker"
                else ""
            )
        )
        if (
            self.cfg.get("max_steps") is None
            and self.cfg.get("max_wall_hours") is None
            and type(self.strategy).should_stop is EvolutionStrategy.should_stop
        ):
            logger.warning(
                "No max_steps/max_wall_hours set — this run stops only on "
                "stop_win_rate, the stop file, or a signal."
            )
        stop_callbacks = [
            cb
            for cb in self.generation_callbacks.callbacks
            if isinstance(cb, StopEvolutionCallback)
        ]
        for cb in stop_callbacks:
            cb.start()

        rng = random.Random(self.seed)
        stats = Stats()
        if self.resume:
            individuals = self.strategy.resume(self._load_resume_state(), rng)
        else:
            individuals = self.strategy.initial_population(rng)
        population = Population(individuals, fitness_key=self.strategy.fitness_key)
        active: dict[Job, Individual] = {}
        self._active_jobs = active

        summary = {"best_genome": None, "best_fitness": None}
        try:
            while True:
                if self._stop_file_requested():
                    self._request_stop("stop file present")

                launched = 0
                while len(active) < self.workers:
                    individual = self.strategy.next_pending(
                        population, len(active), rng
                    )
                    if individual is None:
                        break
                    active[self._launch(individual)] = individual
                    launched += 1

                finished = [job for job in active if not job.poll()]
                for job in finished:
                    individual = active.pop(job)
                    exit_code = job.wait()
                    individual.fitness = self._read_fitness(individual)
                    individual.global_step = int(
                        (individual.fitness or {}).get("global_step", 0) or 0
                    )
                    individual.stopped_early = bool(
                        (individual.fitness or {}).get("stopped_early", False)
                    )
                    fitness = individual.fitness or {}
                    if fitness.get("errored") or individual.global_step == 0:
                        stats.crashed += 1
                    if individual.stopped_early:
                        stats.stopped_early += 1
                    self._history.append(
                        {
                            "resources": float(
                                fitness.get("mean_end_total_resources", 0.0) or 0.0
                            ),
                            "working": float(
                                fitness.get("mean_end_working_population", 0.0) or 0.0
                            ),
                            "population": float(
                                fitness.get("mean_end_population", 0.0) or 0.0
                            ),
                            "buildings": float(
                                fitness.get("buildings_completed", 0.0) or 0.0
                            ),
                            "win_rate": float(fitness.get("win_rate", 0.0) or 0.0),
                        }
                    )
                    if self._best_individual is None or self.strategy.fitness_key(
                        individual.fitness
                    ) > self.strategy.fitness_key(self._best_individual.fitness):
                        self._best_individual = individual
                    last = self._history[-1]
                    warm = individual.meta.get("warm_start_from")
                    logger.info(
                        f"  finished {individual.phase}/c{individual.slot} "
                        f"(exit={exit_code}) win_rate={last['win_rate']:.3f} "
                        f"res={last['resources']:.0f} "
                        f"working={last['working']:.1f} "
                        f"buildings={last['buildings']:.1f}"
                        + (f" warm=c{str(warm).rsplit('child_', 1)[1]}" if warm else "")
                    )
                    self.strategy.on_child_finished(
                        individual, individual.fitness, population, stats, rng
                    )
                if finished:
                    self._log_run_stats(population, stats)
                    self._maybe_report_progress(population, stats)

                done, reason = self._should_stop(population, stats)
                if done:
                    logger.info(f"Stopping: {reason}")
                    break
                if not launched and not finished:
                    time.sleep(float(self.cfg["poll_interval"]))

            self._maybe_report_progress(population, stats, force=True)
            summary = self._summary(population, stats)
            if self._wandb_run is not None:
                self._finish_wandb(self._wandb_run, summary)
        finally:
            for cb in stop_callbacks:
                cb.stop()
            self._remove_pid()
            for job in list(active):
                try:
                    job.terminate()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
            active.clear()
            self._active_jobs = {}
            self.launcher.close()
            if self._wandb_run is not None:
                self._wandb_run.finish()

        with open(os.path.join(self.root_dir, "best.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _should_stop(self, population: Population, stats: Stats) -> tuple[bool, str]:
        if self.cfg.get("max_wall_hours"):
            if stats.elapsed() / 3600.0 >= float(self.cfg["max_wall_hours"]):
                return True, f"max_wall_hours={self.cfg['max_wall_hours']}"
        if self.cfg.get("max_steps"):
            if stats.completed_steps >= int(self.cfg["max_steps"]):
                return True, f"max_steps={self.cfg['max_steps']}"
        return self.strategy.should_stop(population, stats)

    # --- progress / trend reporting -----------------------------------------

    _TRENDS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("resources", "{:.0f}"),
        ("working", "{:.2f}"),
        ("population", "{:.2f}"),
        ("buildings", "{:.2f}"),
        ("win_rate", "{:.3f}"),
    )

    def _trend(self, key: str, window: int) -> dict | None:
        """Mean of the last ``window`` children vs the ``window`` before them."""
        vals = [h[key] for h in self._history]
        if len(vals) < 2:
            return None
        k = max(1, min(int(window), len(vals) // 2))
        recent = vals[-k:]
        previous = vals[-2 * k : -k]
        recent_mean = sum(recent) / len(recent)
        if not previous:
            return {"recent": recent_mean, "previous": None, "delta": None}
        previous_mean = sum(previous) / len(previous)
        return {
            "recent": recent_mean,
            "previous": previous_mean,
            "delta": recent_mean - previous_mean,
        }

    @staticmethod
    def _format_trend(label: str, trend: dict | None, fmt: str) -> str:
        if trend is None:
            return f"{label}: n/a"
        text = f"{label} {fmt.format(trend['recent'])}"
        if trend["delta"] is None:
            return text
        sign = "+" if trend["delta"] >= 0 else ""
        return f"{text} ({sign}{fmt.format(trend['delta'])})"

    def _maybe_report_progress(
        self, population: Population, stats: Stats, *, force: bool = False
    ) -> None:
        every = int(self.cfg.get("progress_every", 10) or 0)
        if not every:
            return
        if not force and stats.completed - self._last_progress_at < every:
            return
        self._last_progress_at = stats.completed
        window = int(self.cfg.get("progress_window", 10) or 10)

        parts = [
            self._format_trend(key, self._trend(key, window), fmt)
            for key, fmt in self._TRENDS
        ]
        best = self._best_individual
        best_text = ""
        if best is not None and best.fitness:
            best_fitness = best.fitness or {}
            best_text = (
                f" | best c{best.slot} (res="
                f"{float(best_fitness.get('mean_end_total_resources', 0.0) or 0.0):.0f}, "
                f"working="
                f"{float(best_fitness.get('mean_end_working_population', 0.0) or 0.0):.1f})"
            )
        logger.info(
            f"progress: children={stats.completed} steps={stats.completed_steps} "
            f"elapsed={stats.elapsed() / 3600.0:.2f}h | "
            + " | ".join(parts)
            + best_text
        )
        self._write_progress_json(population, stats, window)

    def _write_progress_json(
        self, population: Population, stats: Stats, window: int
    ) -> None:
        evaluated = population.evaluated()

        def _mean(key: str) -> float | None:
            values = [float((m.fitness or {}).get(key, 0.0) or 0.0) for m in evaluated]
            return sum(values) / len(values) if values else None

        best = self._best_individual
        snapshot = {
            "children": stats.completed,
            "completed_steps": stats.completed_steps,
            "elapsed_hours": round(stats.elapsed() / 3600.0, 4),
            "crashed": stats.crashed,
            "stopped_early": stats.stopped_early,
            "trend_window": window,
            "trend": {key: self._trend(key, window) for key, _ in self._TRENDS},
            "pool": {
                "size": len(population.members),
                "mean_win_rate": _mean("win_rate"),
                "mean_end_total_resources": _mean("mean_end_total_resources"),
                "mean_end_working_population": _mean("mean_end_working_population"),
                "mean_buildings_completed": _mean("buildings_completed"),
            },
            "best": (
                {
                    "phase": best.phase,
                    "slot": best.slot,
                    "checkpoint": best.checkpoint,
                    "warm_start_from": best.meta.get("warm_start_from"),
                    "genome": self._compact(best.genome),
                    "fitness": best.fitness,
                }
                if best is not None
                else None
            ),
        }
        with open(os.path.join(self.root_dir, "progress.json"), "w") as f:
            json.dump(snapshot, f, indent=2)

    def _summary(self, population: Population, stats: Stats) -> dict:
        best = population.best
        evaluated = population.evaluated()
        win_rates = [
            float((m.fitness or {}).get("win_rate", 0.0) or 0.0) for m in evaluated
        ]
        summary = {
            "best_genome": best.genome if best else None,
            "best_fitness": best.fitness if best else None,
            "completed_children": stats.completed,
            "completed_steps": stats.completed_steps,
            "crashed_children": stats.crashed,
            "elapsed_hours": round(stats.elapsed() / 3600.0, 4),
        }
        if win_rates:
            summary["mean_win_rate"] = sum(win_rates) / len(win_rates)
            summary["median_win_rate"] = statistics.median(win_rates)
        return summary

    # --- strategy hooks (EvolutionHooks) -------------------------------------

    def phase_start(self, phase: int, budget: int, genomes: list[dict]) -> None:
        self._phase_budget = budget
        logger.info(f"=== phase {phase} budget={budget} ===")
        self.generation_callbacks.on_generation_start(phase, budget, genomes)

    def phase_end(self, phase: int, ranked: list[tuple[dict, dict]]) -> bool:
        best_fitness = ranked[0][1] if ranked else {}
        if ranked:
            logger.info(
                f"phase {phase}: best win_rate={best_fitness.get('win_rate')} "
                f"mean_win_turns={best_fitness.get('mean_win_turns')} "
                f"buildings_completed={best_fitness.get('buildings_completed')}"
            )
            logger.info(f"phase {phase}: best genome = {self._compact(ranked[0][0])}")
        if self._wandb_run is not None:
            self._log_generation_wandb(
                self._wandb_run, phase, ranked, self._phase_budget
            )
        stop_win_rate = self.cfg.get("stop_win_rate")
        if stop_win_rate is not None and ranked:
            if float(best_fitness.get("win_rate", 0.0) or 0.0) >= float(stop_win_rate):
                logger.info(
                    f"Stopping: best win_rate {best_fitness.get('win_rate')} "
                    f">= {stop_win_rate}"
                )
                return False
        return True

    def phase_snapshot(
        self,
        phase: int,
        ranked: list[tuple[dict, dict]],
        next_population: list[dict],
    ) -> None:
        self.generation_callbacks.on_generation_end(phase, ranked, next_population)

    def stop_requested(self) -> bool:
        return self._stop_requested

    # --- children ------------------------------------------------------------

    def _launch(self, individual: Individual) -> Job:
        individual.child_dir = self._child_dir(individual)
        child_config = self._child_config(individual)
        if child_config.get("checkpoint"):
            # remember the lineage for the W&B pool table
            individual.meta["warm_start_from"] = child_config["checkpoint"]
        job = self.launcher.launch(
            child_index=individual.slot,
            generation=individual.phase_index,
            runner_spec=self.child_runner_path,
            base_args=self.base_args_dict,
            genome=individual.genome,
            child_config=child_config,
        )
        individual.job_name = job.name
        # After launch, ``checkpoint`` points at what this run produces, so a
        # later child can warm-start from it.
        individual.checkpoint = os.path.join(
            individual.child_dir, "checkpoints", "final.pt"
        )
        logger.info(
            f"  started {individual.phase}/c{individual.slot} as {job.name!r} "
            f"dir={individual.child_dir}"
        )
        return job

    def _child_dir(self, individual: Individual) -> str:
        return os.path.join(
            self.root_dir, "children", individual.phase, f"child_{individual.slot}"
        )

    def _child_config(self, individual: Individual) -> dict:
        cfg = self.cfg
        return {
            "child_dir": individual.child_dir,
            "instances": cfg["instances"],
            "eval_instances": cfg["eval_instances"],
            "eval_episodes": cfg["eval_episodes"],
            "eval_every": cfg["eval_every"],
            "max_steps": individual.budget,
            "seed": self.seed + individual.seed_offset,
            "port_offset": self.launcher.port_offset(
                individual.slot, cfg["port_stride"]
            ),
            # Donor checkpoint for a warm start (PBT); None = train from scratch.
            "checkpoint": individual.checkpoint,
            "no_load_rng": True,
            "wandb_name": (
                f"evo-{self.exp_name}-g{individual.phase_index}-c{individual.slot}"
            ),
            "wandb_group": self.wandb_group,
            "wandb_extra_tags": (
                f"evolution,{individual.phase},child_{individual.slot}"
            ),
            "win_rate_gate": cfg["win_rate_gate"],
            "build_gate_threshold": cfg["build_gate_threshold"],
            "build_gate_patience": cfg["build_gate_patience"],
        }

    def _read_fitness(self, individual: Individual) -> dict:
        path = os.path.join(individual.child_dir, "fitness.json")
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # crashed / no fitness -> worst (win_rate -1 sorts below everything)
            return {"win_rate": -1.0}

    # --- W&B run stats (every strategy) --------------------------------------

    def _define_run_metrics(self) -> None:
        """Plot ``evolution/run/*`` against children completed.

        Keeps these series independent of the generational ``step=generation``
        logs, so both can coexist on one W&B run.
        """
        run = self._wandb_run
        if run is None or self._run_metrics_defined:
            return
        self._run_metrics_defined = True
        try:
            run.define_metric("evolution/run/children", hidden=True)
            run.define_metric("evolution/run/*", step_metric="evolution/run/children")
        except Exception as e:  # noqa: BLE001 - metrics are best-effort
            logger.warning(f"could not define wandb run metrics: {e}")

    def _log_run_stats(self, population: Population, stats: Stats) -> None:
        run = self._wandb_run
        if run is None:
            return
        self._define_run_metrics()

        members = population.evaluated()
        # Crashed children (win_rate -1) are counted separately below; folding
        # them into the averages would bury the fitness signal.
        successful = [
            m for m in members if self.strategy.fitness_key(m.fitness) != WORST_KEY
        ]
        fitnesses = [m.fitness or {} for m in successful]
        win_rates = [float(f.get("win_rate", 0.0) or 0.0) for f in fitnesses]
        buildings = [float(f.get("buildings_completed", 0.0) or 0.0) for f in fitnesses]
        steps = [int(f.get("global_step", 0) or 0) for f in fitnesses]
        resources = [
            float(f.get("mean_end_total_resources", 0.0) or 0.0) for f in fitnesses
        ]
        populations = [
            float(f.get("mean_end_population", 0.0) or 0.0) for f in fitnesses
        ]
        working = [
            float(f.get("mean_end_working_population", 0.0) or 0.0) for f in fitnesses
        ]
        returns = [
            float(f["mean_return"])
            for f in fitnesses
            if f.get("mean_return") is not None
        ]
        elapsed = max(stats.elapsed(), 1e-9)

        scalars = {
            "evolution/run/children": stats.completed,
            "evolution/run/children_running": len(self._active_jobs),
            "evolution/run/pool_size": len(population.members),
            "evolution/run/completed_steps": stats.completed_steps,
            "evolution/run/elapsed_hours": elapsed / 3600.0,
            "evolution/run/crashed": stats.crashed,
            "evolution/run/stopped_early": stats.stopped_early,
        }
        if fitnesses:
            scalars.update(
                {
                    "evolution/run/mean_win_rate": sum(win_rates) / len(win_rates),
                    "evolution/run/median_win_rate": statistics.median(win_rates),
                    "evolution/run/best_win_rate": max(win_rates),
                    "evolution/run/worst_win_rate": min(win_rates),
                    "evolution/run/mean_buildings_completed": sum(buildings)
                    / len(buildings),
                    "evolution/run/best_buildings_completed": max(buildings),
                    "evolution/run/mean_global_step": sum(steps) / len(steps),
                    # reward-invariant selection signal
                    "evolution/run/mean_end_resources": sum(resources) / len(resources),
                    "evolution/run/mean_end_population": sum(populations)
                    / len(populations),
                    "evolution/run/mean_end_working_population": sum(working)
                    / len(working),
                }
            )
        if returns:
            scalars["evolution/run/mean_return"] = sum(returns) / len(returns)
        if elapsed > 1.0:  # meaningless over a just-started run
            scalars["evolution/run/steps_per_second"] = stats.completed_steps / elapsed
        best = self._best_individual
        if best is not None and best.fitness:
            scalars["evolution/run/best_mean_end_resources"] = float(
                (best.fitness or {}).get("mean_end_total_resources", 0.0) or 0.0
            )
        resources_trend = self._trend(
            "resources", int(self.cfg.get("progress_window", 10) or 10)
        )
        if resources_trend and resources_trend["delta"] is not None:
            scalars["evolution/run/delta_mean_end_resources"] = resources_trend["delta"]
        try:
            run.log(scalars)
        except Exception as e:  # noqa: BLE001 - metrics are best-effort
            logger.warning(f"failed to log run stats to wandb: {e}")

        every = int(self.cfg.get("wandb_table_every", 20) or 0)
        if every and stats.completed - self._last_table_at >= every:
            self._last_table_at = stats.completed
            self._log_pool_table(run, population)

    def _log_pool_table(self, run, population: Population) -> None:
        ranked = population.ranked()

        def _cell(individual: Individual, key: str):
            value = (individual.fitness or {}).get(key)
            return value

        try:
            table = wandb.Table(
                columns=[
                    "phase",
                    "slot",
                    "win_rate",
                    "mean_win_turns",
                    "buildings_completed",
                    "mean_end_resources",
                    "mean_end_population",
                    "mean_end_working_population",
                    "global_step",
                    "stopped_early",
                    "warm_start_from",
                    "genome",
                ],
                data=[
                    [
                        m.phase,
                        m.slot,
                        _cell(m, "win_rate"),
                        _cell(m, "mean_win_turns"),
                        _cell(m, "buildings_completed"),
                        _cell(m, "mean_end_total_resources"),
                        _cell(m, "mean_end_population"),
                        _cell(m, "mean_end_working_population"),
                        m.global_step,
                        m.stopped_early,
                        os.path.basename(m.meta.get("warm_start_from") or "") or None,
                        json.dumps(self._compact(m.genome)),
                    ]
                    for m in ranked
                ],
            )
            run.log({"evolution/run/pool": table})
        except Exception as e:  # noqa: BLE001 - metrics are best-effort
            logger.warning(f"failed to log pool table to wandb: {e}")

    # --- W&B summary run -----------------------------------------------------

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

    # --- pause / resume ------------------------------------------------------

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

    def _load_resume_state(self) -> dict:
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
        logger.info(f"Resuming evolution from snapshot {latest}")
        return state

    @staticmethod
    def _compact(genome: dict) -> dict:
        return {
            k: (v if not isinstance(v, list) else v[:4] + ["..."])
            for k, v in genome.items()
        }
