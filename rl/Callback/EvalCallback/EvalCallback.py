from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback
from rl_tools.rl.Environment import Environment


class EvalCallback(Callback):
    """Run periodic full-episode evaluation on dedicated envs."""

    def __init__(
        self,
        *,
        envs: Sequence[Environment],
        every_timesteps: int,
        n_episodes: int,
        max_episode_steps: int = 10_000,
        log_prefix: str = "eval",
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        if not envs:
            raise ValueError("envs must be a non-empty sequence")
        if every_timesteps <= 0:
            raise ValueError(f"every_timesteps must be positive, got {every_timesteps}")
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")
        if max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be positive, got {max_episode_steps}"
            )
        self.envs = list(envs)
        self.every_timesteps = every_timesteps
        self.n_episodes = n_episodes
        self.max_episode_steps = max_episode_steps
        self.log_prefix = log_prefix
        self.deterministic = deterministic
        self._last_eval_step = 0
        self._eval_run_index = 0
        self._pool = ThreadPoolExecutor(max_workers=len(self.envs))

    def on_train_start(self) -> None:
        if self.agent is not None:
            self._last_eval_step = self.agent.global_step

    def on_train_end(self) -> None:
        self._pool.shutdown(wait=False)

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self, rollout: TensorDict) -> None:
        pass

    def on_step(
        self,
        *,
        actions: Any,
        rewards: Sequence[float],
        dones: Sequence[bool],
        infos: Sequence[dict],
    ) -> bool:
        return True

    def on_update_start(self, rollout: TensorDict) -> None:
        pass

    def on_update_end(self, update_info: dict) -> None:
        if self.agent is None:
            return
        if self.agent.global_step - self._last_eval_step < self.every_timesteps:
            return
        self._last_eval_step = self.agent.global_step
        self._run_eval()

    def _run_eval(self) -> None:
        assert self.agent is not None
        agent = self.agent
        agent.info(
            f"EvalCallback: evaluating for {self.n_episodes} episodes "
            f"on {len(self.envs)} env(s) at global_step={agent.global_step}"
        )
        agent.set_eval_mode(True, deterministic=self.deterministic)
        try:
            returns, lengths, wins, infos = self._collect_episodes()
        finally:
            agent.set_eval_mode(False)

        if not returns:
            agent.warning("EvalCallback: no episodes completed")
            return

        returns_arr = np.asarray(returns[: self.n_episodes], dtype=np.float64)
        lengths_arr = np.asarray(lengths[: self.n_episodes], dtype=np.float64)
        wins_arr = np.asarray(wins[: self.n_episodes], dtype=bool)
        infos = infos[: self.n_episodes]
        prefix = self.log_prefix
        blackboard = agent.blackboard
        step = agent.global_step

        def _log(key: str, value: float) -> None:
            blackboard.record(f"{prefix}/{key}", float(value), step)

        _log("mean_return", returns_arr.mean())
        if returns_arr.size > 1:
            _log("std_return", returns_arr.std())
        _log("mean_length", lengths_arr.mean())
        _log("n_episodes", returns_arr.size)
        n_wins = int(wins_arr.sum())
        _log("n_wins", n_wins)
        _log("win_rate", n_wins / wins_arr.size)

        # Summary-based stats only for episodes that ended via the game terminal
        # (truncated episodes carry no reward_breakdown).
        valid = [info for info in infos if info.get("reward_breakdown")]
        if not valid:
            return
        n_valid = len(valid)

        win_turns = [
            float(info["turns"])
            for info in valid
            if info.get("won") and info.get("turns") is not None
        ]
        loss_turns = [
            float(info["turns"])
            for info in valid
            if info.get("lost") and info.get("turns") is not None
        ]
        if win_turns:
            _log("mean_win_turns", np.mean(win_turns))
        if loss_turns:
            _log("mean_loss_turns", np.mean(loss_turns))

        breakdown_totals: dict[str, float] = {}
        for info in valid:
            for key, value in info.get("reward_breakdown", {}).items():
                breakdown_totals[key] = breakdown_totals.get(key, 0.0) + float(value)
        for key, value in sorted(breakdown_totals.items()):
            _log(f"breakdown/{key}", value / n_valid)

        for field in ("buildings_started", "buildings_completed"):
            totals: dict[str, float] = {}
            for info in valid:
                for key, value in info.get(field, {}).items():
                    totals[key] = totals.get(key, 0.0) + float(value)
            for key, value in sorted(totals.items()):
                _log(f"{field}/{key}", value / n_valid)

        population = [float(info.get("population", 0.0)) for info in valid]
        working_population = [
            float(info.get("working_population", 0.0)) for info in valid
        ]
        total_resources = [float(info.get("total_resources", 0.0)) for info in valid]
        production = [float(sum(info.get("production", []) or [])) for info in valid]
        _log("mean_end_population", np.mean(population))
        _log("mean_end_working_population", np.mean(working_population))
        _log("mean_end_total_resources", np.mean(total_resources))
        _log("mean_end_production", np.mean(production))

        blackboard.set("eval/latest_step", step)
        blackboard.set(
            "eval/latest",
            {
                "win_rate": n_wins / wins_arr.size,
                "n_wins": n_wins,
                "n_episodes": int(returns_arr.size),
                "mean_return": float(returns_arr.mean()),
                "mean_length": float(lengths_arr.mean()),
                "mean_end_population": float(np.mean(population)),
                "mean_end_working_population": float(np.mean(working_population)),
                "mean_end_total_resources": float(np.mean(total_resources)),
                "mean_end_production": float(np.mean(production)),
            },
        )

    def _collect_episodes(
        self,
    ) -> tuple[list[float], list[int], list[bool], list[dict]]:
        assert self.agent is not None
        agent = self.agent
        n_envs = len(self.envs)

        start_seeds: list[int | None] = []
        raw_obs = []
        for env in self.envs:
            env.episode_index = 0
            start_seeds.append(env.next_seed())
            raw_obs.append(env.reset(restart_sequence=True))
        agent.info(
            f"EvalCallback: eval start seeds={start_seeds} "
            f"(run_index={self._eval_run_index})"
        )
        self._eval_run_index += 1
        obs = [agent.split_observation(o) for o in raw_obs]
        ep_ret = [0.0] * n_envs
        ep_len = [0] * n_envs
        returns: list[float] = []
        lengths: list[int] = []
        wins: list[bool] = []
        infos: list[dict] = []

        while len(returns) < self.n_episodes:
            batch_obs = torch.stack(obs)
            forward = agent.get_action(batch_obs)
            actions = [
                forward["action"][i].detach().cpu().numpy() for i in range(n_envs)
            ]

            results = list(self._pool.map(self._step_env, self.envs, actions))

            next_obs: list[TensorDict] = []
            for i, (o, reward, done, info) in enumerate(results):
                ep_ret[i] += float(reward)
                ep_len[i] += 1
                finished = bool(done) or ep_len[i] >= self.max_episode_steps
                if finished:
                    returns.append(ep_ret[i])
                    lengths.append(ep_len[i])
                    wins.append(bool(info.get("won", False)))
                    infos.append(info if info is not None else {})
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                    if not done:
                        o = self.envs[i].reset()
                    if len(returns) >= self.n_episodes:
                        next_obs.append(agent.split_observation(o))
                        for j in range(i + 1, n_envs):
                            next_obs.append(obs[j])
                        break
                next_obs.append(agent.split_observation(o))
            else:
                obs = next_obs
                continue
            obs = next_obs
            break

        return returns, lengths, wins, infos

    @staticmethod
    def _step_env(
        env: Environment, action: np.ndarray
    ) -> tuple[dict, float, bool, dict]:
        o, r, d, info = env.step(action)
        if d:
            o = env.reset()
        return o, r, d, info if info is not None else {}
