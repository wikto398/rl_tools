from __future__ import annotations

import json
import os

from rl_tools.rl.Callback.Callback import NoOpCallback


class SaveEvolutionCallback(NoOpCallback):
    """Persist a resume snapshot at each generation boundary.

    On ``on_generation_end`` writes ``state/gen_<g>.json`` with the ranked
    population plus the next generation index and next population, so the
    evolution can be resumed with ``--resume_evolution <root_dir>``. This is
    the generation-level equivalent of the agent's run-end checkpoint save.
    """

    def __init__(self, state_dir: str) -> None:
        super().__init__()
        self.state_dir = state_dir

    def on_generation_end(
        self,
        generation: int,
        ranked: list[tuple[dict, dict]],
        next_population: list[dict],
    ) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        state = {
            "generation": generation,
            "best_genome": ranked[0][0],
            "best_fitness": ranked[0][1],
            "population": [
                {"genome": genome, "fitness": fitness} for genome, fitness in ranked
            ],
            "next_generation": generation + 1,
            "next_population": list(next_population),
        }
        path = os.path.join(self.state_dir, f"gen_{generation}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
