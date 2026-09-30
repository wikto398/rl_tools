"""Container-side entrypoint for one evolutionary child.

The Docker launcher writes ``<child_dir>/job.json`` and runs:

    python -m rl_tools.rl.Evolution.child_main <child_dir>/job.json

which executes exactly the same code path as an in-process child.
"""

from __future__ import annotations

import sys

from rl_tools.rl.Evolution.jobs import read_job, run_child_job


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m rl_tools.rl.Evolution.child_main <job.json>",
            file=sys.stderr,
        )
        return 2
    run_child_job(read_job(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
