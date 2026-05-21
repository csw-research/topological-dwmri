#!/usr/bin/env python3
"""Run all simulation experiments for the DW-MRI paper.

Outputs JSON-serialised results to ``results/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    save_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", default=str(ROOT / "results"),
        help="Output directory for JSON results.",
    )
    parser.add_argument(
        "--experiments", default="1,2,3,4",
        help="Comma-separated list of experiments to run.",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Run a reduced version suitable for laptop validation.",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = {int(x) for x in args.experiments.split(",") if x.strip()}

    if 1 in todo:
        t0 = time.perf_counter()
        kwargs = dict(n_replications=20, n_steps=1000) if args.small else {}
        res = run_experiment_1(**kwargs)
        p = save_experiment(res, out)
        print(f"[exp1] saved -> {p}  ({time.perf_counter()-t0:.1f}s)")

    if 2 in todo:
        t0 = time.perf_counter()
        kwargs = dict(n_trials=50) if args.small else {}
        res = run_experiment_2(**kwargs)
        p = save_experiment(res, out)
        print(f"[exp2] saved -> {p}  ({time.perf_counter()-t0:.1f}s)")

    if 3 in todo:
        t0 = time.perf_counter()
        kwargs = dict(n_voxels=20, n_dirs=60) if args.small else {}
        res = run_experiment_3(**kwargs)
        p = save_experiment(res, out)
        print(f"[exp3] saved -> {p}  ({time.perf_counter()-t0:.1f}s)")

    if 4 in todo:
        t0 = time.perf_counter()
        kwargs = dict(n_voxels=20, n_dirs=60) if args.small else {}
        res = run_experiment_4(**kwargs)
        p = save_experiment(res, out)
        print(f"[exp4] saved -> {p}  ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
