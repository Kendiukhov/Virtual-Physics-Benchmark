from __future__ import annotations

import pathlib
import random
import sys

import sympy as sp

# Make package importable when running directly.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laws_benchmark import (
    BenchmarkEnv,
    EnvironmentSpec,
    random_push_policy,
    generate_trajectory,
)
from laws_benchmark.dataset import load_trajectory_json, save_trajectory_json
from laws_benchmark.evaluation import symbolic_expression_similarity


def run_symbolic_check() -> None:
    t = sp.symbols("t")
    x0, v0 = sp.symbols("x0 v0")
    true_exprs = [-0.5 * x0 + 0.1 * v0]
    candidate_exprs = ["-0.5*x0 + 0.05*v0"]
    result = symbolic_expression_similarity(true_exprs, candidate_exprs, [t, x0, v0])
    print("Symbolic equivalence -> equivalent:", result.equivalent, "max err:", round(result.max_error, 6))


def run_dataset_and_env_demo() -> None:
    spec = EnvironmentSpec(dim=2, num_bodies=2, steps=50, dt=0.05, complexity="medium", seed=3)
    policy = random_push_policy(0.2, random.Random(42))
    traj, law = generate_trajectory(spec, policy)
    out_path = ROOT / "examples" / "demo_trajectory.json"
    save_trajectory_json(out_path, traj, metadata={"law": law.name})
    restored = load_trajectory_json(out_path)
    print("Saved and reloaded trajectory frames:", len(restored.states), "->", out_path.name)

    env = BenchmarkEnv(spec)
    obs = env.reset(seed=7)
    for _ in range(3):
        obs, reward, done, info = env.step(None)
    print("Env sample time:", round(obs["time"], 3), "law:", info["law_name"], "remaining:", obs["step_remaining"])


def main() -> None:
    run_symbolic_check()
    run_dataset_and_env_demo()


if __name__ == "__main__":
    main()
