from __future__ import annotations

import pathlib
import random
import sys

# Make the package importable when running the script directly.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laws_benchmark.evaluation import acceleration_mse, predictive_position_mse
from laws_benchmark.generator import EnvironmentSpec, generate_trajectory, random_push_policy
from laws_benchmark.laws import CompositeLaw, NewtonianInversePowerLaw, VelocityCouplingLaw


def main() -> None:
    spec = EnvironmentSpec(dim=3, num_bodies=3, steps=200, dt=0.05, complexity="hard", seed=7)
    policy = random_push_policy(0.4, random.Random(13))
    traj, true_law = generate_trajectory(spec, policy)

    candidate = CompositeLaw(
        [
            NewtonianInversePowerLaw(strength=1.0, power=2.2),
            VelocityCouplingLaw([0.05, -0.02, 0.01], cross=0.02),
        ]
    )

    acc_metrics = acceleration_mse(traj, true_law, candidate)
    pred_metrics = predictive_position_mse(traj, candidate)

    print("Trajectory frames:", len(traj.states))
    print("Acceleration MSE:", round(acc_metrics.mse, 6), "max error:", round(acc_metrics.max_error, 4))
    print("Position MSE:", round(pred_metrics.mse, 6), "max position error:", round(pred_metrics.max_error, 4))


if __name__ == "__main__":
    main()
