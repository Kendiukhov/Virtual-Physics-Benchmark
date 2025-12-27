Virtual Physics Benchmark
=========================

This repo sketches a benchmark harness where AI agents must rediscover hidden physical laws by running experiments inside virtual worlds. Each world is governed by a fully known but potentially "weird" law; we can score agents on how well their discovered formulas predict motion or match the ground truth.

What is implemented
-------------------
- A lightweight simulation core (`laws_benchmark/`) that supports arbitrary force laws, multiple bodies, and RK4/Euler integrators.
- A library of composable law primitives (`laws.py`) including inverse-power gravity, velocity coupling, time-varying fields, polynomial wells, hidden latent pulses, region-gated switches, nonlinear superposition, stochastic kicks, hidden-coordinate forces, and symbolic laws.
- A law generator (`generator.py`) that assembles easy/medium/hard universes and random initial conditions in any dimension.
- Experiment/rollout support (`simulation.py`) that lets a policy inject forces each step, producing trajectories with actions attached.
- Evaluation helpers (`evaluation.py`) that score a candidate law by acceleration fit, trajectory replay error, and symbolic equivalence checks (`symbolic_expression_similarity`).
- Dataset export/import (`dataset.py`) and a minimal gym-style interface (`envs.py`) so agents can interact step-by-step or store rollouts.
- Example scripts: `examples/basic_run.py` (hard scenario with naive law) and `examples/symbolic_and_export.py` (symbolic matching, dataset export, env demo).

Quick start
-----------
- Run a hard scenario with numeric laws:
  ```
  python examples/basic_run.py
  ```
  This builds a 3-body, 3D world with a composite law (gravity + velocity coupling + time field + hidden pulses + region switch + nonlinear mixing), rolls out 200 steps with random pushes, and prints acceleration/position errors for a naive candidate law.

- See symbolic checks, dataset export, and env usage:
  ```
  python examples/symbolic_and_export.py
  ```

- Use active-exploration policies and curriculum (snippet):
  ```python
  from laws_benchmark import EnvironmentSpec, generate_trajectory, axis_sweep_policy, latin_hypercube_policy, linear_curriculum
  import random

  spec = EnvironmentSpec(dim=3, num_bodies=3, steps=150, complexity="medium", seed=0)
  policy = axis_sweep_policy(strength=0.3)
  traj, law = generate_trajectory(spec, policy)

  # Build a staged curriculum of increasingly hard worlds
  steps = linear_curriculum(levels=4, start_dim=2, max_dim=4, start_bodies=2, max_bodies=4)
  for step in steps:
      print(step.description)
  ```

Core concepts
-------------
- **Law primitives**: `NewtonianInversePowerLaw`, `VelocityCouplingLaw`, `TimeVaryingFieldLaw`, `PolynomialWellLaw`, `HiddenDimensionPulseLaw`, `RegionSwitchLaw`, `NonlinearSuperpositionLaw`, `StochasticKickLaw`, `HiddenCoordinateLaw`, and `SymbolicLaw`. They can be summed via `CompositeLaw`, letting you build exotic universes (velocity-dependent forces, time oscillations, piecewise rules, latent modulation, stochasticity, and symbolic definitions).
- **Random law generator**: `EnvironmentSpec` defines dimension, body count, timestep, and complexity. `generate_trajectory` produces a rollout plus the ground-truth law. Complexity levels adjust which primitives are combined.
- **Experiment policies**: Provide a function `(time, state) -> forces` to inject probes. Included options: `random_push_policy`, `axis_sweep_policy`, `latin_hypercube_policy`, `scheduled_impulse_policy`, and `zero_policy`. Policies are passed to `Simulation.rollout`.
- **Evaluation**: `acceleration_mse` compares predicted accelerations against the true law on recorded states; `predictive_position_mse` replays the trajectory with a candidate law under the same actions and measures positional drift; `symbolic_expression_similarity` uses SymPy to test whether a discovered formula matches the ground truth up to numerical tolerance.
- **Datasets**: `save_trajectory_json`/`load_trajectory_json` serialize trajectories (including actions) for offline analysis.
- **Env interface**: `BenchmarkEnv` provides `reset`/`step` methods with observations and optional rewards, mirroring gym-style loops without adding dependencies.
- **Curriculum**: `linear_curriculum` builds a progression of `EnvironmentSpec` objects that increase dimension, body count, steps, and complexity, useful for staged training/evaluation.

Design notes and extension ideas
--------------------------------
- Expand experiment policies further (gradient probes, adaptive excitation tuned to local curvature).
- Add richer law families (higher-order derivatives, discontinuities, explicit control-affine terms, hidden coordinates that only appear in some regions).
- Add more export formats (Arrow, Parquet) and automated dataset sampling utilities.
- Introduce more curriculum shapes (randomized progressions, competence-based advancement).

Repository layout
-----------------
- `laws_benchmark/core.py`: dataclasses for bodies, world state, and derivatives.
- `laws_benchmark/integrators.py`: Euler and RK4 integrators.
- `laws_benchmark/laws.py`: law primitives and samplers.
- `laws_benchmark/simulation.py`: simulator and trajectory recording with action hooks.
- `laws_benchmark/generator.py`: environment spec, random initial states, and policies.
- `laws_benchmark/evaluation.py`: scoring utilities.
- `examples/basic_run.py`: runnable demo.
