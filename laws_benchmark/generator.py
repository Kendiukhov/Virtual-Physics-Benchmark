from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .core import BodyState, WorldState
from .laws import (
    CompositeLaw,
    HiddenDimensionPulseLaw,
    HiddenCoordinateLaw,
    Law,
    NonlinearSuperpositionLaw,
    NewtonianInversePowerLaw,
    RegionSwitchLaw,
    StochasticKickLaw,
    TimeVaryingFieldLaw,
    sample_law,
)
from .simulation import Simulation, Trajectory
from .utils import Vec, copy_vec, seed_everything, zeros

Policy = Callable[[float, WorldState], List[Vec] | None]


@dataclass
class EnvironmentSpec:
    dim: int = 2
    num_bodies: int = 2
    dt: float = 0.05
    steps: int = 500
    complexity: str = "medium"
    seed: int | None = None


def _random_body(dim: int, rng: random.Random) -> BodyState:
    pos = [rng.uniform(-2.5, 2.5) for _ in range(dim)]
    vel = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    mass = rng.uniform(0.5, 3.0)
    return BodyState(pos, vel, mass)


def _choose_law(spec: EnvironmentSpec, rng: random.Random) -> Law:
    if spec.complexity == "easy":
        return NewtonianInversePowerLaw(strength=rng.uniform(0.5, 2.0), power=2.0)
    if spec.complexity == "hard":
        base = sample_law(spec.dim, rng)
        gate_radius = rng.uniform(1.5, 3.5)

        def region(pos: Vec) -> bool:
            return sum(x * x for x in pos) ** 0.5 < gate_radius

        switch = RegionSwitchLaw(
            region=region,
            inner=HiddenDimensionPulseLaw(mixer=rng.uniform(0.5, 1.5), omega=rng.uniform(0.5, 2.5)),
            outer=TimeVaryingFieldLaw(
                direction=[rng.uniform(-1, 1) for _ in range(spec.dim)],
                amplitude=rng.uniform(0.2, 1.5),
                omega=rng.uniform(0.1, 3.0),
                phase=rng.uniform(0, 3.14),
            ),
        )
        hidden_coord = HiddenCoordinateLaw(
            latent=[rng.uniform(-1, 1) for _ in range(spec.dim)],
            amplitude=rng.uniform(0.3, 1.2),
            omega=rng.uniform(0.3, 1.5),
            bias=rng.uniform(-0.2, 0.2),
        )
        noise = StochasticKickLaw(sigma=rng.uniform(0.01, 0.05), decay=rng.uniform(0.0, 0.01), seed=spec.seed)
        return CompositeLaw([NonlinearSuperpositionLaw([base, hidden_coord], power=rng.uniform(1.3, 2.0)), switch, noise])
    # medium by default
    return sample_law(spec.dim, rng)


def make_simulation(spec: EnvironmentSpec) -> tuple[Simulation, Law]:
    seed_everything(spec.seed)
    rng = random.Random(spec.seed)
    law = _choose_law(spec, rng)
    bodies = [_random_body(spec.dim, rng) for _ in range(spec.num_bodies)]
    state = WorldState(spec.dim, bodies)
    sim = Simulation(law, state, dt=spec.dt)
    return sim, law


def zero_policy(_: float, state: WorldState) -> List[Vec]:
    return [zeros(state.dim) for _ in state.bodies]


def random_push_policy(strength: float, rng: random.Random) -> Policy:
    def policy(_: float, state: WorldState) -> List[Vec]:
        pushes = []
        for _ in state.bodies:
            pushes.append([rng.uniform(-strength, strength) for _ in range(state.dim)])
        return pushes

    return policy


def axis_sweep_policy(strength: float) -> Policy:
    counter = {"step": 0}

    def policy(_: float, state: WorldState) -> List[Vec]:
        counter["step"] += 1
        dim = (counter["step"] - 1) % state.dim
        force = [0.0 for _ in range(state.dim)]
        force[dim] = strength if (counter["step"] // state.dim) % 2 == 0 else -strength
        return [force for _ in state.bodies]

    return policy


def latin_hypercube_policy(strength: float, samples: int, dim: int, rng: random.Random) -> Policy:
    directions = [
        [rng.uniform(-strength, strength) for _ in range(dim)]
        for _ in range(max(samples, 1))
    ]
    rng.shuffle(directions)
    counter = {"idx": 0}

    def policy(_: float, state: WorldState) -> List[Vec]:
        if not directions:
            return None
        force = directions[counter["idx"] % len(directions)]
        counter["idx"] += 1
        return [force for _ in state.bodies]

    return policy


def scheduled_impulse_policy(interval: int, impulse: Vec) -> Policy:
    impulse_copy = copy_vec(impulse)
    counter = {"step": 0}

    def policy(_: float, state: WorldState) -> List[Vec] | None:
        counter["step"] += 1
        if counter["step"] % interval == 0:
            return [impulse_copy for _ in state.bodies]
        return None

    return policy


def generate_trajectory(spec: EnvironmentSpec, policy: Policy | None = None) -> Tuple[Trajectory, Law]:
    sim, law = make_simulation(spec)
    rollout = sim.rollout(spec.steps, policy)
    return rollout, law
