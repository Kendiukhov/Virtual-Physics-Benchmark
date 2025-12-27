from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from .core import ExternalForces, StateDerivative, WorldState, empty_forces
from .integrators import euler_step, rk4_step
from .laws import Law
from .utils import Vec, add, scale

IntegratorName = str
PolicyFn = Callable[[float, WorldState], ExternalForces]


@dataclass
class Trajectory:
    dt: float
    initial: WorldState
    times: List[float]
    states: List[WorldState]
    forces: List[ExternalForces]


class Simulation:
    def __init__(self, law: Law, initial: WorldState, dt: float = 0.05, integrator: IntegratorName = "rk4"):
        self.law = law
        self.state = initial.copy()
        self.dt = dt
        self.time = 0.0
        self.integrator = rk4_step if integrator == "rk4" else euler_step

    def _derivative(self, t: float, state: WorldState, external: ExternalForces) -> StateDerivative:
        base_acc = self.law.acceleration(t, state)
        if external is None:
            total_acc = base_acc
        else:
            total_acc = []
            for body, base, force in zip(state.bodies, base_acc, external):
                extra = scale(force, 1.0 / body.mass)
                total_acc.append(add(base, extra))
        return StateDerivative(
            [b.velocity[:] for b in state.bodies],
            total_acc,
        )

    def step(self, external: ExternalForces = None) -> WorldState:
        forces = external if external is not None else empty_forces(len(self.state.bodies), self.state.dim)
        self.state = self.integrator(self.state, self.time, self.dt, self._derivative, forces)
        self.time += self.dt
        return self.state.copy()

    def rollout(self, steps: int, policy: PolicyFn | None = None) -> Trajectory:
        start_state = self.state.copy()
        times: List[float] = []
        states: List[WorldState] = []
        forces: List[ExternalForces] = []

        for _ in range(steps):
            action = policy(self.time, self.state.copy()) if policy else None
            next_state = self.step(action)
            times.append(self.time)
            states.append(next_state.copy())
            forces.append(action)
        return Trajectory(self.dt, start_state, times, states, forces)
