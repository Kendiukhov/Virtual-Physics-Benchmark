from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Tuple

from .core import WorldState
from .generator import EnvironmentSpec, make_simulation
from .laws import Law
from .simulation import Simulation

Observation = Dict[str, object]
RewardFn = Callable[[WorldState], float]


class BenchmarkEnv:
    def __init__(self, spec: EnvironmentSpec, reward_fn: RewardFn | None = None):
        self.spec = spec
        self.reward_fn = reward_fn
        self.sim: Simulation | None = None
        self.law: Law | None = None
        self.remaining_steps = 0

    def reset(self, seed: int | None = None) -> Observation:
        effective_spec = self.spec if seed is None else replace(self.spec, seed=seed)
        self.sim, self.law = make_simulation(effective_spec)
        self.remaining_steps = effective_spec.steps
        return self._obs(self.sim.state, self.sim.time)

    def step(self, action: List[List[float]] | None) -> Tuple[Observation, float, bool, Dict[str, object]]:
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")
        next_state = self.sim.step(action)
        self.remaining_steps -= 1
        reward = self.reward_fn(next_state) if self.reward_fn else 0.0
        done = self.remaining_steps <= 0
        info = {"law_name": getattr(self.law, "name", None), "t": self.sim.time}
        return self._obs(next_state, self.sim.time), reward, done, info

    def _obs(self, state: WorldState, time: float) -> Observation:
        return {
            "time": time,
            "dim": state.dim,
            "positions": [b.position for b in state.bodies],
            "velocities": [b.velocity for b in state.bodies],
            "masses": [b.mass for b in state.bodies],
            "step_remaining": self.remaining_steps,
        }
