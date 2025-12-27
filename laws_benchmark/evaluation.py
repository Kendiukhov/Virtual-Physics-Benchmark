from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .core import WorldState
from .laws import Law
from .simulation import Simulation, Trajectory
from .symbolic_utils import EquivalenceResult, vector_symbolic_equivalence
from .utils import distance


@dataclass
class MetricResult:
    mse: float
    max_error: float
    count: int


def acceleration_mse(traj: Trajectory, true_law: Law, candidate: Law) -> MetricResult:
    total = 0.0
    max_err = 0.0
    count = 0

    for t, state in zip(traj.times, traj.states):
        target = true_law.acceleration(t, state)
        predicted = candidate.acceleration(t, state)
        for t_vec, p_vec in zip(target, predicted):
            for tv, pv in zip(t_vec, p_vec):
                err = (tv - pv) ** 2
                total += err
                max_err = max(max_err, abs(tv - pv))
                count += 1
    mse = total / max(count, 1)
    return MetricResult(mse=mse, max_error=max_err, count=count)


def _replay(traj: Trajectory, law: Law) -> List[WorldState]:
    sim = Simulation(law, traj.initial, dt=traj.dt)
    states: List[WorldState] = []
    for force in traj.forces:
        states.append(sim.step(force))
    return states


def predictive_position_mse(traj: Trajectory, candidate: Law) -> MetricResult:
    predicted_states = _replay(traj, candidate)
    total = 0.0
    max_err = 0.0
    count = 0

    for ref_state, pred_state in zip(traj.states, predicted_states):
        for ref_body, pred_body in zip(ref_state.bodies, pred_state.bodies):
            err = distance(ref_body.position, pred_body.position) ** 2
            total += err
            max_err = max(max_err, err**0.5)
            count += 1
    mse = total / max(count, 1)
    return MetricResult(mse=mse, max_error=max_err, count=count)


def symbolic_expression_similarity(
    true_exprs: List[object],
    candidate_exprs: List[object],
    symbols: List[object],
    samples: int = 32,
    domain: tuple[float, float] = (-2.0, 2.0),
    tol: float = 1e-3,
) -> EquivalenceResult:
    return vector_symbolic_equivalence(true_exprs, candidate_exprs, symbols, samples=samples, domain=domain, tol=tol)
