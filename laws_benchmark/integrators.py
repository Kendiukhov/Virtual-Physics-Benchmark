from __future__ import annotations

from typing import Callable

from .core import ExternalForces, StateDerivative, WorldState
from .utils import add, scale

DerivativeFn = Callable[[float, WorldState, ExternalForces], StateDerivative]


def apply_derivative(state: WorldState, deriv: StateDerivative, factor: float) -> WorldState:
    new_bodies = []
    for body, dpos, dvel in zip(state.bodies, deriv.dpos, deriv.dvel):
        new_pos = add(body.position, scale(dpos, factor))
        new_vel = add(body.velocity, scale(dvel, factor))
        new_bodies.append(type(body)(new_pos, new_vel, body.mass))
    return WorldState(state.dim, new_bodies)


def euler_step(state: WorldState, t: float, dt: float, deriv_fn: DerivativeFn, external: ExternalForces) -> WorldState:
    k1 = deriv_fn(t, state, external)
    return apply_derivative(state, k1, dt)


def rk4_step(state: WorldState, t: float, dt: float, deriv_fn: DerivativeFn, external: ExternalForces) -> WorldState:
    k1 = deriv_fn(t, state, external)
    s2 = apply_derivative(state, k1, dt * 0.5)
    k2 = deriv_fn(t + dt * 0.5, s2, external)

    s3 = apply_derivative(state, k2, dt * 0.5)
    k3 = deriv_fn(t + dt * 0.5, s3, external)

    s4 = apply_derivative(state, k3, dt)
    k4 = deriv_fn(t + dt, s4, external)

    combo = k1.scaled(1 / 6) + k2.scaled(1 / 3) + k3.scaled(1 / 3) + k4.scaled(1 / 6)
    return apply_derivative(state, combo, dt)
