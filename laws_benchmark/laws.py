from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Protocol

from .core import WorldState
from .utils import Vec, add, distance, norm, scale, sub, unit, zeros


class Law(Protocol):
    name: str

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        ...


def _empty_like(state: WorldState) -> List[Vec]:
    return [zeros(state.dim) for _ in state.bodies]


@dataclass
class NewtonianInversePowerLaw:
    strength: float = 1.0
    power: float = 2.0
    softening: float = 0.1
    name: str = "inverse_power"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        forces = []
        for i, body in enumerate(state.bodies):
            acc = zeros(state.dim)
            for j, other in enumerate(state.bodies):
                if i == j:
                    continue
                offset = sub(other.position, body.position)
                dist = norm(offset) + self.softening
                if dist == 0:
                    continue
                scale_factor = self.strength * other.mass / (dist ** (self.power + 1))
                acc = add(acc, scale(offset, scale_factor))
            forces.append(acc)
        return forces


@dataclass
class VelocityCouplingLaw:
    coupling: Iterable[float]
    cross: float = 0.0
    name: str = "velocity_coupling"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        coeffs = list(self.coupling)
        forces = []
        for body in state.bodies:
            base = zeros(state.dim)
            for d, v in enumerate(body.velocity):
                base[d] += coeffs[d % len(coeffs)] * v
            if self.cross and len(body.velocity) >= 2:
                rotated = [self.cross * body.velocity[-1]] + [self.cross * v for v in body.velocity[:-1]]
                base = add(base, rotated)
            forces.append(base)
        return forces


@dataclass
class TimeVaryingFieldLaw:
    direction: Vec
    amplitude: float
    omega: float
    phase: float = 0.0
    name: str = "time_field"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        mag = self.amplitude * math.sin(self.omega * t + self.phase)
        vec = scale(unit(self.direction), mag)
        return [vec for _ in state.bodies]


@dataclass
class PolynomialWellLaw:
    coeffs: Vec
    name: str = "polynomial_well"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        forces = []
        for body in state.bodies:
            force = []
            for coord in body.position:
                val = 0.0
                power = coord
                for c in self.coeffs:
                    val += c * power
                    power *= coord
                force.append(-val)
            forces.append(force)
        return forces


@dataclass
class HiddenDimensionPulseLaw:
    mixer: float
    omega: float
    offset: float = 0.0
    name: str = "hidden_pulse"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        phase = math.sin(self.omega * t + self.offset)
        forces = []
        for body in state.bodies:
            latent = (sum(body.position) + sum(body.velocity)) * self.mixer
            val = math.tanh(latent + phase)
            forces.append([val for _ in range(state.dim)])
        return forces


@dataclass
class RegionSwitchLaw:
    region: Callable[[Vec], bool]
    inner: Law
    outer: Law
    name: str = "region_switch"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        inner_acc = self.inner.acceleration(t, state)
        outer_acc = self.outer.acceleration(t, state)
        forces: List[Vec] = []
        for body, i_acc, o_acc in zip(state.bodies, inner_acc, outer_acc):
            if self.region(body.position):
                forces.append(i_acc)
            else:
                forces.append(o_acc)
        return forces


@dataclass
class CompositeLaw:
    components: List[Law]
    name: str = "composite"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        total = _empty_like(state)
        for law in self.components:
            acc = law.acceleration(t, state)
            total = [add(tot, a) for tot, a in zip(total, acc)]
        return total


def radial_trap(strength: float, power: float, boundary: float) -> RegionSwitchLaw:
    def inside(pos: Vec) -> bool:
        return distance(pos, [0.0 for _ in pos]) < boundary

    return RegionSwitchLaw(inside, PolynomialWellLaw([strength, power]), TimeVaryingFieldLaw([1.0] * 2, 0.0, 1.0))


@dataclass
class NonlinearSuperpositionLaw:
    components: List[Law]
    power: float = 1.5
    name: str = "nonlinear_superposition"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        total = _empty_like(state)
        for comp in self.components:
            acc = comp.acceleration(t, state)
            for i, vec in enumerate(acc):
                for d, val in enumerate(vec):
                    total[i][d] += math.copysign(abs(val) ** self.power, val)
        for vec in total:
            for d, val in enumerate(vec):
                vec[d] = math.copysign(abs(val) ** (1.0 / self.power), val)
        return total


@dataclass
class StochasticKickLaw:
    sigma: float = 0.05
    decay: float = 0.0
    seed: int | None = None
    name: str = "stochastic_kick"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        scale_t = math.exp(-self.decay * t) if self.decay else 1.0
        return [
            [self._rng.gauss(0.0, self.sigma) * scale_t for _ in range(state.dim)]
            for _ in state.bodies
        ]


@dataclass
class HiddenCoordinateLaw:
    latent: Vec
    amplitude: float = 1.0
    omega: float = 1.0
    bias: float = 0.0
    name: str = "hidden_coordinate"

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        direction = unit(self.latent)
        forces = []
        for body in state.bodies:
            projection = sum(p * w for p, w in zip(body.position, direction))
            val = math.tanh(self.amplitude * math.sin(self.omega * projection + t) + self.bias)
            forces.append([val * w for w in direction])
        return forces


@dataclass
class SymbolicLaw:
    expressions: List[object]
    dim: int
    name: str = "symbolic"
    _func: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import sympy as sp
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("sympy is required for SymbolicLaw") from exc
        t = sp.symbols("t")
        x_symbols = sp.symbols(" ".join(f"x{i}" for i in range(self.dim)))
        v_symbols = sp.symbols(" ".join(f"v{i}" for i in range(self.dim)))
        self._vars = [t, *x_symbols, *v_symbols]
        self._func = sp.lambdify(self._vars, self.expressions, "math")

    def acceleration(self, t: float, state: WorldState) -> List[Vec]:
        results: List[Vec] = []
        for body in state.bodies:
            args = [t, *body.position, *body.velocity]
            vals = self._func(*args)
            if isinstance(vals, (int, float)):
                vec = [float(vals) for _ in range(self.dim)]
            else:
                vec = [float(x) for x in vals]
            if len(vec) != self.dim:
                raise ValueError(f"SymbolicLaw expected dimension {self.dim}, got {len(vec)}")
            results.append(vec)
        return results


def sample_law(dim: int, rng: random.Random) -> Law:
    base = NewtonianInversePowerLaw(strength=rng.uniform(0.5, 3.0), power=rng.uniform(1.5, 3.5))
    vel = VelocityCouplingLaw([rng.uniform(-0.4, 0.4) for _ in range(dim)], cross=rng.uniform(-0.2, 0.2))
    time_field = TimeVaryingFieldLaw(
        [rng.uniform(-1, 1) for _ in range(dim)],
        amplitude=rng.uniform(0.2, 1.5),
        omega=rng.uniform(0.1, 3.0),
        phase=rng.uniform(0, math.pi),
    )
    hidden = HiddenDimensionPulseLaw(mixer=rng.uniform(0.2, 1.2), omega=rng.uniform(0.5, 2.5))
    well = PolynomialWellLaw([rng.uniform(-0.1, 0.3) for _ in range(3)])
    latent = HiddenCoordinateLaw(latent=[rng.uniform(-1, 1) for _ in range(dim)], amplitude=rng.uniform(0.5, 1.5), omega=rng.uniform(0.5, 2.0))
    return CompositeLaw([NonlinearSuperpositionLaw([base, vel], power=rng.uniform(1.2, 2.0)), time_field, hidden, well, latent])
