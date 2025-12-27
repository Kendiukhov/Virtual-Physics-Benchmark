from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .utils import Vec, copy_vec, zeros


@dataclass
class BodyState:
    position: Vec
    velocity: Vec
    mass: float = 1.0

    def copy(self) -> "BodyState":
        return BodyState(copy_vec(self.position), copy_vec(self.velocity), self.mass)


@dataclass
class WorldState:
    dim: int
    bodies: List[BodyState]

    def copy(self) -> "WorldState":
        return WorldState(self.dim, [b.copy() for b in self.bodies])


@dataclass
class StateDerivative:
    dpos: List[Vec]
    dvel: List[Vec]

    def scaled(self, factor: float) -> "StateDerivative":
        return StateDerivative(
            [[x * factor for x in v] for v in self.dpos],
            [[x * factor for x in v] for v in self.dvel],
        )

    def __add__(self, other: "StateDerivative") -> "StateDerivative":
        return StateDerivative(
            [[x + y for x, y in zip(a, b)] for a, b in zip(self.dpos, other.dpos)],
            [[x + y for x, y in zip(a, b)] for a, b in zip(self.dvel, other.dvel)],
        )


ExternalForces = Sequence[Vec] | None


def empty_forces(num_bodies: int, dim: int) -> List[Vec]:
    return [zeros(dim) for _ in range(num_bodies)]
