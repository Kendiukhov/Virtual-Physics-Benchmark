from __future__ import annotations

import math
import random
from typing import Iterable, List

Vec = List[float]


def zeros(dim: int) -> Vec:
    return [0.0 for _ in range(dim)]


def add(a: Vec, b: Vec) -> Vec:
    return [x + y for x, y in zip(a, b)]


def sub(a: Vec, b: Vec) -> Vec:
    return [x - y for x, y in zip(a, b)]


def scale(v: Vec, factor: float) -> Vec:
    return [x * factor for x in v]


def dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Vec) -> float:
    return math.sqrt(dot(v, v))


def distance(a: Vec, b: Vec) -> float:
    return norm(sub(a, b))


def unit(v: Vec, eps: float = 1e-9) -> Vec:
    n = norm(v)
    if n < eps:
        return zeros(len(v))
    return [x / n for x in v]


def copy_vec(v: Iterable[float]) -> Vec:
    return [float(x) for x in v]


def seed_everything(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
