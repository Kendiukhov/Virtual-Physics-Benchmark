from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .generator import EnvironmentSpec


@dataclass
class CurriculumStep:
    spec: EnvironmentSpec
    description: str


def linear_curriculum(
    levels: int = 5,
    start_dim: int = 2,
    max_dim: int = 4,
    start_bodies: int = 2,
    max_bodies: int = 4,
    base_steps: int = 200,
    max_steps: int = 400,
    complexities: Iterable[str] = ("easy", "medium", "hard"),
    seed: int | None = None,
) -> List[CurriculumStep]:
    levels = max(levels, 1)
    curriculum: List[CurriculumStep] = []
    for i in range(levels):
        frac = i / max(levels - 1, 1)
        dim = int(round(start_dim + frac * (max_dim - start_dim)))
        bodies = int(round(start_bodies + frac * (max_bodies - start_bodies)))
        steps = int(round(base_steps + frac * (max_steps - base_steps)))
        complexity_list = list(complexities)
        complexity = complexity_list[min(i, len(complexity_list) - 1)] if complexity_list else "medium"
        spec = EnvironmentSpec(dim=dim, num_bodies=bodies, steps=steps, complexity=complexity, seed=None if seed is None else seed + i)
        desc = f"Level {i+1}: {complexity}, dim={dim}, bodies={bodies}, steps={steps}"
        curriculum.append(CurriculumStep(spec, desc))
    return curriculum
