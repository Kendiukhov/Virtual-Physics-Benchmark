from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class EquivalenceResult:
    equivalent: bool
    symbolic_zero: bool
    max_error: float
    mean_error: float


def _ensure_sympy():
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("sympy is required for symbolic equivalence checks") from exc
    return sp


def _coerce_symbols(symbols: Sequence[object], sp) -> List[object]:
    resolved = []
    for sym in symbols:
        if isinstance(sym, str):
            resolved.append(sp.symbols(sym))
        else:
            resolved.append(sym)
    return resolved


def _coerce_expr(expr: object, syms: Sequence[object], sp):
    if isinstance(expr, str):
        local = {s.name: s for s in syms}
        return sp.sympify(expr, locals=local)
    return sp.sympify(expr)


def symbolic_equivalence(
    true_expr: object,
    candidate_expr: object,
    symbols: Sequence[object],
    samples: int = 32,
    domain: tuple[float, float] = (-2.0, 2.0),
    tol: float = 1e-3,
) -> EquivalenceResult:
    sp = _ensure_sympy()
    syms = _coerce_symbols(symbols, sp)
    t_expr = _coerce_expr(true_expr, syms, sp)
    c_expr = _coerce_expr(candidate_expr, syms, sp)

    diff = sp.simplify(t_expr - c_expr)
    symbolic_zero = diff == 0

    errors = []
    for _ in range(max(samples, 1)):
        vals = {sym: random.uniform(domain[0], domain[1]) for sym in syms}
        val = float(abs(diff.evalf(subs=vals)))
        errors.append(val)

    max_err = max(errors) if errors else 0.0
    mean_err = sum(errors) / len(errors) if errors else 0.0
    equivalent = symbolic_zero or max_err < tol
    return EquivalenceResult(equivalent=equivalent, symbolic_zero=symbolic_zero, max_error=max_err, mean_error=mean_err)


def vector_symbolic_equivalence(
    true_exprs: Iterable[object],
    candidate_exprs: Iterable[object],
    symbols: Sequence[object],
    samples: int = 32,
    domain: tuple[float, float] = (-2.0, 2.0),
    tol: float = 1e-3,
) -> EquivalenceResult:
    true_list = list(true_exprs)
    cand_list = list(candidate_exprs)
    if len(true_list) != len(cand_list):
        raise ValueError("true_exprs and candidate_exprs must have the same length")

    parts = [
        symbolic_equivalence(t_expr, c_expr, symbols, samples=samples, domain=domain, tol=tol)
        for t_expr, c_expr in zip(true_list, cand_list)
    ]
    max_err = max((p.max_error for p in parts), default=0.0)
    mean_err = sum(p.mean_error for p in parts) / len(parts) if parts else 0.0
    equivalent = all(p.equivalent for p in parts)
    symbolic_zero = all(p.symbolic_zero for p in parts)
    return EquivalenceResult(equivalent=equivalent, symbolic_zero=symbolic_zero, max_error=max_err, mean_error=mean_err)
