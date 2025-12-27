from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .core import BodyState, WorldState
from .simulation import Trajectory
from .utils import copy_vec


def _body_to_dict(body: BodyState) -> Dict[str, Any]:
    return {"position": list(body.position), "velocity": list(body.velocity), "mass": body.mass}


def _body_from_dict(data: Dict[str, Any]) -> BodyState:
    return BodyState(copy_vec(data["position"]), copy_vec(data["velocity"]), float(data.get("mass", 1.0)))


def _state_to_dict(state: WorldState) -> Dict[str, Any]:
    return {"dim": state.dim, "bodies": [_body_to_dict(b) for b in state.bodies]}


def _state_from_dict(data: Dict[str, Any]) -> WorldState:
    return WorldState(int(data["dim"]), [_body_from_dict(b) for b in data["bodies"]])


def trajectory_to_dict(traj: Trajectory, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "dt": traj.dt,
        "initial": _state_to_dict(traj.initial),
        "times": traj.times,
        "states": [_state_to_dict(s) for s in traj.states],
        "forces": traj.forces,
        "meta": metadata or {},
    }


def trajectory_from_dict(data: Dict[str, Any]) -> Trajectory:
    initial = _state_from_dict(data["initial"])
    states = [_state_from_dict(s) for s in data["states"]]
    return Trajectory(
        dt=float(data["dt"]),
        initial=initial,
        times=[float(t) for t in data["times"]],
        states=states,
        forces=data["forces"],
    )


def save_trajectory_json(path: str | Path, traj: Trajectory, metadata: Dict[str, Any] | None = None) -> None:
    payload = trajectory_to_dict(traj, metadata=metadata)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_trajectory_json(path: str | Path) -> Trajectory:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return trajectory_from_dict(data)
