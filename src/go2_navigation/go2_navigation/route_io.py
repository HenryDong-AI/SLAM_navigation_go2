"""Validated, atomic route-file handling independent of ROS.

Copyright (c) 2026 Go2 SLAM Navigation Maintainers. MIT License.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import yaml


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float = 0.0
    wait: float = 0.0
    name: str = ""
    timeout: float = 120.0


@dataclass(frozen=True)
class Route:
    name: str
    frame_id: str
    loop: bool
    waypoints: List[Waypoint]


def _finite(value, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def parse_routes(data: Mapping) -> Dict[str, Route]:
    if not isinstance(data, Mapping):
        raise ValueError("route file root must be a mapping")
    raw_routes = data.get("routes")
    if not isinstance(raw_routes, Mapping) or not raw_routes:
        raise ValueError("route file must contain a non-empty 'routes' mapping")
    parsed: Dict[str, Route] = {}
    for route_name, raw_route in raw_routes.items():
        if not isinstance(route_name, str) or not route_name.strip():
            raise ValueError("route names must be non-empty strings")
        if not isinstance(raw_route, Mapping):
            raise ValueError(f"route {route_name!r} must be a mapping")
        frame_id = str(raw_route.get("frame_id", "map")).strip()
        if not frame_id or frame_id.startswith("/"):
            raise ValueError(f"route {route_name!r} has an invalid frame_id")
        raw_waypoints = raw_route.get("waypoints")
        if not isinstance(raw_waypoints, list) or not raw_waypoints:
            raise ValueError(f"route {route_name!r} needs at least one waypoint")
        waypoints = []
        for index, raw in enumerate(raw_waypoints):
            if not isinstance(raw, Mapping):
                raise ValueError(f"{route_name}.waypoints[{index}] must be a mapping")
            waypoint = Waypoint(
                x=_finite(raw.get("x"), f"{route_name}[{index}].x"),
                y=_finite(raw.get("y"), f"{route_name}[{index}].y"),
                yaw=_finite(raw.get("yaw", 0.0), f"{route_name}[{index}].yaw"),
                wait=max(0.0, _finite(raw.get("wait", 0.0), f"{route_name}[{index}].wait")),
                name=str(raw.get("name", f"waypoint_{index + 1}")),
                timeout=max(1.0, _finite(raw.get("timeout", 120.0), f"{route_name}[{index}].timeout")),
            )
            waypoints.append(waypoint)
        parsed[route_name] = Route(
            name=route_name,
            frame_id=frame_id,
            loop=bool(raw_route.get("loop", False)),
            waypoints=waypoints,
        )
    return parsed


def load_routes(path: os.PathLike) -> Dict[str, Route]:
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        return parse_routes(yaml.safe_load(stream))


def routes_to_dict(routes: Iterable[Route]) -> dict:
    result = {"routes": {}}
    for route in routes:
        result["routes"][route.name] = {
            "frame_id": route.frame_id,
            "loop": route.loop,
            "waypoints": [
                {
                    "name": point.name,
                    "x": round(point.x, 6),
                    "y": round(point.y, 6),
                    "yaw": round(point.yaw, 6),
                    "wait": round(point.wait, 3),
                    "timeout": round(point.timeout, 3),
                }
                for point in route.waypoints
            ],
        }
    return result


def atomic_write_routes(path: os.PathLike, routes: Iterable[Route]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(routes_to_dict(routes), sort_keys=False)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
