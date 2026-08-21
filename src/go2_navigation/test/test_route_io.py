from pathlib import Path

import pytest

from go2_navigation.route_io import Route, Waypoint, atomic_write_routes, load_routes, parse_routes


def test_parse_route_and_defaults():
    routes = parse_routes({"routes": {"lab": {"waypoints": [{"x": 1, "y": 2}]}}})
    assert routes["lab"].frame_id == "map"
    assert routes["lab"].waypoints[0].timeout == 120.0


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"routes": {}},
        {"routes": {"x": {"waypoints": []}}},
        {"routes": {"x": {"frame_id": "/map", "waypoints": [{"x": 0, "y": 0}]}}},
        {"routes": {"x": {"waypoints": [{"x": float("nan"), "y": 0}]}}},
    ],
)
def test_invalid_routes_rejected(bad):
    with pytest.raises(ValueError):
        parse_routes(bad)


def test_atomic_round_trip(tmp_path: Path):
    destination = tmp_path / "routes.yaml"
    route = Route("r", "odom", False, [Waypoint(1.2, -0.4, name="one")])
    atomic_write_routes(destination, [route])
    loaded = load_routes(destination)
    assert loaded["r"].waypoints[0].name == "one"
    assert loaded["r"].waypoints[0].x == pytest.approx(1.2)
