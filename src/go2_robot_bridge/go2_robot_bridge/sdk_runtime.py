"""Late-loading helpers for the non-ROS Unitree Python SDK."""

import importlib
import os
import site
import sys
from typing import Any, Iterable, Tuple


DEFAULT_SDK_PYTHON_PATH = "/home/unitree/Documents/demov1/unitree_sdk2_python"
DEFAULT_CYCLONEDDS_PYTHON_PATH = os.environ.get(
    "GO2_CYCLONEDDS_PYTHON", site.getusersitepackages()
)


def configure_python_paths(paths: Iterable[str]) -> None:
    """Prepend configured paths before importing Unitree or CycloneDDS."""
    normalized = []
    for path in paths:
        if not path:
            continue
        expanded = os.path.abspath(os.path.expanduser(str(path)))
        if expanded not in normalized:
            normalized.append(expanded)
    for path in reversed(normalized):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_camera_sdk(sdk_path: str, cyclonedds_python_path: str) -> Tuple[Any, Any]:
    configure_python_paths((sdk_path, cyclonedds_python_path))
    channel = importlib.import_module("unitree_sdk2py.core.channel")
    video = importlib.import_module("unitree_sdk2py.go2.video.video_client")
    return channel.ChannelFactoryInitialize, video.VideoClient


def load_motion_sdk(
    sdk_path: str, cyclonedds_python_path: str
) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    configure_python_paths((sdk_path, cyclonedds_python_path))
    channel = importlib.import_module("unitree_sdk2py.core.channel")
    sport = importlib.import_module("unitree_sdk2py.go2.sport.sport_client")
    robot_state = importlib.import_module(
        "unitree_sdk2py.go2.robot_state.robot_state_client"
    )
    motion_switcher = importlib.import_module(
        "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"
    )
    obstacles_avoid = importlib.import_module(
        "unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client"
    )
    sport_state = importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_")
    return (
        channel.ChannelFactoryInitialize,
        channel.ChannelSubscriber,
        sport.SportClient,
        robot_state.RobotStateClient,
        motion_switcher.MotionSwitcherClient,
        obstacles_avoid.ObstaclesAvoidClient,
        sport_state.SportModeState_,
    )


def no_shm_runtime_present(expected_fragment: str = "install_noshm/lib") -> bool:
    entries = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    return any(expected_fragment in entry for entry in entries if entry)
