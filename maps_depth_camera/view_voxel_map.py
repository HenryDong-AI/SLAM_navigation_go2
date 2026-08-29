#!/home/unitree/SLAM_nav/.conda/envs/slam_nav/bin/python
"""View the newest saved Go2 RGB voxel map without ROS or RViz.

Examples::

    python view_voxel_map.py
    python view_voxel_map.py go2_map_20260828T094939_429781Z
    python view_voxel_map.py --save latest_voxel_map.png

The interactive window requires a graphical DISPLAY.  ``--save`` renders a
PNG without a display, which is convenient from SSH or an IDE terminal.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


MAP_ROOT = Path(__file__).resolve().parent


def newest_state_file():
    """Return the most recently modified complete snapshot state file."""

    candidates = [
        path
        for path in MAP_ROOT.glob("go2_map_*/state.npz")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "no go2_map_*/state.npz snapshots exist below {}".format(MAP_ROOT)
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def resolve_state_file(value):
    """Resolve a supplied snapshot directory or state.npz path."""

    if not value:
        return newest_state_file()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = MAP_ROOT / path
    if path.is_dir():
        path = path / "state.npz"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def load_voxels(path):
    """Load validated XYZ and optional RGB arrays without pickle."""

    with np.load(str(path), allow_pickle=False) as archive:
        if "voxel_centroids" not in archive:
            raise ValueError("snapshot does not contain voxel_centroids")
        points = np.asarray(archive["voxel_centroids"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("voxel_centroids must have shape (N, 3)")
        if points.shape[0] == 0 or not np.isfinite(points).all():
            raise ValueError(
                "voxel map is empty or contains invalid XYZ values"
            )

        colors = None
        if "voxel_colors" in archive:
            candidate = np.asarray(archive["voxel_colors"], dtype=np.float64)
            if (
                candidate.shape != points.shape
                or not np.isfinite(candidate).all()
            ):
                raise ValueError("voxel_colors do not match voxel_centroids")
            colors = np.clip(candidate, 0.0, 255.0) / 255.0

        metadata = {}
        if "metadata_json" in archive:
            raw = archive["metadata_json"]
            if raw.shape == ():
                metadata = json.loads(str(raw))
    return points, colors, metadata


def subsample(points, colors, max_points):
    """Deterministically limit rendering work while preserving the full map."""

    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    indices = np.linspace(
        0, points.shape[0] - 1, num=max_points, dtype=np.int64
    )
    selected_colors = colors[indices] if colors is not None else None
    return points[indices], selected_colors


def equalize_axes(axis, points):
    """Use equal metric scale on X, Y, and Z."""

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.5, 0.1)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def arguments():
    parser = argparse.ArgumentParser(
        description="View a saved XYZRGB Go2 voxel map without ROS or RViz."
    )
    parser.add_argument(
        "snapshot",
        nargs="?",
        help="snapshot directory or state.npz; default is the newest snapshot",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help=(
            "maximum displayed voxel centers; 0 displays all "
            "(default: 120000)"
        ),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="rendered point size in screen units (default: 1.5)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=28.0,
        help="initial camera elevation in degrees (default: 28)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=-62.0,
        help="initial camera azimuth in degrees (default: -62)",
    )
    parser.add_argument(
        "--save",
        metavar="PNG",
        help="save a PNG instead of opening an interactive window",
    )
    return parser.parse_args()


def main():
    args = arguments()
    if args.max_points < 0:
        raise ValueError("--max-points must be non-negative")
    if args.point_size <= 0.0:
        raise ValueError("--point-size must be positive")
    if not args.save and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "no graphical DISPLAY; reconnect with X11 forwarding or use "
            "--save latest_voxel_map.png"
        )

    # Select a non-GUI backend before pyplot is imported when rendering a file.
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_path = resolve_state_file(args.snapshot)
    all_points, all_colors, metadata = load_voxels(state_path)
    points, colors = subsample(all_points, all_colors, args.max_points)
    voxel_size = metadata.get("voxel_size", "unknown")

    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    if colors is None:
        plot = axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=args.point_size,
            marker="s",
            linewidths=0,
            depthshade=False,
        )
        figure.colorbar(plot, ax=axis, shrink=0.65, label="Z (m)")
        color_description = "height color"
    else:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=args.point_size,
            marker="s",
            linewidths=0,
            depthshade=False,
        )
        color_description = "fused RGB"

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.view_init(elev=args.elevation, azim=args.azimuth)
    equalize_axes(axis, points)
    axis.set_title(
        "Go2 voxel map: {:,}/{:,} displayed, voxel={} m, {}\n{}".format(
            points.shape[0],
            all_points.shape[0],
            voxel_size,
            color_description,
            state_path.parent.name,
        )
    )

    print("Loaded: {}".format(state_path))
    print("Voxels: {:,}; displayed: {:,}".format(
        all_points.shape[0], points.shape[0]
    ))
    print("Voxel size: {} m; colors: {}".format(
        voxel_size, "RGB" if colors is not None else "unavailable"
    ))
    if args.save:
        output = Path(args.save).expanduser().resolve()
        figure.savefig(str(output), dpi=180)
        print("Saved preview: {}".format(output))
        plt.close(figure)
    else:
        print("Mouse: drag to rotate; use the toolbar or scroll to zoom.")
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
