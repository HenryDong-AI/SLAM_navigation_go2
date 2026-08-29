"""ROS-independent depth projection and rigid-transform helpers."""

from typing import Sequence

import numpy as np


def rigid_transform(values: Sequence[float]) -> np.ndarray:
    """Validate and return a row-major rigid 4x4 transform."""
    transform = np.asarray(values, dtype=np.float64)
    if transform.size != 16 or not np.isfinite(transform).all():
        raise ValueError("transform must contain 16 finite values")
    transform = transform.reshape(4, 4)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError("transform has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError("transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError("transform rotation must have determinant +1")
    return transform.copy()


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a finite XYZW quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-9:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(position_xyz: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Return a homogeneous parent-from-child pose matrix."""
    position = np.asarray(position_xyz, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must contain three finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    result[:3, 3] = position
    return result


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an N-by-3 point array."""
    points = np.asarray(points_xyz, dtype=np.float64)
    matrix = rigid_transform(np.asarray(transform).reshape(-1))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def depth_to_camera_points(
    depth_metres: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    max_points: int,
) -> np.ndarray:
    """Deproject a sampled aligned depth image into its optical frame."""
    depth = np.asarray(depth_metres, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth image must be a non-empty 2D array")
    if not all(np.isfinite(value) and value > 0.0 for value in (fx, fy)):
        raise ValueError("camera focal lengths must be finite and positive")
    if not all(np.isfinite(value) for value in (cx, cy)):
        raise ValueError("camera principal point must be finite")
    if pixel_stride <= 0 or max_points <= 0:
        raise ValueError("sampling limits must be positive")
    if min_depth <= 0.0 or max_depth <= min_depth:
        raise ValueError("depth bounds are invalid")

    rows = np.arange(0, depth.shape[0], int(pixel_stride), dtype=np.int32)
    cols = np.arange(0, depth.shape[1], int(pixel_stride), dtype=np.int32)
    sampled = depth[np.ix_(rows, cols)]
    uu, vv = np.meshgrid(cols.astype(np.float32), rows.astype(np.float32))
    valid = np.isfinite(sampled)
    finite_depth = np.where(valid, sampled, 0.0)
    valid &= finite_depth >= float(min_depth)
    valid &= finite_depth <= float(max_depth)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)

    z = sampled[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    if points.shape[0] > max_points:
        stride = int(np.ceil(points.shape[0] / float(max_points)))
        points = points[::stride][:max_points]
    return points


def depth_to_camera_points_rgb(
    depth_metres: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    max_points: int,
):
    """Deproject aligned depth and retain the matching RGB pixel values."""

    depth = np.asarray(depth_metres, dtype=np.float32)
    color = np.asarray(color_bgr)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth image must be a non-empty 2D array")
    if color.shape != depth.shape + (3,) or color.dtype != np.uint8:
        raise ValueError("color image must be uint8 BGR matching depth size")
    if not all(np.isfinite(value) and value > 0.0 for value in (fx, fy)):
        raise ValueError("camera focal lengths must be finite and positive")
    if not all(np.isfinite(value) for value in (cx, cy)):
        raise ValueError("camera principal point must be finite")
    if pixel_stride <= 0 or max_points <= 0:
        raise ValueError("sampling limits must be positive")
    if min_depth <= 0.0 or max_depth <= min_depth:
        raise ValueError("depth bounds are invalid")

    rows = np.arange(0, depth.shape[0], int(pixel_stride), dtype=np.int32)
    cols = np.arange(0, depth.shape[1], int(pixel_stride), dtype=np.int32)
    sampled_depth = depth[np.ix_(rows, cols)]
    sampled_bgr = color[np.ix_(rows, cols)]
    uu, vv = np.meshgrid(cols.astype(np.float32), rows.astype(np.float32))
    valid = np.isfinite(sampled_depth)
    finite_depth = np.where(valid, sampled_depth, 0.0)
    valid &= finite_depth >= float(min_depth)
    valid &= finite_depth <= float(max_depth)
    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )

    z = sampled_depth[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    colors_rgb = sampled_bgr[valid][:, ::-1].copy()
    if points.shape[0] > max_points:
        stride = int(np.ceil(points.shape[0] / float(max_points)))
        points = points[::stride][:max_points]
        colors_rgb = colors_rgb[::stride][:max_points]
    return points, colors_rgb


def decode_depth_image(message, raw_depth_scale: float = 0.001) -> np.ndarray:
    """Decode a ROS Image-like depth message into float32 metres."""
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError("depth image dimensions and step must be positive")
    encoding = str(message.encoding).upper()
    if encoding == "32FC1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
        scale = 1.0
    elif encoding in ("16UC1", "MONO16"):
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
        scale = float(raw_depth_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("raw_depth_scale must be finite and positive")
    else:
        raise ValueError("unsupported depth encoding: {}".format(message.encoding))
    packed_width = width * dtype.itemsize
    if step < packed_width:
        raise ValueError("depth image step is smaller than its packed row")
    data = memoryview(message.data)
    required = (height - 1) * step + packed_width
    if len(data) < required:
        raise ValueError("depth image buffer is shorter than declared")
    rows = [
        np.frombuffer(
            data[row * step:row * step + packed_width], dtype=dtype, count=width
        )
        for row in range(height)
    ]
    return np.vstack(rows).astype(np.float32) * scale


def decode_color_image(message) -> np.ndarray:
    """Decode a ROS Image-like RGB/BGR message into a BGR uint8 array."""
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError("color image dimensions and step must be positive")
    encoding = str(message.encoding).lower()
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError("unsupported color encoding: {}".format(message.encoding))
    packed_width = width * 3
    if step < packed_width:
        raise ValueError("color image step is smaller than its packed row")
    data = memoryview(message.data)
    required = (height - 1) * step + packed_width
    if len(data) < required:
        raise ValueError("color image buffer is shorter than declared")
    rows = [
        np.frombuffer(
            data[row * step:row * step + packed_width],
            dtype=np.uint8,
            count=packed_width,
        ).reshape(width, 3)
        for row in range(height)
    ]
    image = np.stack(rows).copy()
    if encoding == "rgb8":
        image = image[:, :, ::-1].copy()
    return image
