"""Coordinate conversion at the Unitree sensor boundary.

The built-in LiDAR stack on this Go2 publishes its mounted native convention
under ROS-looking ``odom`` and ``base_link`` frame names.  Live observations
place the floor at positive Z.  A 180 degree rotation about X converts that
right-handed mounted convention to REP-103:

    (x, y, z)_rep103 = (x, -y, -z)_native

Both sides of an odometry pose use that basis, so rotations are conjugated by
the same rotation.  Point fields other than x/y/z and organized-cloud row
padding are preserved byte-for-byte.
"""

import math
from typing import Iterable, Sequence, Tuple

import numpy as np


POINT_FIELD_FLOAT32 = 7
AXIS_SIGNS = (1.0, -1.0, -1.0)
SIX_DOF_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0)
TRANSFORM_NAME = "native_mount_to_rep103_rx_pi"


def _finite_tuple(
    values: Iterable[float], size: int, label: str
) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError("{} must contain {} values".format(label, size))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("{} contains a non-finite value".format(label))
    return result


def transform_xyz(values: Iterable[float]) -> Tuple[float, float, float]:
    """Rotate one position or vector from the mounted native basis."""

    x, y, z = _finite_tuple(values, 3, "vector")
    return x, -y, -z


def transform_quaternion(
    values: Iterable[float],
) -> Tuple[float, float, float, float]:
    """Conjugate a quaternion by Rx(pi), returning a unit quaternion."""

    x, y, z, w = _finite_tuple(values, 4, "quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        raise ValueError("quaternion has zero length")
    # Rx(pi) * R(q) * Rx(pi)^-1 maps q to (x, -y, -z, w).
    return x / norm, -y / norm, -z / norm, w / norm


def transform_covariance(values: Sequence[float]) -> Tuple[float, ...]:
    """Apply block-diagonal Rx(pi) to a row-major 6x6 covariance."""

    covariance = _finite_tuple(values, 36, "covariance")
    return tuple(
        SIX_DOF_SIGNS[row]
        * covariance[row * 6 + column]
        * SIX_DOF_SIGNS[column]
        for row in range(6)
        for column in range(6)
    )


def _pointcloud_layout(message):
    width = int(message.width)
    height = int(message.height)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    if width < 0 or height < 0:
        raise ValueError("point cloud dimensions must not be negative")
    if point_step <= 0:
        raise ValueError("point cloud point_step must be positive")
    if row_step < width * point_step:
        raise ValueError("point cloud row_step is smaller than its point data")

    fields = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            if field.name in fields:
                raise ValueError(
                    "point cloud contains duplicate {} fields".format(
                        field.name
                    )
                )
            invalid_type = int(field.datatype) != POINT_FIELD_FLOAT32
            if invalid_type or int(field.count) != 1:
                raise ValueError(
                    "point cloud {} must be one FLOAT32 value".format(
                        field.name
                    )
                )
            fields[field.name] = int(field.offset)
    if set(fields) != {"x", "y", "z"}:
        raise ValueError("point cloud lacks scalar FLOAT32 x/y/z fields")
    intervals = sorted((offset, offset + 4) for offset in fields.values())
    overlaps = any(
        left[1] > right[0]
        for left, right in zip(intervals, intervals[1:])
    )
    if overlaps:
        raise ValueError("point cloud x/y/z fields overlap")
    outside = any(
        offset < 0 or offset + 4 > point_step
        for offset in fields.values()
    )
    if outside:
        raise ValueError("point cloud x/y/z field falls outside point_step")

    required_size = height * row_step
    return width, height, point_step, row_step, fields, required_size


def _transform_pointcloud_buffer(message, data) -> None:
    width, height, point_step, row_step, fields, required_size = (
        _pointcloud_layout(message)
    )
    view = memoryview(data)
    if view.readonly:
        raise ValueError("point cloud data buffer must be writable")
    if view.nbytes < required_size:
        raise ValueError("point cloud data is shorter than height*row_step")
    if width == 0 or height == 0:
        return
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    for axis in ("y", "z"):
        values = np.ndarray(
            shape=(height, width),
            dtype=dtype,
            buffer=view,
            offset=fields[axis],
            strides=(row_step, point_step),
        )
        np.negative(values, out=values)


def transform_pointcloud_in_place(message) -> None:
    """Negate y/z directly in a writable PointCloud2 data buffer.

    Foxy validates every element in the generated ``data`` property setter.
    Avoiding that setter is essential for the Go2's large deskewed cloud.
    """

    _transform_pointcloud_buffer(message, message.data)


def transform_pointcloud_data(message) -> bytearray:
    """Return transformed copy; primarily useful for pure unit tests."""

    data = bytearray(message.data)
    _transform_pointcloud_buffer(message, data)
    return data


def sample_pointcloud_xyz(message, max_points: int) -> np.ndarray:
    """Return bounded XYZ samples without copying the complete cloud."""

    width, height, point_step, row_step, fields, required_size = (
        _pointcloud_layout(message)
    )
    limit = int(max_points)
    if limit <= 0:
        raise ValueError("max_points must be positive")
    view = memoryview(message.data)
    if view.nbytes < required_size:
        raise ValueError("point cloud is shorter than height*row_step")
    point_count = width * height
    if point_count == 0:
        return np.empty((0, 3), dtype=np.float32)
    sample_count = min(point_count, limit)
    indices = np.linspace(
        0, point_count - 1, num=sample_count, dtype=np.int64
    )
    rows, columns = np.divmod(indices, width)
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    result = np.empty((sample_count, 3), dtype=np.float32)
    for output_column, axis in enumerate(("x", "y", "z")):
        values = np.ndarray(
            shape=(height, width),
            dtype=dtype,
            buffer=view,
            offset=fields[axis],
            strides=(row_step, point_step),
        )
        result[:, output_column] = values[rows, columns]
    return result


def transform_odometry_in_place(message) -> None:
    """Convert pose, twist, and both covariances on a ROS Odometry message."""

    pose = message.pose.pose
    px, py, pz = transform_xyz(
        (pose.position.x, pose.position.y, pose.position.z)
    )
    qx, qy, qz, qw = transform_quaternion(
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    )
    pose.position.x, pose.position.y, pose.position.z = px, py, pz
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = (qx, qy, qz, qw)
    message.pose.covariance = transform_covariance(message.pose.covariance)

    twist = message.twist.twist
    lx, ly, lz = transform_xyz(
        (twist.linear.x, twist.linear.y, twist.linear.z)
    )
    ax, ay, az = transform_xyz(
        (twist.angular.x, twist.angular.y, twist.angular.z)
    )
    twist.linear.x, twist.linear.y, twist.linear.z = lx, ly, lz
    twist.angular.x, twist.angular.y, twist.angular.z = ax, ay, az
    message.twist.covariance = transform_covariance(message.twist.covariance)
