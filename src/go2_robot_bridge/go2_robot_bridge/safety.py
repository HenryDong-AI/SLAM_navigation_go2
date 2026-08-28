"""Pure safety and PointCloud2 helpers used by the motion bridge.

This module deliberately has no ROS or Unitree SDK imports, which keeps the
interlock logic deterministic and straightforward to test off-robot.
"""

from dataclasses import dataclass
import math
import struct
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple


POINT_FIELD_FLOAT32 = 7


@dataclass(frozen=True)
class TwistCommand:
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0

    def is_zero(self, epsilon: float = 1.0e-4) -> bool:
        return (
            abs(self.linear_x) <= epsilon
            and abs(self.linear_y) <= epsilon
            and abs(self.angular_z) <= epsilon
        )


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class CloudSafetyObservation:
    healthy: bool
    front_blocked: bool
    plausible_points: int


class PermanentFaultLatch:
    """Remember the first restart-required motion safety fault."""

    def __init__(self) -> None:
        self.reason = ""

    @property
    def faulted(self) -> bool:
        return bool(self.reason)

    def latch(self, reason: str) -> str:
        if not self.reason:
            self.reason = str(reason).strip() or "unspecified safety fault"
        return self.reason


class TimeSyncInterlock:
    """Latch on a clock fault, epoch change, or bridge-process replacement."""

    def __init__(self) -> None:
        self.instance_id: Optional[str] = None
        self.epoch: Optional[int] = None
        self.state = "unseen"
        self.ever_locked = False
        self.fault_reason = ""

    @property
    def ready(self) -> bool:
        return self.state == "locked" and not self.fault_reason

    def latch(self, reason: str) -> str:
        if not self.fault_reason:
            self.fault_reason = str(reason)
        self.state = "fault_latched"
        return self.fault_reason

    def update(self, payload: Mapping[str, Any]) -> str:
        if self.fault_reason:
            return self.fault_reason
        if not isinstance(payload, Mapping):
            return self.latch("time-sync status is not an object")
        state = str(payload.get("state", ""))
        instance_id = str(payload.get("instance_id", "")).strip()
        try:
            epoch = int(payload.get("epoch"))
        except (TypeError, ValueError):
            return self.latch("time-sync status has an invalid epoch")
        if state not in ("warming", "locked", "fault_latched"):
            return self.latch("time-sync status has an invalid state")
        if not instance_id or epoch < 0:
            return self.latch("time-sync status identity is invalid")
        if self.instance_id is None:
            self.instance_id = instance_id
            self.epoch = epoch
        elif instance_id != self.instance_id:
            return self.latch("sensor time bridge process changed")
        if state == "fault_latched":
            return self.latch(
                str(payload.get("fault_reason", "")).strip()
                or "sensor time bridge reported a fault"
            )
        if epoch != self.epoch:
            return self.latch("sensor time epoch changed")
        if self.ever_locked and state != "locked":
            return self.latch("sensor time bridge left the locked state")
        self.state = state
        if state == "locked":
            self.ever_locked = True
        return ""


def finite_or_zero(value: Any) -> float:
    """Return a finite float, replacing NaN, infinity, and bad input with 0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def symmetric_clamp(value: Any, maximum: float) -> float:
    value_f = finite_or_zero(value)
    limit = max(0.0, finite_or_zero(maximum))
    return max(-limit, min(limit, value_f))


def clamp_twist(
    command: TwistCommand,
    max_linear_x: float,
    max_linear_y: float,
    max_angular_z: float,
) -> TwistCommand:
    return TwistCommand(
        symmetric_clamp(command.linear_x, max_linear_x),
        symmetric_clamp(command.linear_y, max_linear_y),
        symmetric_clamp(command.angular_z, max_angular_z),
    )


def slew_value(current: float, target: float, rate_per_second: float, dt: float) -> float:
    """Move ``current`` toward ``target`` without exceeding a rate limit."""
    current_f = finite_or_zero(current)
    target_f = finite_or_zero(target)
    rate = max(0.0, finite_or_zero(rate_per_second))
    elapsed = max(0.0, finite_or_zero(dt))
    maximum_delta = rate * elapsed
    delta = target_f - current_f
    if abs(delta) <= maximum_delta:
        return target_f
    if maximum_delta == 0.0:
        return current_f
    return current_f + math.copysign(maximum_delta, delta)


def slew_twist(
    current: TwistCommand,
    target: TwistCommand,
    linear_rate: float,
    angular_rate: float,
    dt: float,
) -> TwistCommand:
    return TwistCommand(
        slew_value(current.linear_x, target.linear_x, linear_rate, dt),
        slew_value(current.linear_y, target.linear_y, linear_rate, dt),
        slew_value(current.angular_z, target.angular_z, angular_rate, dt),
    )


def _is_fresh(now: float, received_at: Optional[float], timeout: float) -> bool:
    if received_at is None:
        return False
    age = finite_or_zero(now) - finite_or_zero(received_at)
    return -0.1 <= age <= max(0.0, finite_or_zero(timeout))


def source_timestamp_is_fresh(
    *,
    now_nanoseconds: int,
    stamp_sec: int,
    stamp_nanosec: int,
    timeout_sec: float,
    future_tolerance_sec: float = 0.1,
) -> bool:
    """Validate a nonzero source timestamp against the ROS clock."""
    try:
        now_ns = int(now_nanoseconds)
        seconds = int(stamp_sec)
        nanoseconds = int(stamp_nanosec)
    except (TypeError, ValueError):
        return False
    if now_ns <= 0 or seconds < 0 or not 0 <= nanoseconds < 1000000000:
        return False
    stamp_ns = seconds * 1000000000 + nanoseconds
    if stamp_ns <= 0:
        return False
    age_sec = (now_ns - stamp_ns) / 1.0e9
    return (
        -max(0.0, finite_or_zero(future_tolerance_sec))
        <= age_sec
        <= max(0.0, finite_or_zero(timeout_sec))
    )


def evaluate_motion_gate(
    *,
    armed: bool,
    command: TwistCommand,
    now: float,
    last_command_at: Optional[float],
    command_timeout: float,
    require_fresh_odom: bool,
    last_odom_at: Optional[float],
    odom_timeout: float,
    obstacle_guard_enabled: bool,
    front_blocked: bool,
    last_cloud_at: Optional[float],
    cloud_timeout: float,
    fail_closed_on_cloud_timeout: bool,
) -> GateDecision:
    """Evaluate all motion interlocks in an intentionally strict order."""
    if not armed:
        return GateDecision(False, "disarmed")
    if not _is_fresh(now, last_command_at, command_timeout):
        return GateDecision(False, "command_timeout")
    if require_fresh_odom and not _is_fresh(now, last_odom_at, odom_timeout):
        return GateDecision(False, "odometry_stale")
    if command.is_zero():
        return GateDecision(False, "zero_command")

    # Missing/stale safety perception stops every autonomous command. A fresh
    # but blocked forward sector still permits bounded reverse/turn escape.
    if obstacle_guard_enabled:
        cloud_fresh = _is_fresh(now, last_cloud_at, cloud_timeout)
        if fail_closed_on_cloud_timeout and not cloud_fresh:
            return GateDecision(False, "front_cloud_stale")
        if command.linear_x > 0.0 and front_blocked:
            return GateDecision(False, "front_obstacle")

    return GateDecision(True, "motion_allowed")


def _field_value(field: Any, name: str) -> Any:
    if isinstance(field, Mapping):
        return field[name]
    return getattr(field, name)


def point_field_offsets(fields: Sequence[Any]) -> Dict[str, int]:
    """Validate and return x/y/z FLOAT32 offsets from PointCloud2 fields."""
    found: Dict[str, Tuple[int, int, int]] = {}
    for field in fields:
        field_name = str(_field_value(field, "name"))
        if field_name in ("x", "y", "z"):
            found[field_name] = (
                int(_field_value(field, "offset")),
                int(_field_value(field, "datatype")),
                int(_field_value(field, "count")),
            )
    missing = [axis for axis in ("x", "y", "z") if axis not in found]
    if missing:
        raise ValueError("PointCloud2 is missing fields: " + ", ".join(missing))
    for axis, (offset, datatype, count) in found.items():
        if offset < 0 or datatype != POINT_FIELD_FLOAT32 or count != 1:
            raise ValueError("PointCloud2 field %s must be scalar FLOAT32" % axis)
    return {axis: found[axis][0] for axis in ("x", "y", "z")}


def iter_xyz(
    *,
    data: Any,
    width: int,
    height: int,
    point_step: int,
    row_step: int,
    is_bigendian: bool,
    fields: Sequence[Any],
    sample_stride: int = 1,
) -> Iterator[Tuple[float, float, float]]:
    """Iterate finite XYZ points while respecting row padding and endianness."""
    offsets = point_field_offsets(fields)
    point_step_i = int(point_step)
    width_i = max(0, int(width))
    height_i = max(0, int(height))
    row_step_i = int(row_step) if int(row_step) > 0 else width_i * point_step_i
    stride_i = max(1, int(sample_stride))
    if point_step_i <= 0:
        raise ValueError("PointCloud2 point_step must be positive")
    if height_i > 0 and row_step_i < width_i * point_step_i:
        raise ValueError("PointCloud2 row_step is smaller than width * point_step")
    largest_end = max(offsets.values()) + 4
    if largest_end > point_step_i:
        raise ValueError("PointCloud2 XYZ fields exceed point_step")

    view = memoryview(bytes(data))
    if width_i > 0 and height_i > 0:
        required_bytes = (height_i - 1) * row_step_i + width_i * point_step_i
        if len(view) < required_bytes:
            raise ValueError("PointCloud2 data is shorter than its declared layout")
    unpack = struct.Struct((">" if is_bigendian else "<") + "f").unpack_from
    for row in range(height_i):
        row_base = row * row_step_i
        for column in range(0, width_i, stride_i):
            base = row_base + column * point_step_i
            x = unpack(view, base + offsets["x"])[0]
            y = unpack(view, base + offsets["y"])[0]
            z = unpack(view, base + offsets["z"])[0]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                yield x, y, z


def front_sector_blocked(
    points: Iterator[Tuple[float, float, float]],
    *,
    min_x: float,
    max_x: float,
    half_width: float,
    min_z: float,
    max_z: float,
    min_points: int,
) -> bool:
    """Return true once enough points occupy the configured front prism.

    Input is the normalized REP-103 base cloud: X forward, Y left, and Z up.
    Configure the lower bound above the observed floor plane.
    """
    required = max(1, int(min_points))
    hits = 0
    for x, y, z in points:
        if min_x <= x <= max_x and abs(y) <= half_width and min_z <= z <= max_z:
            hits += 1
            if hits >= required:
                return True
    return False


def assess_front_cloud(
    points: Iterator[Tuple[float, float, float]],
    *,
    min_x: float,
    max_x: float,
    half_width: float,
    min_z: float,
    max_z: float,
    obstacle_min_points: int,
    health_min_points: int,
    health_min_range: float,
    health_max_range: float,
) -> CloudSafetyObservation:
    """Assess obstacle occupancy and whole-cloud health in one pass.

    Near-zero placeholders, non-finite points (already omitted by ``iter_xyz``),
    and implausibly distant returns never refresh the safety-cloud watchdog.
    Floor and rear returns still count as sensor-health evidence.
    """

    obstacle_required = max(1, int(obstacle_min_points))
    health_required = max(1, int(health_min_points))
    minimum_range = max(0.0, finite_or_zero(health_min_range))
    maximum_range = max(minimum_range, finite_or_zero(health_max_range))
    minimum_squared = minimum_range * minimum_range
    maximum_squared = maximum_range * maximum_range
    plausible = 0
    obstacle_hits = 0
    for x, y, z in points:
        squared_range = x * x + y * y + z * z
        if squared_range < minimum_squared or squared_range > maximum_squared:
            continue
        plausible += 1
        if min_x <= x <= max_x and abs(y) <= half_width and min_z <= z <= max_z:
            obstacle_hits += 1
    return CloudSafetyObservation(
        healthy=plausible >= health_required,
        front_blocked=obstacle_hits >= obstacle_required,
        plausible_points=plausible,
    )
