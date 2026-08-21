"""Pure shared-clock estimator used by the ROS sensor timestamp boundary."""

from collections import deque
import math
from typing import Deque, Dict, Optional


class TimeSyncReset(RuntimeError):
    """The current message was rejected and the estimator returned to warmup."""


class OdomPoseGuard:
    """Reject physically implausible discontinuities in a stamped pose stream."""

    def __init__(
        self,
        max_translation_step: float = 0.75,
        max_translation_speed: float = 3.0,
        max_angular_step: float = 1.5707963267948966,
        max_angular_speed: float = 8.0,
        minimum_rate_interval_ns: int = 1000000,
    ) -> None:
        limits = (
            max_translation_step,
            max_translation_speed,
            max_angular_step,
            max_angular_speed,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in limits):
            raise ValueError("pose continuity limits must be finite and positive")
        if int(minimum_rate_interval_ns) < 0:
            raise ValueError("minimum_rate_interval_ns must not be negative")
        self.max_translation_step = float(max_translation_step)
        self.max_translation_speed = float(max_translation_speed)
        self.max_angular_step = float(max_angular_step)
        self.max_angular_speed = float(max_angular_speed)
        self.minimum_rate_interval_ns = int(minimum_rate_interval_ns)
        self._last_stamp_ns: Optional[int] = None
        self._last_position = None
        self._last_quaternion = None

    def observe(self, stamp_ns: int, position, quaternion) -> None:
        stamp_ns = int(stamp_ns)
        xyz = tuple(float(value) for value in position)
        q_raw = tuple(float(value) for value in quaternion)
        if stamp_ns <= 0:
            raise ValueError("odometry pose timestamp is zero or negative")
        if len(xyz) != 3 or len(q_raw) != 4:
            raise ValueError("odometry pose has an invalid shape")
        if not all(math.isfinite(value) for value in xyz + q_raw):
            raise ValueError("odometry pose is non-finite")
        q_norm = math.sqrt(sum(value * value for value in q_raw))
        if q_norm < 1.0e-9:
            raise ValueError("odometry quaternion has zero length")
        quaternion_unit = tuple(value / q_norm for value in q_raw)

        if self._last_stamp_ns is not None:
            delta_ns = stamp_ns - self._last_stamp_ns
            if delta_ns <= 0:
                raise ValueError("odometry pose timestamp did not progress")
            translation = math.sqrt(
                sum(
                    (current - previous) * (current - previous)
                    for current, previous in zip(xyz, self._last_position)
                )
            )
            dot = abs(
                sum(
                    current * previous
                    for current, previous in zip(
                        quaternion_unit, self._last_quaternion
                    )
                )
            )
            angle = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
            if translation > self.max_translation_step:
                raise ValueError(
                    "odometry translation step {:.3f}m exceeds {:.3f}m".format(
                        translation, self.max_translation_step
                    )
                )
            if angle > self.max_angular_step:
                raise ValueError(
                    "odometry angular step {:.3f}rad exceeds {:.3f}rad".format(
                        angle, self.max_angular_step
                    )
                )
            if delta_ns >= self.minimum_rate_interval_ns:
                delta_sec = delta_ns / 1.0e9
                if translation / delta_sec > self.max_translation_speed:
                    raise ValueError(
                        "odometry translation rate exceeds {:.3f}m/s".format(
                            self.max_translation_speed
                        )
                    )
                if angle / delta_sec > self.max_angular_speed:
                    raise ValueError(
                        "odometry angular rate exceeds {:.3f}rad/s".format(
                            self.max_angular_speed
                        )
                    )

        self._last_stamp_ns = stamp_ns
        self._last_position = xyz
        self._last_quaternion = quaternion_unit


class SharedTimeEstimator:
    """Normalize one robot clock onto host ROS time with a shared offset.

    Transport delay is non-negative once both clocks share an epoch, so the
    minimum of ``host_receipt - robot_stamp`` is the least-delayed offset sample.
    One rolling window is populated exclusively by odometry; every stream then
    receives the same offset.  The estimator never clamps or restamps a message.
    If adding the offset cannot produce a valid, strictly increasing stream, the
    complete clock epoch is invalidated and must warm up again.
    """

    def __init__(
        self,
        warmup_samples: int = 30,
        window_size: int = 200,
        clock_jump_threshold_ns: int = 1000000000,
        future_tolerance_ns: int = 250000000,
        max_output_age_ns: int = 5000000000,
    ) -> None:
        if warmup_samples <= 0:
            raise ValueError("warmup_samples must be positive")
        if window_size < warmup_samples:
            raise ValueError("window_size must be at least warmup_samples")
        if clock_jump_threshold_ns <= 0:
            raise ValueError("clock_jump_threshold_ns must be positive")
        if future_tolerance_ns < 0 or max_output_age_ns <= 0:
            raise ValueError("output age limits are invalid")
        self.warmup_samples = int(warmup_samples)
        self.window_size = int(window_size)
        self.clock_jump_threshold_ns = int(clock_jump_threshold_ns)
        self.future_tolerance_ns = int(future_tolerance_ns)
        self.max_output_age_ns = int(max_output_age_ns)
        self._delays: Deque[int] = deque(maxlen=self.window_size)
        self._last_source_ns: Dict[str, int] = {}
        self._last_receipt_ns: Dict[str, int] = {}
        self._last_output_ns: Dict[str, int] = {}
        self._offset_ns: Optional[int] = None
        self.reset_count = 0
        self.epoch = 0
        self.last_reset_reason = ""

    @property
    def ready(self) -> bool:
        return self._offset_ns is not None

    @property
    def offset_ns(self) -> Optional[int]:
        return self._offset_ns

    @property
    def warmup_collected(self) -> int:
        return min(len(self._delays), self.warmup_samples)

    def reset(self, reason: str, count: bool = True) -> None:
        """Invalidate all stream history so no old/new clock epochs can mix."""

        self._delays.clear()
        self._last_source_ns.clear()
        self._last_receipt_ns.clear()
        self._last_output_ns.clear()
        self._offset_ns = None
        self.epoch += 1
        self.last_reset_reason = str(reason)
        if count:
            self.reset_count += 1

    def _fail(self, reason: str) -> None:
        self.reset(reason)
        raise TimeSyncReset(reason)

    def _track(self, stream: str, source_ns: int, receipt_ns: int) -> None:
        stream = str(stream)
        source_ns = int(source_ns)
        receipt_ns = int(receipt_ns)
        if not stream:
            raise ValueError("stream must not be empty")
        if source_ns <= 0:
            self._fail("{} source timestamp is zero or negative".format(stream))
        if receipt_ns <= 0:
            self._fail("host receipt timestamp is zero or negative")

        previous_source = self._last_source_ns.get(stream)
        previous_receipt = self._last_receipt_ns.get(stream)
        if previous_source is not None:
            if source_ns <= previous_source:
                self._fail(
                    "{} source timestamp regressed or repeated".format(stream)
                )
            if receipt_ns < previous_receipt:
                self._fail("host clock regressed while receiving {}".format(stream))
            source_delta = source_ns - previous_source
            receipt_delta = receipt_ns - previous_receipt
            if abs(source_delta - receipt_delta) > self.clock_jump_threshold_ns:
                self._fail(
                    "clock-step mismatch detected on {}".format(stream)
                )

        self._last_source_ns[stream] = source_ns
        self._last_receipt_ns[stream] = receipt_ns

    def _normalized(self, stream: str, source_ns: int, receipt_ns: int) -> int:
        if self._offset_ns is None:
            raise RuntimeError("normalization requested before warmup")
        normalized_ns = int(source_ns) + self._offset_ns
        if normalized_ns <= 0:
            self._fail("normalized {} timestamp is not positive".format(stream))
        if normalized_ns > int(receipt_ns) + self.future_tolerance_ns:
            self._fail("normalized {} timestamp is in the future".format(stream))
        if int(receipt_ns) - normalized_ns > self.max_output_age_ns:
            self._fail("normalized {} timestamp is too old".format(stream))
        previous_output = self._last_output_ns.get(stream)
        if previous_output is not None and normalized_ns <= previous_output:
            self._fail(
                "normalized {} timestamp would not be monotonic".format(stream)
            )
        self._last_output_ns[stream] = normalized_ns
        return normalized_ns

    def process_odometry(
        self, source_ns: int, receipt_ns: int
    ) -> Optional[int]:
        """Observe one offset sample and return a stamp once warmup is complete."""

        self._track("odom", source_ns, receipt_ns)
        delay_ns = int(receipt_ns) - int(source_ns)
        self._delays.append(delay_ns)
        if len(self._delays) < self.warmup_samples:
            return None

        candidate_offset = min(self._delays)
        if (
            self._offset_ns is not None
            and abs(candidate_offset - self._offset_ns)
            > self.clock_jump_threshold_ns
        ):
            self._fail("rolling minimum offset jumped")
        self._offset_ns = candidate_offset
        return self._normalized("odom", source_ns, receipt_ns)

    def process_sensor(
        self, stream: str, source_ns: int, receipt_ns: int
    ) -> Optional[int]:
        """Normalize a non-odometry stream using only the shared odom offset."""

        if stream == "odom":
            raise ValueError("odometry must use process_odometry")
        self._track(stream, source_ns, receipt_ns)
        if self._offset_ns is None:
            return None
        return self._normalized(stream, source_ns, receipt_ns)

    def status(self) -> Dict[str, object]:
        return {
            "state": "locked" if self.ready else "warming",
            "epoch": self.epoch,
            "offset_ns": self._offset_ns,
            "warmup_collected": self.warmup_collected,
            "warmup_required": self.warmup_samples,
            "rolling_samples": len(self._delays),
            "rolling_window": self.window_size,
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "last_source_ns": dict(self._last_source_ns),
            "last_output_ns": dict(self._last_output_ns),
        }
