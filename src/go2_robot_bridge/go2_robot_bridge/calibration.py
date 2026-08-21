"""Camera calibration parsing without ROS dependencies."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class CameraCalibration:
    camera_name: str
    width: int
    height: int
    distortion_model: str
    d: Tuple[float, ...]
    k: Tuple[float, ...]
    r: Tuple[float, ...]
    p: Tuple[float, ...]

    @property
    def calibrated(self) -> bool:
        return len(self.k) == 9 and self.k[0] > 0.0 and self.k[4] > 0.0


def _matrix_data(document: Mapping[str, Any], key: str, size: int) -> Tuple[float, ...]:
    value = document.get(key, {})
    if isinstance(value, Mapping):
        value = value.get("data", [])
    if value is None:
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("%s.data must be a numeric sequence" % key)
    numbers = tuple(float(item) for item in value)
    if numbers and len(numbers) != size:
        raise ValueError("%s.data must contain %d values" % (key, size))
    return numbers if numbers else tuple(0.0 for _ in range(size))


def calibration_from_mapping(document: Mapping[str, Any]) -> CameraCalibration:
    if not isinstance(document, Mapping):
        raise ValueError("camera calibration must be a YAML mapping")
    distortion = document.get("distortion_coefficients", {})
    if isinstance(distortion, Mapping):
        distortion = distortion.get("data", [])
    if distortion is None:
        distortion = []
    if not isinstance(distortion, Sequence) or isinstance(distortion, (str, bytes)):
        raise ValueError("distortion_coefficients.data must be a numeric sequence")
    return CameraCalibration(
        camera_name=str(document.get("camera_name", "go2_front_camera")),
        width=max(0, int(document.get("image_width", 0))),
        height=max(0, int(document.get("image_height", 0))),
        distortion_model=str(document.get("distortion_model", "plumb_bob")),
        d=tuple(float(item) for item in distortion),
        k=_matrix_data(document, "camera_matrix", 9),
        r=_matrix_data(document, "rectification_matrix", 9),
        p=_matrix_data(document, "projection_matrix", 12),
    )


def load_camera_calibration(path: str) -> CameraCalibration:
    import yaml

    with open(path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return calibration_from_mapping(document)
