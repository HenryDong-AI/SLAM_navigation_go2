<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Go2 Semantic Mapping contributors -->

# Go2 Semantic Mapping

This ROS 2 Foxy `ament_python` package fuses the Unitree Go2 built-in RGB stream,
a base-frame LiDAR cloud, and robot odometry into a bounded, persistent 3D
semantic voxel map. It is strictly perception-only: it does not publish velocity,
Sport API, posture, or any other motion command.

## Calibration gate

Semantic projection is disabled by default. Before enabling it, measure:

1. Camera intrinsics `fx`, `fy`, `cx`, and `cy` for the exact image resolution.
2. The rigid matrix `T_camera_optical_from_base`, where optical axes are
   x-right, y-down, z-forward and
   `p_camera_optical = T_camera_optical_from_base * p_base`.
3. The time alignment between image, `/go2/lidar/cloud_base`, and `/go2/odom`.

Enter those values in `config/semantic_mapping.yaml` and set
`calibration_confirmed: true`. The identity transform is only a placeholder.
Until calibration is confirmed, the node can fuse unknown geometry but will not
project camera labels or colors onto it.

## Inputs and outputs

Default inputs:

- `/go2/camera/image_rect` (`sensor_msgs/Image`), supplied by the built-in Go2
  camera bridge.
- `/go2/lidar/cloud_base` (`sensor_msgs/PointCloud2`) with XYZ already expressed in
  REP-103 `base_link` (X forward, Y left, Z up).
- `/go2/odom` (`nav_msgs/Odometry`) with pose `odom -> base_link`.

Outputs:

- `/go2/semantic/cloud`: `PointCloud2` with fields `x`, `y`, `z`, `rgb`,
  `label`, and `confidence`. Label 0 means unknown; Ultralytics class N is stored
  as N+1.
- `/go2/semantic/markers`: bounded `MarkerArray` text labels.
- `/go2/semantic/annotated_image`: detector boxes/masks and status overlay.

When calibration is enabled, the annotated image also shows a bounded sparse
sample of projected LiDAR points colored by relative depth, even when no detector
or model is configured. Use this overlay to validate camera intrinsics and the
base-to-optical extrinsic before trusting semantic labels. Its cost is capped by
`max_projected_points_overlay`.

The image callback is rate-limited and pairs each image with the
timestamp-nearest buffered cloud, then the timestamp-nearest odometry inside
configurable maximum deltas. A
median/MAD depth gate rejects background and foreground points inside detection
regions. Segmentation-capable Ultralytics models are preferred because their
masks reduce box-background contamination.

## Detector behavior

Ultralytics is optional. With `detector_enabled: false` or an empty
`detector_model`, the package never imports Ultralytics and continues as a
geometry-only mapper. On this Jetson, `scripts/create_conda_env.sh` creates the
project-local Python 3.8 runtime and `scripts/activate_conda.sh` exports it as
`GO2_SEMANTIC_PYTHON`. The launch file uses that interpreter so Foxy
`rclpy`/`cv_bridge`, Jetson Torch, and Ultralytics coexist; without activation,
it falls back to the device's validated `venv-yolo`. The configured
local model is `/home/unitree/Documents/demov1/yolov8n.pt`; aliases that trigger
network downloads are not used.

## Build and run

```bash
cd ~/go2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select go2_semantic_mapping
source install/setup.bash
ros2 launch go2_semantic_mapping semantic_mapping.launch.py
```

Override device-specific topic names at launch time:

```bash
ros2 launch go2_semantic_mapping semantic_mapping.launch.py \
  image_topic:=/go2/camera/image_rect \
  cloud_topic:=/go2/lidar/cloud_base \
  odom_topic:=/go2/odom
```

The sensor time bridge applies one shared host-clock offset to native cloud and
odometry stamps. The camera bridge stamps the midpoint of the blocking image
RPC. The semantic node deliberately rejects associations outside the configured
time bounds; calibrate `camera_time_offset_sec` for motion-sensitive projection.
The node also binds its map to one `sensor_time_bridge` process and epoch.
An epoch/process change, explicit clock fault, or stale status clears the map,
invalidates in-flight inference, and requires a complete-stack restart.

## Persistence and reset

```bash
ros2 service call /go2/semantic/save std_srvs/srv/Trigger '{}'
ros2 service call /go2/semantic/reset std_srvs/srv/Trigger '{}'
```

Save creates a uniquely named bundle under `save_directory`. `semantic_map.ply`
contains voxel position, RGB, numeric label, and confidence. `semantic_map.json`
contains class names, observation counts, timestamps, voxel keys, and complete
per-class votes. Both files are fsynced in a temporary directory and made visible
together with one atomic directory rename.

The map defaults to 100,000 bounded voxels; least-recently-observed voxels are evicted.
`max_cloud_points_per_fusion`, `inference_rate_hz`, and `max_label_markers` bound
CPU, memory, and visualization load.

## ROS-independent tests

```bash
python3 -m pytest -q test/test_core.py
```

The tests cover projection, depth-gated association, odometry transforms, bounded
voxel voting, and atomic PLY/JSON persistence without importing ROS.
