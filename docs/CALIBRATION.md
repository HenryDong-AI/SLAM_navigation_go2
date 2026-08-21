# Camera and LiDAR calibration

Accurate 3D semantic mapping requires calibration of the exact robot. The Go2
camera RPC returns an image but does not publish factory `CameraInfo`, and no
camera-to-L1 calibration file was present on this device. Do not treat the
example numbers in this project as calibrated values.

## 1. Camera intrinsics

1. Print a large checkerboard and measure the square size accurately.
2. Start only the camera bridge and view `/go2/camera/image_raw`; use
   `/go2/camera/image_rect` for projection QA.
3. Capture at least 25 sharp images covering the center, all edges, multiple
   distances, and several board tilts.
4. Calibrate the full 1920×1080 stream with OpenCV or ROS camera calibration.
   Use a fisheye/equidistant model if a pinhole/radtan fit leaves strongly
   curved residuals near the edges.
5. Copy the resulting width, height, distortion model, `D`, `K`, `R`, and `P`
   into `src/go2_robot_bridge/config/camera_info.yaml`.

The semantic node consumes `/go2/camera/image_rect`, so set `fx`, `fy`, `cx`,
and `cy` from the rectified `P` matrix. Its live `CameraInfo` gate checks that
frame, resolution, and `P` agree before projection is enabled.

## 2. Base-to-camera optical transform

The semantic mapper expects a 4×4 homogeneous matrix that converts points from
`base_link` into the camera optical convention (`x` right, `y` down, `z`
forward). Determine it with a target visible in both the L1 point cloud and the
camera, using a standard LiDAR-camera calibration package or a carefully
measured target plus `cv2.solvePnP`.

Do not substitute an axis permutation for calibration. `sensor_time_bridge`
first rotates the mounted native L1 output into REP-103 `base_link` coordinates
(X forward, Y left, Z up). Calibrate the camera transform against the normalized
`/go2/lidar/cloud_base`, never raw `/utlidar/cloud_base`, and validate the
complete rigid transform.

## 3. QA before semantic recording

1. Keep the robot stationary and view the semantic debug image.
2. Confirm projected LiDAR points land on the same wall/door/object edges from
   the image center to all corners.
3. Repeat at 1 m, 3 m, and 6 m.
4. Slowly rotate in place and confirm overlay error does not grow. Growing
   error indicates timestamp skew, not just extrinsic error.
5. Only then set `calibration_confirmed: true` in the semantic configuration.

The built-in camera and LiDAR are not hardware synchronized. The time bridge
first maps the native LiDAR/odometry clock into the Jetson ROS clock. The
camera RPC has no exposure timestamp, so its bridge uses the RPC midpoint.
Measure residual temporal error while slowly rotating, then set one correction:
prefer `camera_time_offset_sec` in the semantic configuration (negative when
the image corresponds to an earlier LiDAR sample). Keep the camera bridge's
`timestamp_offset_sec` at zero unless every camera consumer needs the same
correction. Keep inference rates modest and robot speed low while recording.
