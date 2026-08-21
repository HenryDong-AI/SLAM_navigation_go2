# Go2 Robot Bridge

`go2_robot_bridge` is a clean-room ROS 2 Foxy package for the built-in Unitree
Go2 sensors, front camera, and high-level velocity API. It contains three
separate processes so camera/RPC failures cannot share mutable state with
motion safety or sensor-clock state:

- `sensor_time_bridge` converts the built-in LiDAR/odometry native clock to
  host ROS time before any safety, recording, mapping, or navigation consumer.

- `camera_bridge` retrieves JPEG samples with Unitree `VideoClient` in a
  private non-ROS subprocess, then publishes `sensor_msgs/CompressedImage`,
  decoded `sensor_msgs/Image`, and `sensor_msgs/CameraInfo`. Process
  isolation is required because the Unitree Python CycloneDDS binding and
  Foxy's ROS CycloneDDS RMW cannot create a domain safely in one process on
  this device.
- `motion_bridge` translates fresh, bounded `geometry_msgs/Twist` commands to
  `SportClient.Move`. It starts **DISARMED**, never stands the robot, and stops
  on stale commands, stale odometry, a forward obstacle, malformed/stale safety
  cloud data, SDK failure, an explicit stop request, or shutdown. Because
  Unitree's Python CycloneDDS binding and `rclpy` cannot safely initialize DDS
  domains in one process on this Go2, the ROS node never imports that SDK. It
  creates a private, non-ROS worker only after an explicit successful enable
  request.

This package does not copy implementation code from Unitree demos or the
reference projects. It uses their public runtime interfaces only and is
licensed under MIT.

## Critical hardware safety

Keep the physical remote/E-stop in an operator's hand whenever this node can
reach the robot. Commission with the Go2 supported, in a clear stair-free
area, at low speed. Software guards are not a substitute for the physical
stop. The bridge deliberately does **not** call `StandUp`, `RecoveryStand`, or
`BalanceStand`; put the robot in a stable standing mode with the official
controller before arming this bridge.

Calling `/go2/motion/stop` is latching and disarms motion. Re-enabling always
clears the cached command and requires a new `/cmd_vel` message.

The motion worker receives bounded, length-framed JSON over a private inherited
socket. Its dispatcher accepts only `Move(x, y, yaw)` and `StopMove()`; there is
no remote path to standing, posture, or mode APIs. Every value is checked for
finiteness. RPCs, startup, and process reaping are bounded by timeout, and the
worker makes a final best-effort `StopMove` on parent EOF. An unconfirmed
`StopMove` or any SDK/transport failure after arming latches a restart-required
safety fault. A disarmed bridge has no motion worker process.

## Mandatory CycloneDDS runtime

On this Jetson, `/usr/local/lib/libddsc.so` was built with iceoryx shared-memory
transport and can crash on Unitree SDK writes/RPC calls. The proven
`ENABLE_SHM=OFF` build must be first in `LD_LIBRARY_PATH` **before Python or ROS
starts**. Changing `LD_LIBRARY_PATH` inside a running node is too late.

```bash
export GO2_SDK_ROOT=/home/unitree/Documents/demov1/unitree_sdk2_python
export LD_LIBRARY_PATH="$GO2_SDK_ROOT/cyclonedds/install_noshm/lib:${LD_LIBRARY_PATH:-}"
export GO2_CYCLONEDDS_PYTHON="$(python3 -m site --user-site)"
export PYTHONPATH="$GO2_SDK_ROOT:$GO2_CYCLONEDDS_PYTHON:${PYTHONPATH:-}"

source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml
```

Both nodes refuse to initialize the Unitree client when the no-SHM path is not
visible. Set `require_noshm_runtime:=false` only on a machine whose CycloneDDS
build has independently been verified safe.

`GO2_CYCLONEDDS_PYTHON` is account-specific. When launching from a service or
another login, set it to that account's installed CycloneDDS Python directory;
`scripts/doctor.py` rejects an unusable path before launch.

## Build and launch

From the containing ROS workspace:

```bash
colcon build --symlink-install --packages-select go2_robot_bridge
source install/setup.bash
ros2 launch go2_robot_bridge robot_bridge.launch.py network_interface:=eth0
```

Useful camera parameters include `publish_rate_hz`, `image_topic`,
`compressed_topic`, `camera_info_topic`, `frame_id`, `calibration_file`,
and `worker_watchdog_sec`. The bridge keeps only the newest bounded frame,
rejects stale generations, and restarts a child that stops making IPC progress.
The supplied `config/camera_info.yaml` is intentionally uncalibrated. Replace
it with calibration for the exact camera and resolution before semantic 3D
projection; camera-to-LiDAR extrinsics must also be calibrated separately.
Until then, `/go2/camera/image_rect` is explicitly an unrectified passthrough
for geometry-only operation; the semantic projection gate remains closed.

The launch file loads motion limits and interlocks from `config/safety.yaml`.
The obstacle guard consumes normalized `/go2/lidar/cloud_base` as a standard
PointCloud2, and its odometry interlock consumes `/go2/odom`.
The time boundary rotates the built-in sensor's mounted convention 180 degrees
about X, so all `/go2` outputs use REP-103: X forward, Y left, and Z up. Its
default vertical band excludes floor returns rather than treating them as
obstacles.

### Native sensor clock boundary

`robot_bridge.launch.py` always starts `sensor_time_bridge`. It requests the
firmware's best-effort native streams `/utlidar/robot_odom`,
`/utlidar/cloud_base`, and `/utlidar/cloud_deskewed`, estimates one shared
host-minus-robot offset from a warmup and rolling minimum of odometry receipt
delays, and reliably publishes `/go2/odom`, `/go2/lidar/cloud_base`, and
`/go2/lidar/cloud_deskewed`. Reliable output remains compatible with downstream
best-effort sensor-data subscriptions while satisfying reliable recorders and
scan-conversion nodes.

The same boundary converts both clouds, the odometry pose and twist, and both
6x6 covariances from the mounted native basis into REP-103. The status message
reports `coordinate_transform=native_mount_to_rep103_rx_pi`.

No message is emitted during warmup. After startup, a zero, repeated,
regressing, stale, future, non-monotonic, unexpected-frame, or physically
implausible odometry sample permanently latches this process. No normalized
sensor output resumes; stop and restart the complete stack so no map, TF,
semantic result, or active goal can span epochs. Source timestamps are never
replaced with receipt time: the same shared offset is added to every accepted
stream, preserving their relative timing.

`/go2/time_sync/status` publishes a JSON `std_msgs/String`. Its `state` is
`warming` until enough odometry samples have arrived, `locked` while output is
enabled, and `fault_latched` after a terminal boundary fault; it also reports
the process identity, coordinate transform, offset, epoch/reset details, last
source/output stamps, and per-stream receive/publish/drop counters.

## Arm, command, and stop

The robot must already be stable, the time-sync state must be `locked`, and
normalized `/go2/odom` must be fresh.

```bash
ros2 service call /go2/motion/enable std_srvs/srv/SetBool "{data: true}"
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
ros2 service call /go2/motion/stop std_srvs/srv/Trigger "{}"
```

Commands received while disarmed are discarded. A command stream must remain
faster than `command_timeout_sec`. Forward motion also requires a fresh safety
cloud when fail-closed behavior is enabled. Reverse and in-place rotation stay
available to escape a front obstacle, but any residual forward slew is stopped
before reverse motion begins.

## Offline checks

The safety/calibration helpers do not import ROS or the Unitree SDK:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test -v
```
