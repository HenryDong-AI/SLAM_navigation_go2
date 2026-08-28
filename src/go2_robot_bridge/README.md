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
  `SportClient.Move`. It starts **DISARMED** and stops on stale commands, stale
  odometry, a forward obstacle, malformed/stale safety cloud data, SDK failure,
  an explicit stop request, or shutdown. Because
  Unitree's Python CycloneDDS binding and `rclpy` cannot safely initialize DDS
  domains in one process on this Go2, the ROS node never imports that SDK. It
  creates a private, non-ROS worker only after an explicit successful enable
  request. That worker detects `mcf` on current firmware or `sport_mode` on
  legacy firmware, ensures the detected controller is running, and verifies
  `mcf` selection and robot standing height before accepting velocity commands.
  On current MCF firmware the obstacle-avoidance API can acknowledge velocity
  while suppressing locomotion, so the armed worker follows the proven direct
  SportClient path: it temporarily disables Unitree avoidance while the
  project's fail-closed front-LiDAR gate is active, then restores Unitree
  avoidance when the worker is stopped.

This package does not copy implementation code from Unitree demos or the
reference projects. It uses their public runtime interfaces only and is
licensed under MIT.

## Critical hardware safety

Keep the physical remote/E-stop in an operator's hand whenever this node can
reach the robot. Commission with the Go2 supported, in a clear stair-free
area, at low speed. Software guards are not a substitute for the physical
stop. The bridge deliberately does **not** call `StandUp`, `RecoveryStand`, or
`BalanceStand` merely because the stack launches. The explicit enable operation
may run `Damp` -> `RecoveryStand` -> `BalanceStand` if fresh sport-state
telemetry reports a body height below `standing_min_body_height`. Keep the robot
supported during initial commissioning and expect posture motion when enabling.

Calling `/go2/motion/stop` is latching and disarms motion. Re-enabling always
clears the cached command and requires a new `/cmd_vel` message.

The motion worker receives bounded, length-framed JSON over a private inherited
socket. Its dispatcher accepts only `Move(x, y, yaw)` and `StopMove()`; there is
no remote path to standing, posture, or arbitrary mode APIs. The required
firmware controller selection and posture preparation happen internally only
during the explicit enable operation. Every value is checked for finiteness,
and arming fails if fresh sport state or standing-height verification is
unavailable. The semantic stop operation uses `Move(0, 0, 0)` on current `mcf`
firmware, where the SDK's `StopMove()` returns `-1`, and retains `StopMove()`
on legacy `sport_mode`. RPCs, startup, and process reaping are bounded by
timeout, and the worker makes a final best-effort controller-appropriate stop
on parent EOF. An unconfirmed
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
The current firmware's built-in streams are already REP-103: X forward, Y left,
and Z up. The time boundary preserves those coordinates, so floor returns stay
below the robot and are excluded by the default obstacle band.

### Native sensor clock boundary

`robot_bridge.launch.py` always starts `sensor_time_bridge`. It requests the
firmware's best-effort native streams `/utlidar/robot_odom`,
`/utlidar/cloud_base`, and `/utlidar/cloud_deskewed`, estimates one shared
host-minus-robot offset from a warmup and rolling minimum of odometry receipt
delays, and reliably publishes `/go2/odom`, `/go2/lidar/cloud_base`, and
`/go2/lidar/cloud_deskewed`. Reliable output remains compatible with downstream
best-effort sensor-data subscriptions while satisfying reliable recorders and
scan-conversion nodes.

The same boundary preserves both clouds, the odometry pose and twist, and both
6x6 covariances in the firmware's verified REP-103 basis. The status message
reports `coordinate_transform=identity_rep103`. Set
`apply_native_axis_conversion:=true` only for legacy firmware whose measured
floor is positive Z and whose standing odometry height is negative Z.

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
normalized `/go2/odom` must be fresh. Enabling can take several seconds if the
firmware's `mcf` controller is not already active. Wait for the response and
the `motion command gate is ready` log before publishing a new command.

```bash
ros2 service call /go2/motion/enable std_srvs/srv/SetBool "{data: true}"
ros2 topic pub --rate 10 --times 20 --print 10 \
  /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
ros2 service call /go2/motion/stop std_srvs/srv/Trigger "{}"
```

Use `--times` instead of wrapping `ros2 topic pub` in a short shell timeout:
Foxy's DDS discovery can take more than two seconds before the first publish.

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
