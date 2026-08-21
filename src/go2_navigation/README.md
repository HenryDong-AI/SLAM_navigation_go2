# go2_navigation

ROS 2 Foxy Nav2 bringup for the Unitree Go2 EDU. `navigation.launch.py` uses the
live `/map` generated in the `odom` frame. `localization.launch.py` loads a
saved occupancy map, converts `/go2/lidar/cloud_base` to `/scan`, and uses AMCL
to establish `map -> odom` before navigating.
Standalone `localization.launch.py` starts a sensor-only time bridge by default.
The top-level `go2_bringup` launch passes `start_sensor_bridge:=false` because
it already owns that single boundary.

The `/go2/lidar` and `/go2/odom` inputs are normalized together by
`sensor_time_bridge`. Raw `/utlidar` messages use the robot clock and must
not be connected directly to host-clock Nav2 or TF consumers.

The command path is deliberately conservative (0.35 m/s maximum). Nav2 only
publishes `/cmd_vel`; the separate robot bridge remains disarmed until an
operator explicitly enables motion. Keep the physical controller/E-stop in
hand and commission in a clear, flat area.

Routes are validated YAML. Record the current pose with:

```bash
ros2 run go2_navigation go2_route --file routes.yaml record inspection --name entrance
```

Then execute it after Nav2 is active and motion is explicitly enabled:

```bash
ros2 run go2_navigation go2_route --file routes.yaml run inspection
```

Every non-success exit (including timeout, rejection, exception, and Ctrl-C)
calls the latching `/go2/motion/stop` service, requests cancellation of the
tracked Nav2 goal, and waits for a terminal action result. If that cannot be
confirmed, the runner sends a protocol-level NavigateToPose cancel-all request.
An aborted route therefore requires an explicit motion re-enable. If the CLI
reports that terminal status could not be confirmed, inspect or restart the
Nav2 lifecycle before re-enabling.

Routes in `odom` are valid for the current mapping session. For repeat routes
after restart, save the map, start `localization.launch.py`, initialize AMCL,
and record/use routes whose `frame_id` is `map`.
