# How to run SLAM_nav on the Unitree Go2 EDU

The mapping launch and motion commands must run in separate terminals. The
launch terminal must remain running for the motion service, mapping, and Nav2
to stay available.

## Safety first

- Put the robot on a flat, clear floor.
- Keep the Unitree controller and physical E-stop ready.
- Manually stand the robot with the Unitree controller or app. This project
  deliberately provides no automatic stand-up command.
- Never leave a pending `enable_motion` command running while restarting the
  stack. Cancel it with `Ctrl+C` first.
- The motion bridge starts disarmed every time.

## 1. One-time build

```bash
cd /home/unitree/SLAM_nav
conda activate slam_nav
./scripts/build.sh
```

## 2. Start the stack in Terminal 1

```bash
cd /home/unitree/SLAM_nav
conda activate slam_nav
./scripts/start_mapping.sh
```

Leave this terminal running. Messages about missing semantic calibration do
not stop SLAM or Nav2.

For a navigation-only test without the camera or semantic mapper:

```bash
./scripts/start_mapping.sh start_semantics:=false start_camera:=false
```

When running through SSH without a graphical display, add `use_rviz:=false`.

## 3. Verify the live stack in Terminal 2

```bash
cd /home/unitree/SLAM_nav
conda activate slam_nav
python3 scripts/doctor.py
ros2 service list | grep /go2/motion/enable
ros2 lifecycle get /controller_server
ros2 lifecycle get /planner_server
ros2 lifecycle get /bt_navigator
```

The Nav2 lifecycle nodes should report `active [3]`, and the motion-enable
service must be listed before continuing.

## 4. Arm motion in Terminal 2

After clearing the area and taking control of the physical E-stop:

```bash
./scripts/enable_motion.sh --i-understand
```

Continue only if the response says:

```text
success=True
motion enabled; waiting for fresh post-start sensors and a new cmd_vel
```

Terminal 1 should then report that fresh post-start time, odometry, and LiDAR
were received and the motion command gate is ready.

On stack startup, Terminal 1 should also report both periodic safety deadlines:

```text
time-sync status deadline active at 2.00 Hz
motion safety control deadline active at 20.00 Hz
```

If either deadline line is missing, do not send a navigation goal. The Foxy
executor is not running the corresponding safety task; stop and restart the
complete stack after rebuilding `go2_robot_bridge`.

If it says `waiting for service to become available`, press `Ctrl+C`. Terminal
1 is not running, or its `motion_bridge` exited. Restart the full stack first,
then run the arming command again. Do not leave the pending arming request open
during a restart because it could arm as soon as the service reappears.

Arming alone does not make the robot walk. Nav2 must receive a goal.

## 5. Send a navigation goal

Use RViz **Nav2 Goal** / **2D Goal Pose**, or run a validated named route:

```bash
ros2 run go2_navigation go2_route \
  --file src/go2_navigation/config/routes.yaml \
  validate

ros2 run go2_navigation go2_route \
  --file src/go2_navigation/config/routes.yaml \
  run inspection
```

Confirm that `/cmd_vel` is active while a goal is executing:

```bash
ros2 topic hz /cmd_vel
```

## 6. Stop safely

Stop and disarm before saving, calibration, or shutting down:

```bash
./scripts/stop_motion.sh
```

Then cancel the launch in Terminal 1 with `Ctrl+C`.

## 7. Save maps

Keep Terminal 1 running, stop/disarm the robot, then run:

```bash
./scripts/save_maps.sh
```

The geometric and semantic map bundles are written under `maps/` and
`semantic_maps/`.

## Semantic calibration messages

These messages do not block LiDAR SLAM or route navigation:

- `camera_info.yaml is an uncalibrated placeholder`
- `CALIBRATION REQUIRED: semantic projection is disabled`
- occasional `GetImageSample returned SDK code 3104`
- occasional `Corrupt JPEG data`

Until the exact camera intrinsics and camera-to-LiDAR transform are measured,
the semantic mapper publishes unknown geometry but does not attach YOLO labels.
Follow [docs/CALIBRATION.md](docs/CALIBRATION.md). Never set
`calibration_confirmed: true` while the zero intrinsics and identity transform
are still present.

## Quick troubleshooting

Check that the stack is still running:

```bash
pgrep -af 'start_mapping|full_stack.launch|motion_bridge'
```

Check required sensor and motion interfaces:

```bash
ros2 topic hz /go2/odom
ros2 topic hz /go2/lidar/cloud_base
ros2 service list | grep /go2/motion
```

If a safety fault is reported as latched, stop motion and restart the complete
stack. Do not restart only `motion_bridge` or `sensor_time_bridge`.
