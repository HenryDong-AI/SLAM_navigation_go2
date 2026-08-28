# Commissioning safety

- Test on a flat, clear floor with a second person holding the physical Go2
  controller and able to stop motion immediately.
- The software bridge starts disarmed and never changes posture merely because
  the stack launched. The explicitly acknowledged enable operation reads
  `rt/sportmodestate`; if body height is below the configured standing minimum,
  it may execute `Damp` -> `RecoveryStand` -> `BalanceStand` before arming.
- Arming starts the isolated SportClient while holding a controller-appropriate
  zero-velocity stop. Because startup and possible posture recovery block ROS
  callbacks briefly, the bridge requires new clock, odometry, and LiDAR
  callbacks afterward before it accepts a new velocity command.
- Confirm raw `/utlidar` inputs and normalized `/go2/lidar` plus `/go2/odom`
  are current, and that `/go2/time_sync/status` reports `locked`, before
  enabling motion. Any time/frame/pose discontinuity latches the boundary and
  downstream consumers; stop and restart the complete stack. Do not restart
  only the bridge or merge maps across epochs.
- Start at the configured 0.35 m/s cap or lower. The L1 can miss low obstacles;
  do not rely on it for cables, glass, drop-offs, or objects below roughly
  0.3 m.
- Current MCF firmware can acknowledge obstacle-avoidance API velocity without
  moving. While the explicitly armed worker exists, it temporarily disables
  Unitree obstacle avoidance and uses direct SportClient velocity behind this
  project's fail-closed front-LiDAR gate. Disarming restores Unitree avoidance.
  Keep the physical E-stop available; software safeguards are not an E-stop.
- Stop motion before saving, calibration, inspecting a fault, changing network
  configuration, or restarting DDS.
- A loss of normalized time status, odometry, or healthy LiDAR after arming is
  restart-required. An empty/all-zero/NaN cloud does not count as fresh.
- Never run two command bridges or teleoperation programs at the same time.
- Measure the standing footprint on the exact robot and update Nav2's
  conservative commissioning footprint before route operation.
