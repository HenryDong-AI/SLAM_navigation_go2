# Go2 depth-camera mapping backend

The source folder is named `go2_mapping_depthCam` as requested. Its ROS 2
package name is lowercase, `go2_mapping_depthcam`, to follow ROS naming rules.

This backend opens the Intel RealSense D435i directly with `pyrealsense2`; it
does not require `realsense2_camera`. The bridge aligns depth to RGB, applies
edge-preserving disparity-domain spatial filtering, and reconstructs one atomic
camera-frame XYZRGB `PointCloud2` on `/go2/depth_camera/points`. The message
uses the hardware capture time mapped into ROS time. The mapper interpolates the
Go2 SE(3) pose at that capture boundary instead of using a post-processing or
nearest-odometry timestamp.

It produces the same public map interfaces as the LiDAR mapper:

- `/go2/map/cloud` -- stable RGB surface voxels with `x`, `y`, `z`, and
  packed `rgb`; RViz's **3D Map** display shows measured camera colors
- `/map` -- 2D occupancy projection used by Nav2
- `/go2/mapping/status` -- mapper status JSON
- `/go2/map/save` and `/go2/map/reset` -- map services

Atomic map snapshots are written below
`/home/unitree/SLAM_nav/maps_depth_camera`. Each contains a colored
`map.ply`, the Nav2 `map.pgm` and `map.yaml`, and a resumable `state.npz`.
RGB, observation confidence, and surface variance survive save/load.

The camera captures 848x480 at 15 Hz and publishes the newest bounded cloud at
5 Hz, with pixel stride 2 and a 0.04 m final voxel size. Both ends use KEEP_LAST
depth 1, and the mapper consumes one latest-only cloud mailbox. Full-map output
and autosave run on a separate coalescing worker.

Moving keyframes use robust full-SE(3) point-to-plane registration against a
short RGB-D submap. A rejected moving scan is excluded from the permanent map;
after three consecutive failures only the short recovery target is reseeded.
Accepted frames contribute one equal-weight observation per voxel rather than
one weight per pixel, and a surface needs two consistent observations before it
is published or saved. This suppresses flying pixels and close-view density
bias without changing robot odometry, TF, Nav2, or motion commands.

Monitor `frames_registration_skipped`, `frames_keyframe_skipped`,
`odometry_pose_source`, `surface_voxel_count`, registration acceptance, and
the output-worker fields in `/go2/mapping/status`. `frames_superseded` is
intentional latest-frame load shedding. This local method does not provide
global loop closure or retroactively repair an old snapshot.

## Mount calibration

`config/depth_mapping.yaml` contains the transform measured for the D435i
mount on this Go2. If the bracket moves, start the complete stack, keep the
robot stationary in a geometrically varied scene, and run:

```bash
ros2 run go2_mapping_depthcam extrinsic_calibrator \
  --ros-args -p sample_count:=30
```

The sensor-time bridge publishes a bounded
`/go2/lidar/cloud_calibration` stream only while this utility listens, avoiding
a second full-size LiDAR DDS subscription. Only use a result marked
`credible: true`; copy the complete row-major matrix into
`base_from_camera_optical`, rebuild this package, and restart the stack.

## Run

From the workspace root after building:

```bash
./scripts/start_mapping.sh mapping_backend:=depth_camera use_depth_viewer:=true
```

To test only the D435i and open the side-by-side RGB/RGB-D viewer:

```bash
./scripts/start_depth_camera.sh
```

Press `q` or Esc in the viewer to close it. A graphical `DISPLAY` is required;
use X forwarding when running it through SSH.
