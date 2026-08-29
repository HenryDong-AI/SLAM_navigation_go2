# Go2 depth-camera mapping backend

The source folder is named `go2_mapping_depthCam` as requested. Its ROS 2
package name is lowercase, `go2_mapping_depthcam`, to follow ROS naming rules.

This backend opens the Intel RealSense D435i directly with `pyrealsense2`; it
does not require `realsense2_camera`. It publishes registered RGB and metric
depth ROS messages, reconstructs aligned XYZRGB points, transforms XYZ through
the Go2 odometry pose, and fuses both geometry and color in each world-frame
voxel. It produces the same public map interfaces as the LiDAR mapper:

- `/go2/map/cloud` -- fused voxel `PointCloud2` with `x`, `y`, `z`, and packed
  `rgb` fields; RViz's **3D Map** display shows the measured camera colors
- `/map` -- 2D occupancy projection used by Nav2
- `/go2/mapping/status` -- mapper status JSON
- `/go2/map/save` and `/go2/map/reset` -- map services

Atomic map snapshots are written below
`/home/unitree/SLAM_nav/maps_depth_camera`. Each snapshot contains a colored
`map.ply`, the Nav2 `map.pgm` and `map.yaml`, and a resumable `state.npz` whose
`voxel_colors` and `voxel_color_counts` arrays preserve RGB fusion state.

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
