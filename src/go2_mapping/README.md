# go2_mapping

`go2_mapping` is a clean-room, MIT-licensed ROS 2 Foxy package for bounded map
accumulation from a Unitree Go2 EDU LiDAR localization stream. It maintains a
voxelized 3D point map and projects obstacle-height returns into a sparse 2D
log-odds occupancy grid with ray clearing.

The default configuration consumes the host-clock-normalized copies of the Go2
firmware LIO outputs: `/go2/lidar/cloud_deskewed` and `/go2/odom`. Start
`sensor_time_bridge` first (the top-level bringup does this). The cloud
**must already be registered in `world_frame`**; this node deliberately does
not guess or apply a missing transform. It is a bounded mapping accumulator,
not a scan-matching frontend. Loop closure, drift correction, and saved-map
relocalization require an optional SLAM/localization backend such as Point-LIO
plus a compatible loop-closure or localization component.

## Build and run

```bash
cd ~/SLAM_nav
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select go2_mapping
source install/setup.bash
ros2 launch go2_mapping mapping.launch.py
```

The launch file exposes the sensor topics and frame. For the topic convention
used by `autonomy_stack_go2`, the same package can consume Point-LIO output:

```bash
ros2 launch go2_mapping mapping.launch.py \
  cloud_topic:=/registered_scan \
  odom_topic:=/state_estimation \
  world_frame:=camera_init \
  require_time_sync_status:=false
```

Check the actual `header.frame_id` values first. Both cloud and odometry parent
frames must equal `world_frame`, otherwise messages are rejected and the reason
appears on `/go2/mapping/status`. Disabling the status guard is only for an
external backend that has independently verified one coherent host clock,
coordinate convention, and restart epoch; never disable it for `/go2` inputs.

## ROS interfaces

Inputs use best-effort, volatile, sensor-data-compatible QoS:

- `cloud_topic` (`sensor_msgs/msg/PointCloud2`), default
  `/go2/lidar/cloud_deskewed`
- `odom_topic` (`nav_msgs/msg/Odometry`), default `/go2/odom`

Reliable, transient-local map outputs are:

- `/go2/map/cloud` (`sensor_msgs/msg/PointCloud2`), voxel centroids
- `/map` (`nav_msgs/msg/OccupancyGrid`), 2D probability projection

`/go2/mapping/status` is a compact JSON `std_msgs/msg/String` with state,
counters, map sizes, age, cropping, and the last guard warning.

Services:

- `ros2 service call /go2/map/save std_srvs/srv/Trigger {}`
- `ros2 service call /go2/map/reset std_srvs/srv/Trigger {}`

Save creates one atomically published timestamped directory below `output_dir`.
It contains `map.ply`, `map.pgm`, `map.yaml`, and a pickle-free `state.npz` with
arrays plus JSON metadata. Set `load_state_path` to either that directory or its
`state.npz` before startup. Voxel size, grid resolution, and world frame must
match the current configuration. Saved trinary maps use gray 205 with strict
`free_thresh: 0.196`, so Foxy map_server reloads unobserved cells as unknown
rather than free.

While mapping, `autosave_interval_sec` (60 seconds by default) writes a normal
atomic timestamped snapshot after new clouds have been fused. A clean shutdown
attempts one final snapshot. Set the interval to zero to disable autosave;
manual `/go2/map/save` snapshots continue to work.

Saving is rejected until the sensor-time boundary is locked and fault-free. A
bridge process/epoch change, malformed status, or reported time fault clears
the accumulated geometric state and latches the mapper; restart the complete
stack before creating another map.

## TF ownership and localization mode

The mapper itself publishes no TF. `mapping.launch.py` starts exactly one
`odom_tf_bridge`, which preserves each normalized odometry timestamp and broadcasts the
configured `odom_frame` (the launch `world_frame`, `odom` by default) to
`base_link`. If another driver already owns that transform, launch with
`publish_odom_tf:=false`.

For saved-map localization without accumulating a new map, launch only the
bridge alongside the selected localization backend:

```bash
ros2 launch go2_mapping odom_tf.launch.py \
  odom_topic:=/state_estimation odom_frame:=camera_init \
  require_time_sync_status:=false
```

Only one node in the system should publish a given parent/child TF pair.

## Bounded behavior

Input count, finite coordinates, host-clock age, order, frames, odometry synchronization,
range, relative height, number of voxels, sparse grid cells, rays per cloud, ray
length, and dense output area all have explicit limits in `config/mapping.yaml`.
When the sparse global extent would allocate an excessive rectangular grid, the
published/saved 2D view is cropped around the latest robot cell while the sparse
state remains bounded by `max_grid_cells`. Set `retention_radius` above zero for
a rolling local 3D map; zero retains global voxels until oldest-entry eviction.
