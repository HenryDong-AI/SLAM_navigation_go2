# go2_bringup

This package composes the project without hiding its safety gates. The mapping
launch starts the camera, disarmed motion bridge, selected geometric mapper,
semantic mapper, Nav2, and one RViz process. The localization launch uses a
saved 2D map with AMCL and does not start a second mapper.

```bash
ros2 launch go2_bringup full_stack.launch.py
ros2 launch go2_bringup full_stack.launch.py mapping_backend:=depth_camera
ros2 launch go2_bringup localization_stack.launch.py map:=/absolute/map.yaml
```

`mapping_backend` is either `lidar` (default) or `depth_camera`; the launch file
starts only one mapper and only one `odom -> base_link` TF authority.

Motion remains disarmed until `/go2/motion/enable` is called explicitly. Set
`start_motion_bridge:=false` for perception-only operation.
