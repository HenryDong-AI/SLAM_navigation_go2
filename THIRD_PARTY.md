# Reference and dependency notes

This project is a clean-room integration that uses ROS topic/API behavior from
the requested references; it does not vendor their source code.

- `amberhandal/Go2-Inspector`, inspected at commit
  `281cccaecf8aeffa5476cd7b70c34f8ad984f8f4`, informed the RTAB-Map/Nav2 topic
  topology, Go2 footprint, and the real-robot L1 mounting caution. Its semantic
  path assumes an added RealSense and external SAM service, so that path is not
  copied here.
- `jizhang-cmu/autonomy_stack_go2`, inspected at commit
  `43d5f54b389b251713f0097893c30fa76c870d54`, informed the verified Foxy/L1
  topics, terrain-navigation constraints, and onboard camera stream pattern.
  Its repository has mixed or unclear licensing, so it is not automatically
  downloaded or linked.
- Unitree SDK2 Python is an external runtime dependency already installed on
  this robot. The camera adapter follows the public `VideoClient` sample API
  at
  <https://github.com/unitreerobotics/unitree_sdk2/blob/9754cd153af3da471b0fe5f3aa535e426fb11db3/example/go2/go2_video_client.cpp>
  without copying sample implementation. Unitree's official L1 SDK coordinate
  definition at <https://github.com/unitreerobotics/unilidar_sdk> establishes a
  right-handed sensor frame; the project combines that definition with a
  stationary live floor-sign check on this mounted Go2. Review Unitree's SDK
  licenses before redistribution.
- Ultralytics models/runtimes are optional external dependencies. Model files
  are intentionally excluded; review their software and model licenses for
  the intended deployment.

The source files authored in this project are MIT licensed. External ROS,
Unitree, model, and driver components retain their own licenses.
