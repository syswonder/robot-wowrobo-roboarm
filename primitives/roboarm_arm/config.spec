# roboarm_arm instance config (passed to on_init(cfg) via robonix_manifest.yaml)

required:
  arm_offset:
    type: array
    description: Five joint angle offsets in degrees, matching arm/calibrate_offset.py output.

optional:
  package_root:
    type: string
    description: Primitive package root; defaults to RBNX_PACKAGE_ROOT env.
  urdf_path:
    type: string
    default: <package_root>/src/assets/urdf/lerobo/low_cost_robot.urdf
  lerobot_src:
    type: string
    default: <package_root>/src/vendor/lerobot/src
  serial_port:
    type: string
    default: COM3
    description: Serial port for the Lerobo Koch follower (e.g. COM3 or /dev/ttyUSB0).
  arm_backend:
    type: string
    default: real
    description: "real for hardware, sim for Isaac Sim HTTP service."
  arm_sim_host:
    type: string
    default: 127.0.0.1
  arm_sim_port:
    type: number
    default: 8770
  arm_sim_timeout_s:
    type: number
    default: 10.0
  calibration_dir:
    type: string
    default: <package_root>/src/assets/calibration
  robot_id:
    type: string
    default: koch_follower
  publish_rate_hz:
    type: number
    default: 20
  end_pose_frame:
    type: string
    default: base_link
  joint_names:
    type: array
    default: [joint1, joint2, joint3, joint4, joint5, gripper]
    description: JointState names; five arm joints plus gripper.
  joint_states_topic:
    type: string
    default: /roboarm/joint_states
  end_pose_topic:
    type: string
    default: /roboarm/end_pose
  joint_command_topic:
    type: string
    default: /roboarm/joint_command
  gripper_open_width_m:
    type: number
    default: 0.080
  motion_steps:
    type: number
    default: 20
    description: Interpolation steps for move_to_home and smooth motions.
  move_to_home_on_activate:
    type: boolean
    default: true
  allow_missing_hardware:
    type: boolean
    default: false
    description: If true, activate succeeds without arm hardware (degraded mode).

example:
  serial_port: /dev/ttyUSB0
  arm_backend: real
  arm_offset: [-5.58, -21.45, -36.84, -73.54, 2.42]
  robot_id: koch_follower
  publish_rate_hz: 20
  joint_names: [joint1, joint2, joint3, joint4, joint5, gripper]
  gripper_open_width_m: 0.080
  move_to_home_on_activate: true
