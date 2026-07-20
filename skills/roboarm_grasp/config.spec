# roboarm_grasp skill instance config (passed to @skill.on_init as JSON dict)

optional:
  package_root:
    type: string
    description: Skill package root; defaults to RBNX_PACKAGE_ROOT env.
  config_yaml:
    type: string
    description: Path to config.yaml; default ${package_root}/config/config.yaml
  assets_dir:
    type: string
    description: Static assets root; default ${package_root}/assets
  arm_provider_id:
    type: string
    default: roboarm_arm
  camera_provider_id:
    type: string
    default: orbbec_camera_roboarm
  joint_names:
    type: array
    description: Must match roboarm_arm primitive joint_names
  gripper_open_width_m:
    type: number
    default: 0.080
  motion_steps:
    type: number
    default: 20

required:
  arm_offset:
    type: array
    description: Five joint offsets in degrees, matching roboarm_arm primitive

example:
  arm_offset: [-5.58, -21.45, -36.84, -73.54, 2.42]
  joint_names: [joint1, joint2, joint3, joint4, joint5, gripper]
  gripper_open_width_m: 0.080
  motion_steps: 20
