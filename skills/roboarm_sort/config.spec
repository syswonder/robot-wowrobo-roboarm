# roboarm_sort skill instance config (passed to @skill.on_init as JSON dict)

optional:
  package_root:
    type: string
    description: Skill package root; defaults to RBNX_PACKAGE_ROOT env.
  config_yaml:
    type: string
    description: Path to sort config.yaml; default ${package_root}/config/config.yaml
  grasp_config_yaml:
    type: string
    description: Path to roboarm_grasp config.yaml; default ../roboarm_grasp/config/config.yaml
  grasp_assets_dir:
    type: string
    description: roboarm_grasp assets directory; default sibling skill assets/
  arm_provider_id:
    type: string
    default: roboarm_arm
  camera_provider_id:
    type: string
    default: orbbec_camera_roboarm
  joint_names:
    type: array
    default: [joint1, joint2, joint3, joint4, joint5, gripper]
  gripper_open_width_m:
    type: number
    default: 0.080
  motion_steps:
    type: integer
    default: 20

required:
  arm_offset:
    type: array
    description: Joint home offsets (same as roboarm_grasp).

example:
  arm_offset:
    - -2.51
    - -20.88
    - -22.77
    - -79.47
    - -1.80
  grasp_config_yaml: ../roboarm_grasp/config/config.yaml
