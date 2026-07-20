# orbbec_camera_roboarm instance config (passed to on_init(cfg) via robonix_manifest.yaml)

optional:
  camera_ip:
    type: string
    default: ""
    description: Empty for local USB camera; set for remote HTTP MJPEG stream.
  camera_port:
    type: number
    default: 8083
  enable_color:
    type: boolean
    default: true
  enable_depth:
    type: boolean
    default: true
  publish_rate_hz:
    type: number
    default: 15
  intrinsics_rate_hz:
    type: number
    default: 1.0
  rgb_topic:
    type: string
    default: /roboarm/camera/rgb
  depth_topic:
    type: string
    default: /roboarm/camera/depth
  intrinsics_topic:
    type: string
    default: /roboarm/camera/camera_info
  extrinsics_topic:
    type: string
    default: /roboarm/camera/extrinsics
  optical_frame:
    type: string
    default: camera_color_optical_frame
  base_frame:
    type: string
    default: base_link
  frame_timeout_ms:
    type: number
    default: 100
  camera_width:
    type: number
    default: 640
  camera_height:
    type: number
    default: 480
  fx:
    type: number
    default: 600.0
  fy:
    type: number
    default: 600.0
  cx:
    type: number
    default: 320.0
  cy:
    type: number
    default: 240.0
  extrinsics_translation:
    type: array
    default: [0.0, 0.0, 0.0]
  extrinsics_rotation_xyzw:
    type: array
    default: [0.0, 0.0, 0.0, 1.0]
  remote_depth_as_rgb:
    type: boolean
    default: true
    description: Publish remote HTTP depth visualization as rgb8 on depth topic.
  allow_missing_hardware:
    type: boolean
    default: false
    description: If true, activate succeeds without a camera (degraded mode, no frames).

example:
  camera_ip: ""
  camera_port: 8083
  enable_color: true
  enable_depth: true
  publish_rate_hz: 15
