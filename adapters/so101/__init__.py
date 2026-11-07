"""SO-101 adapter package.

Maps the existing SO-101 hardware and algorithm code to the generic interfaces
in interfaces/.  Everything in this directory is SO-101 specific; no other
adapter package depends on it.

Adapter registration (in a serve config YAML)::

    robot:
      bus:
        adapter_class: adapters.so101.bus.FeetechBusAdapter
        port: /dev/ttyUSB0
        baudrate: 1000000
        motor_ids: [1, 2, 3, 4, 5, 6]
        write_retries: 4

      controller:
        adapter_class: adapters.so101.controller.SO101RobotController
        commands_per_chunk: 5
        command_period_s: 0.04

      kinematics:
        adapter_class: adapters.so101.kinematics.SO101Kinematics
        mujoco_model_path: null   # resolved from package at runtime

    cameras:
      - role: column
        adapter_class: adapters.so101.camera.OpenCVCameraAdapter
        device_alias: /dev/v4l/by-id/<column-camera-id>
        width: 640
        height: 480
        fourcc: YUYV
      - role: wrist
        adapter_class: adapters.so101.camera.OpenCVCameraAdapter
        device_alias: /dev/v4l/by-id/<wrist-camera-id>
        width: 640
        height: 480
        fourcc: MJPG

    perception:
      projection:
        adapter_class: adapters.so101.projection.SO101LinearProjection
        calibration_path: null   # set in workspace config
      gripper_grounding:
        adapter_class: adapters.so101.grounding.FCPGGripperGrounding
        checkpoint_path: null    # set in serve config
      target_tracking:
        adapter_class: adapters.so101.tracking.TemplateTargetTracker
        profile_root: null       # set in workspace config

    planning:
      motion_planner:
        adapter_class: adapters.so101.planner.FCPGLeWMPlanner
        dynamics_checkpoint: null
        bridge_checkpoint: null
      safety_checker:
        adapter_class: adapters.so101.safety.SO101SafetyChecker
      phase_tracker:
        adapter_class: adapters.so101.phases.ManipulationPhaseTracker
"""
