from roboarm_core.arm.arm_base import Arm, StepCallback

__all__ = ["Arm", "StepCallback", "RobonixArm"]


def __getattr__(name: str):
    if name == "RobonixArm":
        from roboarm_core.arm.robonix_arm import RobonixArm

        return RobonixArm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
