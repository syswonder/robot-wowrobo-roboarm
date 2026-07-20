from dataclasses import dataclass
from typing import Optional

from roboarm_core.config import get_config_value


@dataclass
class DetectedFromLLM:
    id: int
    class_name: str
    box_center_x: float
    box_center_y: float
    box_width: float
    box_height: float
    thinking_process: str | None = None
    failed: bool | None = None

    def is_valid(self) -> bool:
        return (
            not self.failed
            and 0.0 <= self.box_center_x <= 1.0
            and 0.0 <= self.box_center_y <= 1.0
            and 0.0 <= self.box_width <= 1.0
            and 0.0 <= self.box_height <= 1.0
        )

    def to_detected_box(
        self,
        img_w: int,
        img_h: int,
        confidence: Optional[float] = None,
        box_rotation_deg: float = 0.0,
    ) -> "DetectedBox":
        if not self.is_valid():
            raise ValueError("Invalid box parameters, cannot convert to DetectedBox.")
        cx = round(self.box_center_x * img_w)
        cy = round(self.box_center_y * img_h)
        if get_config_value("RotationCam2Arm", raise_if_missing=False):
            cx = img_w - cx
            cy = img_h - cy
        return DetectedBox(
            class_name=self.class_name,
            box_center_x=cx,
            box_center_y=cy,
            box_width=round(self.box_width * img_w),
            box_height=round(self.box_height * img_h),
            box_rotation_deg=box_rotation_deg,
            confidence=confidence,
        )


@dataclass
class DetectedBox:
    class_name: str
    box_center_x: int | float
    box_center_y: int | float
    box_width: int | float
    box_height: int | float
    box_rotation_deg: float = 0
    confidence: Optional[float] = None
