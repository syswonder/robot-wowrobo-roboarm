from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from roboarm_core.config import get_config_value


def normalize_box_rotation_deg(angle_deg: float) -> float:
    angle = float(angle_deg) % 360.0
    if angle > 180.0:
        angle -= 360.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


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
            box_rotation_deg=0.0,
            confidence=confidence,
        )


class DetectedItem(BaseModel):
    """单个检测目标（用于 LLM JSON schema）。"""

    id: int
    class_name: str
    box_center_x: float = Field(ge=0.0, le=1.0)
    box_center_y: float = Field(ge=0.0, le=1.0)
    box_width: float = Field(ge=0.0, le=1.0)
    box_height: float = Field(ge=0.0, le=1.0)

    def to_detected_from_llm(self) -> "DetectedFromLLM":
        return DetectedFromLLM(
            id=self.id,
            class_name=self.class_name,
            box_center_x=self.box_center_x,
            box_center_y=self.box_center_y,
            box_width=self.box_width,
            box_height=self.box_height,
        )


class InstructionDetectResponse(BaseModel):
    """user_instruction_prompt 的结构化输出。"""

    thinking_process: str = ""
    failed: bool = False
    objects: list[DetectedItem] = Field(default_factory=list)


@dataclass
class DetectedBox:
    class_name: str
    box_center_x: int | float
    box_center_y: int | float
    box_width: int | float
    box_height: int | float
    box_rotation_deg: float = 0
    confidence: Optional[float] = None
