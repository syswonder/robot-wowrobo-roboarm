from __future__ import annotations

import base64
import json
import threading
from collections import Counter
from concurrent import futures
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import TypeAdapter

from roboarm_core.config import get_config_value
from roboarm_core.llm.dataclass import DetectedBox, DetectedFromLLM
from roboarm_core.llm.llm_api import LLMAPI, extract_json_from_markdown

BATCH_SIZE = 1


class LLMDetect:
    def __init__(self) -> None:
        self.llm_api = LLMAPI()

    def detect_frame(
        self,
        frame: cv2.typing.MatLike,
        prompt_key: str,
        replace_map: dict[str, str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> futures.Future[ChatCompletion] | None:
        if get_config_value("RotationCam2Arm", raise_if_missing=False):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        _, img_encoded = cv2.imencode(".jpg", frame)
        image_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
        if BATCH_SIZE == 1:
            return self.llm_api.chat_img_async(
                image_base64=image_base64,
                prompt_key=prompt_key,
                replace_map=replace_map,
                schema=schema,
            )
        sub_tasks: list[futures.Future[ChatCompletion]] = []
        for _ in range(BATCH_SIZE):
            task = self.llm_api.chat_img_async(
                image_base64=image_base64,
                prompt_key=prompt_key,
                replace_map=replace_map,
                schema=schema,
            )
            if task is not None:
                sub_tasks.append(task)
        if not sub_tasks:
            return None
        aggregated_future: futures.Future[Any] = futures.Future()

        def _wait_and_aggregate() -> None:
            try:
                json_strs: list[str | None] = []
                for task in sub_tasks:
                    try:
                        completion = task.result()
                        if completion.choices:
                            json_strs.append(completion.choices[0].message.content)
                        else:
                            json_strs.append(None)
                    except Exception as exc:
                        print(f"批量请求中某次失败: {exc}")
                        json_strs.append(None)
                aggregated = _aggregate_box_json(json_strs)
                msg_mock = SimpleNamespace(content=aggregated)
                choice_mock = SimpleNamespace(message=msg_mock)
                mock = SimpleNamespace(
                    choices=[choice_mock] if aggregated is not None else []
                )
                aggregated_future.set_result(mock)
            except Exception as exc:
                aggregated_future.set_exception(exc)

        threading.Thread(target=_wait_and_aggregate, daemon=True).start()
        return aggregated_future


def _filter_outlier_boxes(boxes: list[DetectedFromLLM]) -> list[DetectedFromLLM]:
    if len(boxes) <= 2:
        return boxes
    cxs = np.array([b.box_center_x for b in boxes])
    cys = np.array([b.box_center_y for b in boxes])
    cx_med, cy_med = float(np.median(cxs)), float(np.median(cys))
    dists = np.sqrt((cxs - cx_med) ** 2 + (cys - cy_med) ** 2)
    threshold = max(float(np.median(dists)) * 2, 0.05)
    keep = [b for b, d in zip(boxes, dists, strict=True) if d <= threshold]
    return keep if keep else boxes


def _aggregate_box_json(json_strs: list[str | None]) -> str | None:
    valid: list[DetectedFromLLM] = []
    for text in json_strs:
        if text is None:
            continue
        try:
            box = TypeAdapter(DetectedFromLLM).validate_json(
                extract_json_from_markdown(text)
            )
            if box.is_valid():
                valid.append(box)
        except Exception as exc:
            print(f"解析/校验 JSON 失败: {exc}")
    if not valid:
        return None
    filtered = _filter_outlier_boxes(valid)
    class_name = Counter(b.class_name for b in filtered).most_common(1)[0][0]
    return json.dumps(
        {
            "id": filtered[0].id,
            "class_name": class_name,
            "box_center_x": float(np.mean([b.box_center_x for b in filtered])),
            "box_center_y": float(np.mean([b.box_center_y for b in filtered])),
            "box_width": float(np.mean([b.box_width for b in filtered])),
            "box_height": float(np.mean([b.box_height for b in filtered])),
        }
    )


def draw_boxes_on_frame(
    boxes: list[DetectedBox],
    frame: cv2.typing.MatLike,
) -> cv2.typing.MatLike:
    annotated_frame = frame.copy()
    for box in boxes:
        box_points = cv2.boxPoints(
            (
                (float(box.box_center_x), float(box.box_center_y)),
                (float(box.box_width), float(box.box_height)),
                float(box.box_rotation_deg),
            )
        )
        box_points = np.int64(box_points)
        cv2.drawContours(annotated_frame, [box_points], 0, (0, 0, 255), 2)
        x1, y1 = box_points.min(axis=0)
        cv2.putText(
            annotated_frame,
            box.class_name,
            (int(x1), max(int(y1) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return annotated_frame


def json2box(json_str: str, img_w: int, img_h: int) -> DetectedBox | None:
    try:
        box: DetectedFromLLM = TypeAdapter(DetectedFromLLM).validate_json(
            extract_json_from_markdown(json_str)
        )
    except Exception as exc:
        print(f"解析/校验 JSON 失败: {exc}")
        return None
    return box.to_detected_box(img_w, img_h) if box.is_valid() else None
