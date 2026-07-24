from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import time
from concurrent import futures
from typing import Any

import toml
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from PIL import ImageFont
from pydantic import TypeAdapter

from roboarm_core.config import get_config_dir, get_config_value
from roboarm_core.llm.dataclass import DetectedFromLLM


def _load_cjk_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "--format=%{file}\n"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        for line in result.stdout.splitlines():
            path = line.strip()
            if path and os.path.exists(path):
                return ImageFont.truetype(path, size)
    except Exception:
        pass
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


font = _load_cjk_font(16)


class LLMAPI:
    def __init__(self) -> None:
        https_proxy = get_config_value("https_proxy", None, raise_if_missing=False)
        if https_proxy:
            os.environ["http_proxy"] = https_proxy
            os.environ["https_proxy"] = https_proxy

        prompts_file = get_config_value("prompts_file")
        prompts_path = get_config_dir() / prompts_file
        self.prompts = toml.load(prompts_path)["prompts"]
        self.base_url = get_config_value("llm_base_url")
        api_key = get_config_value("llm_api_key")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=api_key)

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._start_times: dict[futures.Future, float] = {}

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def chat_img_async(
        self,
        image_base64: str,
        prompt_key: str,
        replace_map: dict[str, str] | None = None,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> futures.Future[ChatCompletion] | None:
        prompt = self.prompts.get(prompt_key)
        if prompt is None:
            return None
        for key, value in (replace_map or {}).items():
            prompt = prompt.replace(key, value)
        completion_coroutine = self.async_client.chat.completions.create(
            model=model or get_config_value("llm_model"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=temperature,
            response_format=(
                {"type": "json_object"}
                if schema is None
                else {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "detection_output",
                        "schema": inline_schema_refs(schema),
                    },
                }
            ),
        )
        future = asyncio.run_coroutine_threadsafe(completion_coroutine, self._loop)
        self._start_times[future] = time.time()
        return future

    def await_task(
        self,
        task: futures.Future[ChatCompletion],
        blocking: bool = False,
    ) -> tuple[str | None, bool]:
        if blocking:
            try:
                completion = task.result()
            except Exception as exc:
                print(f"Error in task: {exc}")
                return None, True
        elif not task.done():
            return None, False
        else:
            try:
                completion = task.result()
            except Exception as exc:
                print(f"Error in task: {exc}")
                return None, True
            finally:
                self._start_times.pop(task, None)
        if completion.choices is None or len(completion.choices) == 0:
            return None, True
        return extract_completion_text(completion.choices[0].message), True


def extract_completion_text(message: Any) -> str | None:
    """从 LLM 回复中提取可用的检测 JSON 文本。

    部分推理模型（如 Qwen thinking）会把完整结构化结果放在 reasoning_content，
    而 content 里可能只有 thinking_process 叙述。
    """
    candidates: list[str] = []
    for attr in ("content", "reasoning_content"):
        value = getattr(message, attr, None)
        if value and str(value).strip():
            candidates.append(str(value).strip())

    for candidate in candidates:
        if _has_detection_payload(candidate):
            return candidate
    return candidates[0] if candidates else None


def _has_detection_payload(raw: str) -> bool:
    try:
        parsed = json.loads(extract_json_from_markdown(raw))
    except json.JSONDecodeError:
        return False
    if isinstance(parsed, list):
        return bool(parsed)
    if isinstance(parsed, dict):
        if parsed.get("failed") is True:
            return True
        if parsed.get("objects"):
            return True
        if "box_center_x" in parsed:
            return True
    return False


def inline_schema_refs(schema: dict) -> dict:
    defs = schema.get("$defs", {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve(defs[ref_name])
            return {k: resolve(v) for k, v in obj.items() if k != "$defs"}
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)


def extract_json_from_markdown(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S)
    if match:
        return match.group(1).strip()
    text = re.sub(r"```+\s*$", "", text).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :].strip()
        text = re.sub(r"```+\s*$", "", text).strip()
    return text
