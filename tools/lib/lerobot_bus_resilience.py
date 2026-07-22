"""LeRobot 串口通信容错：不稳定总线上增加重试，标定循环中跳过瞬时失败。"""

from __future__ import annotations

import time
import types
from collections.abc import Callable
from functools import wraps
from pprint import pformat
from typing import Any

from lerobot.utils.utils import enter_pressed, move_cursor_up


def apply_unstable_serial_patches(
    bus: Any,
    *,
    num_retry: int = 10,
    retry_delay_s: float = 0.05,
) -> None:
    """为 bus 打补丁：通信失败时多次重试，标定读角循环不因单次断联退出。"""
    num_retry = max(0, int(num_retry))
    retry_delay_s = max(0.0, float(retry_delay_s))

    _patch_public_comm_methods(bus, default_retry=num_retry)
    bus._sync_read = types.MethodType(
        _make_resilient_sync_read(num_retry, retry_delay_s), bus
    )
    bus.record_ranges_of_motion = types.MethodType(
        _make_resilient_record_ranges(num_retry, retry_delay_s), bus
    )


def _effective_retry(requested: int, default: int) -> int:
    return requested if requested > 0 else default


def _patch_public_comm_methods(bus: Any, *, default_retry: int) -> None:
    for method_name in (
        "sync_read",
        "read",
        "write",
        "sync_write",
        "disable_torque",
        "enable_torque",
    ):
        original: Callable[..., Any] = getattr(bus, method_name)

        @wraps(original)
        def wrapper(*args, _original=original, **kwargs):
            kwargs["num_retry"] = _effective_retry(
                int(kwargs.get("num_retry", 0)), default_retry
            )
            return _original(*args, **kwargs)

        setattr(bus, method_name, wrapper)


def _make_resilient_sync_read(default_retry: int, retry_delay_s: float):
    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ):
        effective = _effective_retry(num_retry, default_retry)
        self._setup_sync_reader(motor_ids, addr, length)
        comm = 0
        for n_try in range(1 + effective):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break
            if n_try < effective:
                time.sleep(retry_delay_s)

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(
                f"{err_msg} {self.packet_handler.getTxRxResult(comm)}"
            )

        values = {
            motor_id: self.sync_reader.getData(motor_id, addr, length)
            for motor_id in motor_ids
        }
        return values, comm

    return _sync_read


def _make_resilient_record_ranges(default_retry: int, retry_delay_s: float):
    def record_ranges_of_motion(
        self,
        motors=None,
        display_values: bool = True,
    ):
        if motors is None:
            motors = list(self.motors)
        elif isinstance(motors, (str, int)):
            motors = [motors]
        elif not isinstance(motors, list):
            raise TypeError(motors)

        def read_positions() -> dict[str, Any]:
            return self.sync_read(
                "Present_Position", motors, normalize=False, num_retry=default_retry
            )

        start_positions = None
        while start_positions is None:
            try:
                start_positions = read_positions()
            except ConnectionError:
                time.sleep(retry_delay_s)

        mins = start_positions.copy()
        maxes = start_positions.copy()
        user_pressed_enter = False
        while not user_pressed_enter:
            try:
                positions = read_positions()
            except ConnectionError:
                time.sleep(retry_delay_s)
                continue

            mins = {
                motor: min(positions[motor], min_) for motor, min_ in mins.items()
            }
            maxes = {
                motor: max(positions[motor], max_) for motor, max_ in maxes.items()
            }

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motors:
                    print(
                        f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}"
                    )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(motors) + 3)

        same_min_max = [motor for motor in motors if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(
                f"Some motors have the same min and max values:\n{pformat(same_min_max)}"
            )

        return mins, maxes

    return record_ranges_of_motion
