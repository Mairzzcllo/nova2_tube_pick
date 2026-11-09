"""GripperController 硬件联调脚本（默认跳过）。

覆盖：
    1. 从配置加载夹爪参数并连接 Nova 2
    2. 空载开合与夹持反馈读取

说明：
    需要真实 Nova 2 与夹爪。`run_smoke_tests.py` 默认跳过 `*_hw.py`。
    当前 `config.gripper.control_mode` 默认是 `unknown`，协议确定前本脚本
    会在开合步骤抛出 `GripperError`，属于预期行为。

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_gripper_controller_hw.py
"""

import os
import sys
import time
from dataclasses import fields

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml

from common.types import GripperConfig
from robot.gripper_controller import GripperController
from robot.nova2_client import DEFAULT_FEEDBACK_PORT, Nova2Client


# ============================================================
# 1. 配置加载
# ============================================================

def load_gripper_config(path: str = "config/robot_config.yaml") -> tuple[dict, GripperConfig]:
    """读取机器人与夹爪配置。

    Returns:
        `(robot_config, gripper_config)`
    """

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    # 只取 GripperConfig 已定义的字段，忽略 type/宽度等占位键
    names = {field.name for field in fields(GripperConfig)}
    gripper_values = {
        key: value for key, value in raw["gripper"].items() if key in names
    }

    return raw["robot"], GripperConfig(**gripper_values)


# ============================================================
# 2. 硬件联调
# ============================================================

def main() -> None:
    """连接真实硬件并执行空载开合。"""

    robot_config, gripper_config = load_gripper_config()

    client = Nova2Client(
        ip=robot_config["ip"],
        port=robot_config["port"],
        feedback_port=robot_config.get("feedback_port", DEFAULT_FEEDBACK_PORT),
    )
    controller = GripperController(client, gripper_config)

    client.connect()
    try:
        print(f"夹爪控制方式：{gripper_config.control_mode}")
        print(f"初始夹持状态：{controller.is_grasped()}")

        print("张开夹爪...")
        controller.open()
        time.sleep(1.0)

        print("闭合夹爪...")
        controller.close()
        time.sleep(1.0)

        print(f"最终夹持状态：{controller.is_grasped()}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
