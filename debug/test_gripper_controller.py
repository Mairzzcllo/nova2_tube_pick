"""GripperController 离线测试。

覆盖：
    1. 协议未配置（unknown）时安全降级
    2. tool_do 模式的 DO 编号与电平映射
    3. 夹持反馈到 GraspStatus 的映射
    4. 缺少 DO 编号、客户端异常、未实现协议的异常分类
    5. 工具电源按需上电

说明：
    不连接真实硬件；通过注入假 client 验证纯逻辑。

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_gripper_controller.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import FatalError, GripperError
from common.types import GraspStatus, GripperConfig
from robot.gripper_controller import GripperController


# ============================================================
# 1. 测试替身
# ============================================================

class FakeClient:
    """最小假 Nova2Client，记录 IO 调用并返回可配置的状态。"""

    def __init__(self, di_value=False, do_error=None, di_error=None):
        self.do_calls = []
        self.di_calls = []
        self.power_calls = []
        self.di_value = di_value
        self.do_error = do_error
        self.di_error = di_error

    def set_tool_do(self, index, on):
        if self.do_error is not None:
            raise self.do_error
        self.do_calls.append((index, on))

    def read_tool_di(self, index):
        if self.di_error is not None:
            raise self.di_error
        self.di_calls.append(index)
        return self.di_value

    def set_tool_power(self, on):
        self.power_calls.append(on)


# ============================================================
# 2. 安全降级测试
# ============================================================

def test_unknown_mode_is_safe():
    """验证 unknown 模式不开合、不假装夹持成功。"""

    controller = GripperController(FakeClient(), GripperConfig())

    assert controller.is_grasped() is GraspStatus.UNKNOWN

    for action in (controller.open, controller.close):
        try:
            action()
            raise AssertionError("unknown 模式开合应抛 GripperError")
        except GripperError as exc:
            assert exc.recoverable is False

    print("[PASS] unknown 模式安全降级")


def test_pending_modes_raise():
    """验证 dual_do / serial 仍为占位。"""

    for mode in ("dual_do", "serial"):
        controller = GripperController(
            FakeClient(),
            GripperConfig(control_mode=mode, open_do_index=1, close_do_index=2),
        )
        try:
            controller.open()
            raise AssertionError(f"{mode} 模式应抛 GripperError")
        except GripperError as exc:
            assert exc.recoverable is False

    print("[PASS] 未实现协议占位")


# ============================================================
# 3. tool_do 输出测试
# ============================================================

def test_tool_do_open_and_close():
    """验证单 DO 模式下开/合使用相反电平。"""

    client = FakeClient()
    controller = GripperController(
        client,
        GripperConfig(
            control_mode="tool_do",
            open_do_index=3,
            do_active_level=1,
        ),
    )

    controller.open()
    assert client.do_calls[-1] == (3, True)

    controller.close()
    assert client.do_calls[-1] == (3, False)

    print("[PASS] tool_do 开合电平映射")


def test_tool_do_low_active_level():
    """验证低电平有效时开/合电平反转。"""

    client = FakeClient()
    controller = GripperController(
        client,
        GripperConfig(
            control_mode="tool_do",
            open_do_index=0,
            do_active_level=0,
        ),
    )

    controller.open()
    assert client.do_calls[-1] == (0, False)

    controller.close()
    assert client.do_calls[-1] == (0, True)

    print("[PASS] 低电平有效映射")


def test_missing_do_index():
    """验证缺少 DO 编号时抛 GripperError。"""

    controller = GripperController(
        FakeClient(),
        GripperConfig(control_mode="tool_do", open_do_index=None),
    )

    try:
        controller.open()
        raise AssertionError("缺少 DO 编号应抛 GripperError")
    except GripperError as exc:
        assert exc.recoverable is False

    print("[PASS] 缺少 DO 编号检查")


# ============================================================
# 4. 反馈测试
# ============================================================

def test_feedback_mapping():
    """验证 DI 到 GraspStatus 的映射。"""

    config = GripperConfig(
        control_mode="tool_do",
        open_do_index=1,
        feedback_di_index=0,
        feedback_active_level=1,
    )

    high = GripperController(FakeClient(di_value=True), config)
    assert high.is_grasped() is GraspStatus.GRASPED

    low = GripperController(FakeClient(di_value=False), config)
    assert low.is_grasped() is GraspStatus.NOT_GRASPED

    print("[PASS] 夹持反馈映射")


def test_feedback_error_wrapped():
    """验证底层反馈异常转换为 GripperError。"""

    config = GripperConfig(
        control_mode="tool_do",
        open_do_index=1,
        feedback_di_index=0,
    )
    controller = GripperController(
        FakeClient(di_error=FatalError("串口断开")),
        config,
    )

    try:
        controller.is_grasped()
        raise AssertionError("反馈读取失败应抛 GripperError")
    except GripperError as exc:
        assert exc.recoverable is False

    print("[PASS] 反馈异常转换")


# ============================================================
# 5. 工具电源与入参测试
# ============================================================

def test_tool_power_on_demand():
    """验证按配置只上电一次。"""

    client = FakeClient()
    controller = GripperController(
        client,
        GripperConfig(
            control_mode="tool_do",
            open_do_index=1,
            tool_power_required=True,
        ),
    )

    controller.open()
    controller.close()
    assert client.power_calls == [True]

    print("[PASS] 工具电源按需上电")


def test_invalid_config_type():
    """验证构造参数类型检查。"""

    try:
        GripperController(FakeClient(), config={"control_mode": "tool_do"})
        raise AssertionError("非 GripperConfig 应抛 ValueError")
    except ValueError:
        pass

    print("[PASS] 配置类型检查")


# ============================================================
# 6. 入口
# ============================================================

def main() -> None:
    """依次执行全部离线测试。"""

    test_unknown_mode_is_safe()
    test_pending_modes_raise()
    test_tool_do_open_and_close()
    test_tool_do_low_active_level()
    test_missing_do_index()
    test_feedback_mapping()
    test_feedback_error_wrapped()
    test_tool_power_on_demand()
    test_invalid_config_type()
    print("GripperController 离线测试全部通过")


if __name__ == "__main__":
    main()
