"""MotionController 离线测试。

覆盖：
    1. move_to / move_linear 的调用顺序与默认速度
    2. 当前位置已在容差内时跳过运动
    3. 运动未启动、未停止、停止后未到位的异常处理
    4. 底层 FatalError 原样透传
    5. wait_arrival 委托与非法超时
    6. 位姿/速度校验与角度环绕

说明：
    不连接真实硬件；通过注入假 client 验证纯逻辑。

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_motion_controller.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import FatalError, RecoverableError
from common.types import Pose
from robot.motion_controller import MotionController


# ============================================================
# 1. 测试替身
# ============================================================

class FakeClient:
    """最小假 Nova2Client，记录调用并返回可配置的状态。"""

    def __init__(
        self,
        pose_before,
        pose_after=None,
        moving=True,
        stopped=True,
        move_error=None,
        start_error=None,
        stop_error=None,
    ):
        self.pose = pose_before
        self.pose_after = pose_after if pose_after is not None else pose_before
        self.moving = moving
        self.stopped = stopped
        self.move_error = move_error
        self.start_error = start_error
        self.stop_error = stop_error
        self.calls = []
        self.stop_calls = 0

    def get_pose(self):
        return self.pose

    def move_j(self, pose, velocity_ratio_pct):
        self.calls.append(("move_j", pose, velocity_ratio_pct))
        if self.move_error is not None:
            raise self.move_error
        self.pose = self.pose_after

    def move_l(self, pose, velocity_ratio_pct):
        self.calls.append(("move_l", pose, velocity_ratio_pct))
        if self.move_error is not None:
            raise self.move_error
        self.pose = self.pose_after

    def wait_until_moving(self, timeout_s):
        self.calls.append(("wait_until_moving", timeout_s))
        if self.start_error is not None:
            raise self.start_error
        return self.moving

    def wait_until_stopped(self, timeout_s):
        self.calls.append(("wait_until_stopped", timeout_s))
        return self.stopped

    def stop(self):
        self.calls.append(("stop",))
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


def near_pose():
    """用于返回“目标已在容差内”的位姿。"""

    return Pose(100.0, 100.0, 100.0, 0.0, 0.0, 0.0)


def far_target():
    """远离当前位姿的目标。"""

    return Pose(200.0, 100.0, 100.0, 0.0, 0.0, 0.0)


# ============================================================
# 2. 正常运动测试
# ============================================================

def test_move_to_calls_joint_and_waits():
    """验证 move_to 调用 move_j 并按默认速度等待到位。"""

    client = FakeClient(pose_before=near_pose(), pose_after=far_target())
    controller = MotionController(client)

    controller.move_to(far_target())

    assert client.calls[0][0] == "move_j"
    assert client.calls[0][2] == 50
    assert [call[0] for call in client.calls] == [
        "move_j",
        "wait_until_moving",
        "wait_until_stopped",
    ]

    print("[PASS] move_to 关节运动与到位等待")


def test_move_linear_default_speed():
    """验证 move_linear 使用默认直线速度 10。"""

    client = FakeClient(pose_before=near_pose(), pose_after=far_target())
    controller = MotionController(client)

    controller.move_linear(far_target())

    assert client.calls[0][0] == "move_l"
    assert client.calls[0][2] == 10
    assert client.stop_calls == 0

    print("[PASS] move_linear 默认速度")


def test_skip_when_already_at_pose():
    """验证当前位姿已在容差内时不发运动指令。"""

    client = FakeClient(pose_before=near_pose(), pose_after=near_pose())
    controller = MotionController(client)

    controller.move_to(near_pose())

    assert client.calls == []
    assert client.stop_calls == 0

    print("[PASS] 已在目标位姿时跳过运动")


def test_angle_wrap_is_within_tolerance():
    """验证 179.9 与 -179.9 的角度环绕不误判。"""

    current = Pose(0.0, 0.0, 0.0, 179.9, 0.0, 0.0)
    target = Pose(0.0, 0.0, 0.0, -179.9, 0.0, 0.0)

    client = FakeClient(pose_before=current, pose_after=current)
    controller = MotionController(client)

    assert controller._is_at_pose(target) is True

    print("[PASS] 角度环绕容差")


# ============================================================
# 3. 异常路径测试
# ============================================================

def test_not_started_and_not_at_pose():
    """验证指令未触发运动且未到位时停止并抛 RecoverableError。"""

    client = FakeClient(pose_before=near_pose(), moving=False)
    controller = MotionController(client)

    try:
        controller.move_to(far_target())
        raise AssertionError("未启动且未到位应抛 RecoverableError")
    except RecoverableError as exc:
        assert exc.recoverable is True

    assert client.stop_calls == 1

    print("[PASS] 未启动异常处理")


def test_not_started_but_at_pose():
    """验证未观测到运动但目标已在容差内时视为成功。"""

    client = FakeClient(
        pose_before=near_pose(),
        pose_after=far_target(),
        moving=False,
    )
    controller = MotionController(client)

    controller.move_to(far_target())

    assert client.stop_calls == 0

    print("[PASS] 未启动但已在目标容差")


def test_not_stopped():
    """验证等待停止超时后停止并抛 RecoverableError。"""

    client = FakeClient(pose_before=near_pose(), stopped=False)
    controller = MotionController(client)

    try:
        controller.move_linear(far_target())
        raise AssertionError("未停止应抛 RecoverableError")
    except RecoverableError:
        pass

    assert client.stop_calls == 1

    print("[PASS] 未停止异常处理")


def test_stopped_but_not_at_pose():
    """验证已停止但未到达目标时停止并抛 RecoverableError。"""

    client = FakeClient(pose_before=near_pose(), pose_after=near_pose())
    controller = MotionController(client)

    try:
        controller.move_to(far_target())
        raise AssertionError("未到位应抛 RecoverableError")
    except RecoverableError:
        pass

    assert client.stop_calls == 1

    print("[PASS] 停止后未到位异常处理")


def test_fatal_error_propagates():
    """验证底层 FatalError 不被包装。"""

    client = FakeClient(
        pose_before=near_pose(),
        move_error=FatalError("触发限位"),
    )
    controller = MotionController(client)

    try:
        controller.move_to(far_target())
        raise AssertionError("FatalError 应透传")
    except FatalError as exc:
        assert exc.recoverable is False

    assert client.stop_calls == 0

    print("[PASS] FatalError 透传")


def test_stop_error_does_not_mask_original():
    """验证停止失败不会掩盖原始 RecoverableError。"""

    client = FakeClient(
        pose_before=near_pose(),
        moving=False,
        stop_error=FatalError("停止指令失败"),
    )
    controller = MotionController(client)

    try:
        controller.move_to(far_target())
        raise AssertionError("应抛 RecoverableError")
    except RecoverableError:
        pass

    print("[PASS] 停止失败不掩盖原始异常")


# ============================================================
# 4. wait_arrival 与入参校验
# ============================================================

def test_wait_arrival_delegates():
    """验证 wait_arrival 直接委托 wait_until_stopped。"""

    client = FakeClient(pose_before=near_pose(), stopped=True)
    controller = MotionController(client)

    assert controller.wait_arrival(3.0) is True
    assert client.calls[-1] == ("wait_until_stopped", 3.0)

    for bad in (0, -1, "3"):
        try:
            controller.wait_arrival(bad)
            raise AssertionError(f"非法超时未抛异常：{bad!r}")
        except ValueError:
            pass

    print("[PASS] wait_arrival 委托与超时校验")


def test_invalid_pose_and_velocity():
    """验证位姿与速度参数的防御式校验。"""

    client = FakeClient(pose_before=near_pose())
    controller = MotionController(client)

    for bad_pose in ((1, 2, 3, 4, 5, 6), Pose(float("nan"), 0, 0, 0, 0, 0)):
        try:
            controller.move_to(bad_pose)
            raise AssertionError(f"非法位姿未抛异常：{bad_pose!r}")
        except ValueError:
            pass

    for bad_velocity in (0, 101, -5, "50"):
        try:
            controller.move_to(far_target(), bad_velocity)
            raise AssertionError(f"非法速度未抛异常：{bad_velocity!r}")
        except ValueError:
            pass

    print("[PASS] 位姿与速度校验")


def test_none_client():
    """验证构造 client 为 None 时抛 ValueError。"""

    try:
        MotionController(None)
        raise AssertionError("None client 应抛 ValueError")
    except ValueError:
        pass

    print("[PASS] client 构造校验")


# ============================================================
# 5. 入口
# ============================================================

def main() -> None:
    """依次执行全部离线测试。"""

    test_move_to_calls_joint_and_waits()
    test_move_linear_default_speed()
    test_skip_when_already_at_pose()
    test_angle_wrap_is_within_tolerance()
    test_not_started_and_not_at_pose()
    test_not_started_but_at_pose()
    test_not_stopped()
    test_stopped_but_not_at_pose()
    test_fatal_error_propagates()
    test_stop_error_does_not_mask_original()
    test_wait_arrival_delegates()
    test_invalid_pose_and_velocity()
    test_none_client()
    print("MotionController 离线测试全部通过")


if __name__ == "__main__":
    main()
