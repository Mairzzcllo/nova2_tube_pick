"""Nova2Client 离线测试。

覆盖：
    1. SDK 返回字符串解析（错误码 / 数值载荷）
    2. 位姿与关节角读取
    3. move_j / move_l 的速度参数映射
    4. 非零返回码与超时异常分类
    5. 基于反馈快照的停止判断与异常状态

说明：
    不连接真实硬件；通过注入假 Dashboard 与手工快照验证纯逻辑。

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_nova2_client.py
"""

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import FatalError, RecoverableError
from common.types import Pose
from robot.nova2_client import Nova2Client, _FeedbackSnapshot


# ============================================================
# 1. 测试替身
# ============================================================

class FakeDashboard:
    """最小假 Dashboard，只返回固定字符串并记录调用。"""

    def __init__(self):
        self.movj_calls = []
        self.movl_calls = []
        self.stop_calls = 0

    def GetPose(self):
        return "0,{10.0,20.0,30.0,1.0,2.0,3.0},GetPose();"

    def GetAngle(self):
        return "0,{1.0,2.0,3.0,4.0,5.0,6.0},GetAngle();"

    def RobotMode(self):
        return "0,{5},RobotMode();"

    def MovJ(self, *args, **kwargs):
        self.movj_calls.append((args, kwargs))
        return "0,{},MovJ();"

    def MovL(self, *args, **kwargs):
        self.movl_calls.append((args, kwargs))
        return "0,{},MovL();"

    def Stop(self):
        self.stop_calls += 1
        return "0,{},Stop();"


def build_client(feedback_port=None):
    """构造已连接状态、使用假 Dashboard 的客户端。"""

    client = Nova2Client("127.0.0.1", 29999, feedback_port=feedback_port)
    client._dashboard = FakeDashboard()
    client._connected = True
    return client


def build_snapshot(robot_mode=5, speed=(0.0, 0.0, 0.0), age_s=0.0):
    """构造反馈快照。"""

    speed_norm = (speed[0] ** 2 + speed[1] ** 2 + speed[2] ** 2) ** 0.5
    return _FeedbackSnapshot(
        timestamp_s=time.monotonic() - age_s,
        robot_mode=robot_mode,
        enable_status=1,
        running_status=0,
        error_status=0,
        tcp_speed_mm_s=speed,
        tcp_speed_norm_mm_s=speed_norm,
    )


# ============================================================
# 2. 解析测试
# ============================================================

def test_parse_reply():
    """验证错误码与载荷拆分。"""

    code, payload = Nova2Client._parse_reply("0,{1,2,3},GetPose();")
    assert code == 0
    assert payload.startswith("{1,2,3}")

    code, _ = Nova2Client._parse_reply("-1,{...},MovJ();")
    assert code == -1

    try:
        Nova2Client._parse_reply("")
        raise AssertionError("空返回应抛 FatalError")
    except FatalError:
        pass

    print("[PASS] 返回字符串解析")


def test_extract_numbers():
    """验证花括号与无花括号两种载荷。"""

    assert Nova2Client._extract_numbers("{1,2,3}") == [1.0, 2.0, 3.0]
    assert Nova2Client._extract_numbers("4.5,6.5,Stop();") == [4.5, 6.5]

    try:
        Nova2Client._extract_numbers("no-number")
        raise AssertionError("无数值应抛 FatalError")
    except FatalError:
        pass

    print("[PASS] 数值载荷提取")


# ============================================================
# 3. 读取测试
# ============================================================

def test_read_pose_and_joints():
    """验证位姿与关节角转换。"""

    client = build_client()

    pose = client.get_pose()
    assert isinstance(pose, Pose)
    assert (pose.x_mm, pose.y_mm, pose.z_mm) == (10.0, 20.0, 30.0)
    assert (pose.rx_deg, pose.ry_deg, pose.rz_deg) == (1.0, 2.0, 3.0)

    joints = client.get_joint_angles()
    assert joints == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    print("[PASS] 位姿与关节角读取")


# ============================================================
# 4. 运动指令测试
# ============================================================

def test_move_commands():
    """验证 move_j / move_l 参数映射与范围校验。"""

    client = build_client()
    pose = Pose(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    client.move_j(pose, 40)
    args, kwargs = client._dashboard.movj_calls[-1]
    assert args == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0)
    assert kwargs["v"] == 40

    client.move_l(pose, 10)
    args, kwargs = client._dashboard.movl_calls[-1]
    assert args == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0)
    assert kwargs["v"] == 10

    for bad in (0, 101, -5, "10"):
        try:
            client.move_j(pose, bad)
            raise AssertionError(f"非法速度未抛异常：{bad!r}")
        except ValueError:
            pass

    try:
        client.move_j((1, 2, 3, 4, 5, 6), 50)
        raise AssertionError("非 Pose 入参未抛异常")
    except ValueError:
        pass

    print("[PASS] 运动指令参数映射")


def test_reply_error_mapping():
    """验证非零返回码与超时的异常分类。"""

    client = build_client()

    try:
        client._check_reply("1,{},MovJ();", "move_j")
        raise AssertionError("非零返回码应抛 FatalError")
    except FatalError:
        pass

    try:
        client._check_reply("2,{timeout},MovL();", "move_l")
        raise AssertionError("超时应抛 RecoverableError")
    except RecoverableError:
        pass

    assert client._check_reply("0,{ok},Stop();", "stop") == "{ok},Stop();"

    print("[PASS] 返回码异常分类")


# ============================================================
# 5. 停止判断测试
# ============================================================

def test_wait_until_stopped_feedback():
    """验证反馈快照停止判断。"""

    client = build_client()
    client._feedback = object()

    client._snapshot = build_snapshot(robot_mode=5, speed=(0.0, 0.0, 0.0))
    assert client.wait_until_stopped(0.5) is True

    client._snapshot = build_snapshot(robot_mode=7, speed=(50.0, 0.0, 0.0))
    assert client.wait_until_stopped(0.05) is False

    print("[PASS] 反馈快照停止判断")


def test_wait_until_stopped_errors():
    """验证反馈异常状态与快照过期。"""

    client = build_client()
    client._feedback = object()

    client._snapshot = build_snapshot(robot_mode=9, speed=(0.0, 0.0, 0.0))
    try:
        client.wait_until_stopped(0.5)
        raise AssertionError("异常状态应抛 FatalError")
    except FatalError:
        pass

    client._snapshot = build_snapshot(robot_mode=5, speed=(0.0, 0.0, 0.0), age_s=2.0)
    try:
        client.wait_until_stopped(0.5)
        raise AssertionError("过期快照应抛 FatalError")
    except FatalError:
        pass

    print("[PASS] 反馈异常状态处理")


def test_build_snapshot():
    """验证反馈帧到快照的字段提取与速度模长。"""

    record = {
        "RobotMode": 5,
        "EnableStatus": 1,
        "RunningStatus": 0,
        "ErrorStatus": 0,
        "TCPSpeedActual": [3.0, 4.0, 0.0],
    }

    snapshot = Nova2Client._build_snapshot(record)
    assert snapshot.robot_mode == 5
    assert snapshot.tcp_speed_mm_s == (3.0, 4.0, 0.0)
    assert abs(snapshot.tcp_speed_norm_mm_s - 5.0) < 1e-9

    print("[PASS] 反馈帧字段提取")


# ============================================================
# 6. 入口
# ============================================================

def main() -> None:
    """依次执行全部离线测试。"""

    test_parse_reply()
    test_extract_numbers()
    test_read_pose_and_joints()
    test_move_commands()
    test_reply_error_mapping()
    test_wait_until_stopped_feedback()
    test_wait_until_stopped_errors()
    test_build_snapshot()
    print("Nova2Client 离线测试全部通过")


if __name__ == "__main__":
    main()
