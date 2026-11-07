"""pose_recorder 离线冒烟测试。

说明：
    使用合成棋盘格图像与假机器人/假相机，验证：
        1. 正常采集样本数量、图像与位姿一一对应
        2. 棋盘格检测失败时跳过且不写伪数据
        3. 机械臂未停止时跳过
        4. 有效样本不足时抛 CalibrationError
        5. 输出目录已有产物时拒绝覆盖

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_pose_recorder.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.errors import CalibrationError
from common.types import Pose

SRC = ROOT / "calibration" / "pose_recorder.py"
spec = importlib.util.spec_from_file_location("pose_recorder_mod", SRC)
pose_recorder_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_recorder_mod)

PoseRecorder = pose_recorder_mod.PoseRecorder


# ============================================================
# 1. 合成棋盘格图像
# ============================================================

W, H = 1280, 960
K_TRUE = np.array([
    [1000.0, 0.0, W / 2],
    [0.0, 1000.0, H / 2],
    [0.0, 0.0, 1.0],
])
D_TRUE = np.zeros(5)

PATTERN = (9, 6)
SQUARE = 25.0


def build_chessboard_template():
    """构造棋盘格模板图与模板外四角像素坐标。"""

    cell, margin = 60, 120

    board_width = (PATTERN[0] + 1) * cell + 2 * margin
    board_height = (PATTERN[1] + 1) * cell + 2 * margin

    board = np.full((board_height, board_width), 255, np.uint8)

    for row in range(PATTERN[1] + 1):
        for col in range(PATTERN[0] + 1):
            if (row + col) % 2 == 0:
                y0 = margin + row * cell
                y1 = y0 + cell
                x0 = margin + col * cell
                x1 = x0 + cell
                board[y0:y1, x0:x1] = 0

    src_pts = np.array([
        [margin, margin],
        [margin + (PATTERN[0] + 1) * cell, margin],
        [
            margin + (PATTERN[0] + 1) * cell,
            margin + (PATTERN[1] + 1) * cell,
        ],
        [margin, margin + (PATTERN[1] + 1) * cell],
    ], np.float32)

    outer_obj = np.array([
        [-SQUARE, -SQUARE, 0],
        [PATTERN[0] * SQUARE, -SQUARE, 0],
        [PATTERN[0] * SQUARE, PATTERN[1] * SQUARE, 0],
        [-SQUARE, PATTERN[1] * SQUARE, 0],
    ], np.float32)

    return board, src_pts, outer_obj


def make_chessboard_images(count):
    """合成指定数量的不同位姿棋盘格图像。"""

    board, src_pts, outer_obj = build_chessboard_template()
    rng = np.random.default_rng(0)
    images = []
    attempts = 0

    while len(images) < count and attempts < 500:
        attempts += 1

        rvec = np.deg2rad(rng.uniform(-20, 20, 3)).astype(np.float64)
        tvec = np.array([0.0, 0.0, 420.0]) + (
            rng.uniform(-60, 60, 3) * np.array([1.0, 1.0, 0.0])
        )

        img_pts, _ = cv2.projectPoints(
            outer_obj, rvec, tvec, K_TRUE, D_TRUE
        )
        img_pts = img_pts.reshape(4, 2).astype(np.float32)

        if (
            np.any(img_pts < 40)
            or np.any(img_pts[:, 0] > W - 40)
            or np.any(img_pts[:, 1] > H - 40)
        ):
            continue

        transform = cv2.getPerspectiveTransform(src_pts, img_pts)
        image = cv2.warpPerspective(
            board, transform, (W, H), borderValue=255
        )
        images.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    if len(images) < count:
        raise RuntimeError("合成棋盘格图像失败")

    return images


# ============================================================
# 2. 假机器人 / 假相机
# ============================================================

class FakeRobot:
    """按顺序返回预设位姿的假机器人。"""

    def __init__(self, poses, stopped=True):
        self._poses = poses
        self.stopped = stopped
        self.pose_calls = 0

    def wait_until_stopped(self, timeout_s=5.0):
        return self.stopped

    def get_pose(self):
        pose = self._poses[self.pose_calls % len(self._poses)]
        self.pose_calls += 1
        return pose


class FakeCamera:
    """循环返回预设图像的假相机。"""

    def __init__(self, images):
        self._images = images
        self._index = 0

    def read(self):
        image = self._images[self._index % len(self._images)]
        self._index += 1
        return image


def make_fake_poses(count):
    """构造一组互不相同的假位姿。"""

    return [
        Pose(
            x_mm=400.0 + 10.0 * index,
            y_mm=100.0 - 5.0 * index,
            z_mm=300.0 + 2.0 * index,
            rx_deg=180.0,
            ry_deg=float(index),
            rz_deg=90.0,
        )
        for index in range(count)
    ]


def make_recorder(output_dir, images, poses, stopped=True):
    """组装被测采集器，并替换交互触发为自动继续。"""

    robot = FakeRobot(poses, stopped=stopped)
    camera = FakeCamera(images)

    recorder = PoseRecorder(
        robot=robot,
        camera=camera,
        output_dir=output_dir,
        pattern_size=PATTERN,
    )
    recorder._wait_for_trigger = lambda: True

    return recorder, robot


# ============================================================
# 3. 测试用例
# ============================================================

def test_record_success():
    """正常采集：样本数、图像与位姿一一对应。"""

    count = 6
    out_dir = tempfile.mkdtemp(prefix="pose_recorder_ok_")

    images = make_chessboard_images(count)
    poses = make_fake_poses(count)

    recorder, _robot = make_recorder(out_dir, images, poses)
    recorder.record(count=count)

    pose_file = Path(out_dir) / "poses.json"

    assert pose_file.exists(), "poses.json 未生成"
    assert not (Path(out_dir) / "poses.json.tmp").exists(), "临时文件未清理"

    with open(pose_file, "r", encoding="utf-8") as file:
        payload = json.load(file)

    samples = payload["samples"]

    assert len(samples) == count, f"样本数错误：{len(samples)}"
    assert payload["pattern_size"] == list(PATTERN)
    assert payload["robot_pose_units"] == {
        "position": "mm",
        "rotation": "deg",
    }

    for index, sample in enumerate(samples):
        assert sample["sample_id"] == index
        assert set(sample["robot_pose"].keys()) == {
            "x_mm", "y_mm", "z_mm",
            "rx_deg", "ry_deg", "rz_deg",
        }
        assert sample["robot_pose"]["x_mm"] == poses[index].x_mm

        image_path = Path(out_dir) / sample["image_path"]
        assert image_path.exists(), f"图像缺失：{image_path}"

    print(f"[PASS] 正常采集 {count} 个样本")


def test_skip_failed_detection():
    """棋盘格检测失败：样本不足抛 CalibrationError，且不写伪数据。"""

    out_dir = tempfile.mkdtemp(prefix="pose_recorder_blank_")
    blank = np.full((H, W, 3), 255, np.uint8)

    recorder, _robot = make_recorder(
        out_dir,
        images=[blank],
        poses=make_fake_poses(1),
    )

    try:
        recorder.record(count=2)
    except CalibrationError as error:
        assert error.recoverable is True
    else:
        raise AssertionError("检测失败时应抛 CalibrationError")

    assert not (Path(out_dir) / "poses.json").exists(), "不得写伪数据"
    assert not any((Path(out_dir) / "images").iterdir()), "不得保存失败图像"

    print("[PASS] 检测失败被正确跳过")


def test_skip_robot_not_stopped():
    """机械臂未停止：不读取位姿，最终样本不足。"""

    out_dir = tempfile.mkdtemp(prefix="pose_recorder_stop_")
    images = make_chessboard_images(1)

    recorder, robot = make_recorder(
        out_dir,
        images=images,
        poses=make_fake_poses(1),
        stopped=False,
    )

    try:
        recorder.record(count=1)
    except CalibrationError as error:
        assert error.recoverable is True
    else:
        raise AssertionError("机械臂未停止时应抛 CalibrationError")

    assert robot.pose_calls == 0, "未停止时不应读取位姿"
    assert not (Path(out_dir) / "poses.json").exists()

    print("[PASS] 机械臂未停止被正确跳过")


def test_refuse_overwrite():
    """输出目录已有产物：默认拒绝覆盖。"""

    out_dir = tempfile.mkdtemp(prefix="pose_recorder_exist_")
    pose_file = Path(out_dir) / "poses.json"

    with open(pose_file, "w", encoding="utf-8") as file:
        json.dump({"samples": []}, file)

    recorder, _robot = make_recorder(
        out_dir,
        images=make_chessboard_images(1),
        poses=make_fake_poses(1),
    )

    try:
        recorder.record(count=1)
    except CalibrationError as error:
        assert error.recoverable is False
    else:
        raise AssertionError("已有产物时应拒绝覆盖")

    print("[PASS] 已有产物被正确拒绝")


def main() -> int:
    """依次执行全部离线测试。"""

    test_record_success()
    test_skip_failed_detection()
    test_skip_robot_not_stopped()
    test_refuse_overwrite()

    print("\n[PASS] pose_recorder 全部离线测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
