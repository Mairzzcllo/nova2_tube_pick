"""手眼标定模块离线测试。

覆盖：
    1. eye-to-hand AX=XB 求解的正确性与矩阵方向
    2. 固定轴 XYZ 欧拉角与位姿矩阵转换
    3. 姿态退化检测
    4. poses.json 契约校验
    5. image_path 相对 poses.json 的解析

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_handeye_calibration.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 确保可从仓库根目录导入项目模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibration.camera_calibration import create_object_points
from calibration.handeye_calibration import (
    calibrate_handeye,
    check_motion_diversity,
    euler_fixed_xyz_to_matrix,
    evaluate_rigidity,
    load_pose_payload,
    pose_dict_to_matrix,
    resolve_image_path,
    solve_eye_to_hand,
    validate_pose_payload,
)
from common.errors import CalibrationError


# ============================================================
# 1. 测试工具
# ============================================================

def random_transform(rng: np.random.Generator) -> np.ndarray:
    """生成随机刚体变换。"""

    rotation_vector = rng.uniform(-0.8, 0.8, 3)
    rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = rng.uniform(-300.0, 300.0, 3)

    return transform


def assert_matrix_close(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
    message: str,
) -> None:
    """断言两个矩阵的逐元素误差在容差内。"""

    error = float(np.max(np.abs(actual - expected)))
    assert error <= tolerance, f"{message}：最大误差 {error:.3e}"


# ============================================================
# 2. eye-to-hand 求解
# ============================================================

def test_eye_to_hand_roundtrip() -> None:
    """已知 T_base_camera 合成数据，应能精确恢复。"""

    rng = np.random.default_rng(20261107)

    true_base_camera = random_transform(rng)
    true_gripper_board = random_transform(rng)

    base_gripper_transforms = []
    cam_board_transforms = []

    for _ in range(12):

        base_gripper = random_transform(rng)

        cam_board = (
            np.linalg.inv(true_base_camera)
            @ base_gripper
            @ true_gripper_board
        )

        base_gripper_transforms.append(base_gripper)
        cam_board_transforms.append(cam_board)

    estimated = solve_eye_to_hand(
        base_gripper_transforms,
        cam_board_transforms,
    )

    assert_matrix_close(
        estimated,
        true_base_camera,
        1e-8,
        "eye-to-hand 求解方向错误",
    )

    translation_std_mm, rotation_std_deg = evaluate_rigidity(
        base_gripper_transforms,
        cam_board_transforms,
        estimated,
    )

    assert translation_std_mm < 1e-6, (
        f"刚性平移一致性异常：{translation_std_mm:.3e} mm"
    )
    assert rotation_std_deg < 1e-6, (
        f"刚性旋转一致性异常：{rotation_std_deg:.3e} deg"
    )


# ============================================================
# 3. 欧拉角与位姿矩阵
# ============================================================

def test_euler_fixed_xyz() -> None:
    """校验固定轴 XYZ 的旋转组合顺序。"""

    identity = euler_fixed_xyz_to_matrix(0.0, 0.0, 0.0)
    assert_matrix_close(identity, np.eye(3), 1e-12, "零角度应为单位阵")

    rotation_x_90 = euler_fixed_xyz_to_matrix(90.0, 0.0, 0.0)
    expected_x_90 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert_matrix_close(
        rotation_x_90,
        expected_x_90,
        1e-12,
        "Rx(90) 错误",
    )

    rotation_z_90 = euler_fixed_xyz_to_matrix(0.0, 0.0, 90.0)
    expected_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert_matrix_close(
        rotation_z_90,
        expected_z_90,
        1e-12,
        "Rz(90) 错误",
    )

    pose = {
        "x_mm": 10.0,
        "y_mm": -20.0,
        "z_mm": 30.0,
        "rx_deg": 0.0,
        "ry_deg": 0.0,
        "rz_deg": 0.0,
    }
    transform = pose_dict_to_matrix(pose)
    assert np.allclose(transform[:3, 3], [10.0, -20.0, 30.0]), (
        "位姿平移转换错误"
    )


# ============================================================
# 4. 姿态退化
# ============================================================

def test_degenerate_motion() -> None:
    """机器人只平移不旋转时，应拒绝求解。"""

    transform = np.eye(4, dtype=np.float64)
    base_gripper_transforms = []

    for index in range(6):
        current = transform.copy()
        current[0, 3] = float(index * 10)
        base_gripper_transforms.append(current)

    try:
        check_motion_diversity(base_gripper_transforms)
    except CalibrationError as error:
        assert error.recoverable, "姿态退化应标记为可恢复"
    else:
        raise AssertionError("姿态退化未被检测")

    cam_board_transforms = [np.eye(4) for _ in base_gripper_transforms]

    try:
        solve_eye_to_hand(base_gripper_transforms, cam_board_transforms)
    except CalibrationError:
        pass
    else:
        raise AssertionError("退化数据不应求解成功")


# ============================================================
# 5. poses.json 契约
# ============================================================

def build_valid_payload() -> dict:
    """构造合法的 poses.json 结构。"""

    return {
        "version": 1,
        "created_at": "2026-11-07T12:00:00+08:00",
        "pattern_size": [9, 6],
        "euler_convention": "fixed_xyz_rpy",
        "robot_pose_units": {
            "position": "mm",
            "rotation": "deg",
        },
        "samples": [
            {
                "sample_id": 0,
                "timestamp": 1762500000.0,
                "robot_pose": {
                    "x_mm": 0.0,
                    "y_mm": 0.0,
                    "z_mm": 0.0,
                    "rx_deg": 0.0,
                    "ry_deg": 0.0,
                    "rz_deg": 0.0,
                },
                "image_path": "images/000.png",
            }
        ],
    }


def expect_calibration_error(payload: dict, message: str) -> None:
    """断言校验失败。"""

    try:
        validate_pose_payload(payload, (9, 6))
    except CalibrationError:
        return

    raise AssertionError(message)


def test_pose_payload_validation() -> None:
    """合法结构通过，非法字段被拒绝。"""

    valid_payload = build_valid_payload()
    samples = validate_pose_payload(valid_payload, (9, 6))
    assert len(samples) == 1, "合法样本解析失败"

    invalid_version = build_valid_payload()
    invalid_version["version"] = 2
    expect_calibration_error(invalid_version, "版本非法未被拒绝")

    invalid_pattern = build_valid_payload()
    invalid_pattern["pattern_size"] = [8, 6]
    expect_calibration_error(invalid_pattern, "棋盘格不一致未被拒绝")

    invalid_euler = build_valid_payload()
    invalid_euler["euler_convention"] = "zyx"
    expect_calibration_error(invalid_euler, "欧拉角约定非法未被拒绝")

    invalid_units = build_valid_payload()
    invalid_units["robot_pose_units"]["rotation"] = "rad"
    expect_calibration_error(invalid_units, "单位非法未被拒绝")

    empty_samples = build_valid_payload()
    empty_samples["samples"] = []
    expect_calibration_error(empty_samples, "空样本未被拒绝")

    try:
        load_pose_payload("/tmp/not_exists_handeye_poses.json")
    except CalibrationError as error:
        assert not error.recoverable, "文件缺失应为不可恢复"
    else:
        raise AssertionError("缺失位姿文件未被拒绝")


def test_resolve_image_path() -> None:
    """优先按 poses.json 所在目录解析图像。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        pose_file = root / "poses.json"
        pose_file.write_text("{}", encoding="utf-8")

        images_dir = root / "images"
        images_dir.mkdir()
        image_file = images_dir / "000.png"
        image_file.write_bytes(b"")

        resolved = resolve_image_path(
            str(pose_file),
            "",
            "images/000.png",
        )

        assert resolved.resolve() == image_file.resolve(), (
            "图像路径未按 poses.json 目录解析"
        )

        fallback_dir = root / "fallback"
        fallback_dir.mkdir()
        fallback_file = fallback_dir / "001.png"
        fallback_file.write_bytes(b"")

        resolved_fallback = resolve_image_path(
            str(pose_file),
            str(fallback_dir),
            "images/001.png",
        )

        assert resolved_fallback.resolve() == fallback_file.resolve(), (
            "图像路径回退失败"
        )


# ============================================================
# 6. 端到端合成标定
# ============================================================

def matrix_to_pose_dict(transform: np.ndarray) -> dict:
    """将 4x4 矩阵转换为固定轴 XYZ 位姿字典。"""

    rotation = transform[:3, :3]

    ry = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    rx = np.arctan2(rotation[2, 1], rotation[2, 2])
    rz = np.arctan2(rotation[1, 0], rotation[0, 0])

    return {
        "x_mm": float(transform[0, 3]),
        "y_mm": float(transform[1, 3]),
        "z_mm": float(transform[2, 3]),
        "rx_deg": float(np.degrees(rx)),
        "ry_deg": float(np.degrees(ry)),
        "rz_deg": float(np.degrees(rz)),
    }


def build_board_texture(
    square_columns: int,
    square_rows: int,
    square_px: int = 100,
    margin_px: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """生成棋盘格纹理及内角点像素坐标。"""

    texture_width = square_columns * square_px + 2 * margin_px
    texture_height = square_rows * square_px + 2 * margin_px

    texture = np.full(
        (texture_height, texture_width),
        255,
        dtype=np.uint8,
    )

    for row in range(square_rows):
        for column in range(square_columns):

            color = 0 if (row + column) % 2 == 0 else 255

            top_left = (
                margin_px + column * square_px,
                margin_px + row * square_px,
            )
            bottom_right = (
                top_left[0] + square_px,
                top_left[1] + square_px,
            )

            cv2.rectangle(
                texture,
                top_left,
                bottom_right,
                color,
                -1,
            )

    inner_corners = np.array(
        [
            [
                margin_px + (column + 1) * square_px,
                margin_px + (row + 1) * square_px,
            ]
            for row in range(square_rows - 1)
            for column in range(square_columns - 1)
        ],
        dtype=np.float32,
    )

    return texture, inner_corners


def test_end_to_end_synthetic() -> None:
    """合成棋盘格图像走完整标定流程，并校验输出矩阵。"""

    rng = np.random.default_rng(20261108)

    columns, rows = 9, 6
    square_size_mm = 25.0
    image_size = (1280, 720)

    camera_matrix = np.array(
        [
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)

    true_base_camera = random_transform(rng)
    true_gripper_board = random_transform(rng)

    object_points = create_object_points(
        (columns, rows),
        square_size_mm,
    )

    texture, texture_corners = build_board_texture(
        columns + 1,
        rows + 1,
    )

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        images_dir = root / "images"
        images_dir.mkdir()

        samples = []

        for index in range(12):

            rotation_vector = rng.uniform(-0.4, 0.4, 3)
            rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)

            cam_board = np.eye(4, dtype=np.float64)
            cam_board[:3, :3] = rotation_matrix
            cam_board[:3, 3] = [
                rng.uniform(-50.0, 50.0),
                rng.uniform(-30.0, 30.0),
                rng.uniform(450.0, 600.0),
            ]

            projected, _jacobian = cv2.projectPoints(
                object_points,
                rotation_vector,
                cam_board[:3, 3],
                camera_matrix,
                dist_coeffs,
            )
            projected = projected.reshape(-1, 2).astype(np.float32)

            homography, _mask = cv2.findHomography(
                texture_corners,
                projected,
                0,
            )

            image = cv2.warpPerspective(
                texture,
                homography,
                image_size,
                flags=cv2.INTER_LINEAR,
                borderValue=255,
            )

            image_path = images_dir / f"{index:03d}.png"
            cv2.imwrite(str(image_path), image)

            base_gripper = (
                true_base_camera
                @ cam_board
                @ np.linalg.inv(true_gripper_board)
            )

            samples.append(
                {
                    "sample_id": index,
                    "timestamp": 0.0,
                    "robot_pose": matrix_to_pose_dict(base_gripper),
                    "image_path": f"images/{index:03d}.png",
                }
            )

        payload = {
            "version": 1,
            "created_at": "2026-11-07T12:00:00+08:00",
            "pattern_size": [columns, rows],
            "euler_convention": "fixed_xyz_rpy",
            "robot_pose_units": {
                "position": "mm",
                "rotation": "deg",
            },
            "samples": samples,
        }

        pose_file = root / "poses.json"
        pose_file.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        intrinsics_path = root / "intrinsics.npz"
        np.savez_compressed(
            intrinsics_path,
            K=camera_matrix,
            D=dist_coeffs,
            rms=0.1,
            image_width=image_size[0],
            image_height=image_size[1],
        )

        output_path = root / "T_base_camera.npy"

        report = calibrate_handeye(
            pose_file=str(pose_file),
            image_dir=str(images_dir),
            intrinsics_path=str(intrinsics_path),
            output_path=str(output_path),
            pattern_size=(columns, rows),
            square_size_mm=square_size_mm,
        )

        assert report.passed, "端到端标定未通过质量门槛"
        assert output_path.is_file(), "T_base_camera.npy 未写入"

        estimated = np.load(output_path)

        assert_matrix_close(
            estimated,
            true_base_camera,
            1.0,
            "端到端手眼矩阵误差过大",
        )


# ============================================================
# 7. 主入口
# ============================================================

def main() -> int:
    """依次运行全部离线测试。"""

    tests = [
        test_eye_to_hand_roundtrip,
        test_euler_fixed_xyz,
        test_degenerate_motion,
        test_pose_payload_validation,
        test_resolve_image_path,
        test_end_to_end_synthetic,
    ]

    for test in tests:
        print(f"[RUN ] {test.__name__}")
        test()
        print(f"[PASS] {test.__name__}")

    print("-" * 60)
    print(f"全部通过：{len(tests)} 项")

    return 0


if __name__ == "__main__":
    sys.exit(main())
