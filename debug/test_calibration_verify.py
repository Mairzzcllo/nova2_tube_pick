"""标定验证模块离线测试。

覆盖：
    1. verify_transform 的正常转换、阈值判定与异常输入
    2. verify_reprojection 的合成棋盘格图片验证
    3. 内参/图片目录/有效视图不足等失败路径

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_calibration_verify.py
"""

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

from calibration.calibration_verify import (
    MIN_POINTS,
    MIN_VIEWS,
    verify_reprojection,
    verify_transform,
)
from calibration.camera_calibration import create_object_points
from common.errors import CalibrationError


# ============================================================
# 1. 测试工具
# ============================================================

def make_transform(rng: np.random.Generator) -> np.ndarray:
    """生成随机 4x4 刚体变换。"""

    rotation_vector = rng.uniform(-0.5, 0.5, 3)
    rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = rng.uniform(-100.0, 100.0, 3)

    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """将 Nx3 点按 4x4 变换转换。"""

    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ homogeneous.T).T[:, :3]


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


def write_intrinsics(
    intrinsics_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    """保存测试用相机内参文件。"""

    np.savez_compressed(
        intrinsics_path,
        K=camera_matrix,
        D=dist_coeffs,
        rms=0.1,
        image_width=image_size[0],
        image_height=image_size[1],
    )


def build_synthetic_dataset(
    root: Path,
    view_count: int,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    image_size: tuple[int, int] = (1280, 720),
    seed: int = 20261109,
) -> tuple[Path, Path]:
    """生成合成棋盘格图片与内参文件。"""

    rng = np.random.default_rng(seed)

    columns, rows = pattern_size

    camera_matrix = np.array(
        [
            [1000.0, 0.0, image_size[0] / 2.0],
            [0.0, 1000.0, image_size[1] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((1, 5), dtype=np.float64)

    object_points = create_object_points(pattern_size, square_size_mm)

    texture, texture_corners = build_board_texture(
        columns + 1,
        rows + 1,
    )

    image_dir = root / "images"
    image_dir.mkdir()

    for index in range(view_count):

        rotation_vector = rng.uniform(-0.35, 0.35, 3)
        translation_vector = np.array(
            [
                rng.uniform(-50.0, 50.0),
                rng.uniform(-30.0, 30.0),
                rng.uniform(450.0, 600.0),
            ]
        )

        projected, _jacobian = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
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

        cv2.imwrite(str(image_dir / f"{index:03d}.png"), image)

    intrinsics_path = root / "intrinsics.npz"
    write_intrinsics(
        intrinsics_path,
        camera_matrix,
        dist_coeffs,
        image_size,
    )

    return intrinsics_path, image_dir


# ============================================================
# 2. verify_transform 测试
# ============================================================

def test_transform_pass() -> None:
    """无噪声测试点应精确通过。"""

    rng = np.random.default_rng(20261110)

    transform = make_transform(rng)
    points_camera = rng.uniform(-200.0, 200.0, (8, 3))
    points_robot = transform_points(transform, points_camera)

    report = verify_transform(transform, points_camera, points_robot)

    assert report.passed, report.reason
    assert report.n_points == 8
    assert report.mean_error_mm < 1e-9
    assert report.max_error_mm < 1e-9
    assert len(report.per_point_errors_mm) == 8
    assert report.reason.startswith("PASS")


def test_transform_max_failure() -> None:
    """单点大误差应触发最大误差失败。"""

    rng = np.random.default_rng(20261111)

    transform = make_transform(rng)
    points_camera = rng.uniform(-200.0, 200.0, (20, 3))
    points_robot = transform_points(transform, points_camera)

    points_robot[0] += np.array([10.0, 0.0, 0.0])

    report = verify_transform(transform, points_camera, points_robot)

    assert not report.passed
    assert report.max_error_mm > 5.0
    assert report.mean_error_mm < 3.0
    assert "最大误差" in report.reason


def test_transform_mean_failure() -> None:
    """均匀中等误差应触发平均误差失败。"""

    rng = np.random.default_rng(20261112)

    transform = make_transform(rng)
    points_camera = rng.uniform(-200.0, 200.0, (20, 3))
    points_robot = transform_points(transform, points_camera)

    points_robot += np.array([4.0, 0.0, 0.0])

    report = verify_transform(transform, points_camera, points_robot)

    assert not report.passed
    assert report.mean_error_mm > 3.0
    assert report.max_error_mm <= 5.0
    assert "平均误差" in report.reason


def test_transform_invalid_inputs() -> None:
    """点数、形状、数值、矩阵与门槛异常应被拒绝。"""

    rng = np.random.default_rng(20261113)

    transform = make_transform(rng)
    points_camera = rng.uniform(-100.0, 100.0, (8, 3))
    points_robot = transform_points(transform, points_camera)

    try:
        verify_transform(transform, points_camera, points_robot[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("点数不一致未被拒绝")

    try:
        verify_transform(transform, points_camera[:, :2], points_robot)
    except ValueError:
        pass
    else:
        raise AssertionError("点形状非法未被拒绝")

    try:
        verify_transform(
            transform,
            points_camera[: MIN_POINTS - 1],
            points_robot[: MIN_POINTS - 1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("测试点数量不足未被拒绝")

    invalid_points = points_camera.copy()
    invalid_points[0, 0] = np.nan

    try:
        verify_transform(transform, invalid_points, points_robot)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN 测试点未被拒绝")

    invalid_transform = transform.copy()
    invalid_transform[3, 0] = 1.0

    try:
        verify_transform(invalid_transform, points_camera, points_robot)
    except CalibrationError as error:
        assert not error.recoverable
    else:
        raise AssertionError("非法变换矩阵未被拒绝")

    try:
        verify_transform(
            transform,
            points_camera,
            points_robot,
            mean_threshold_mm=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("非法平均门槛未被拒绝")

    try:
        verify_transform(
            transform,
            points_camera,
            points_robot,
            mean_threshold_mm=4.0,
            max_threshold_mm=3.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("最大门槛小于平均门槛未被拒绝")


# ============================================================
# 3. verify_reprojection 测试
# ============================================================

def test_reprojection_pass() -> None:
    """合成棋盘格图片应通过重投影验证。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        view_count = MIN_VIEWS + 5
        intrinsics_path, image_dir = build_synthetic_dataset(
            root,
            view_count=view_count,
        )

        report = verify_reprojection(
            str(intrinsics_path),
            str(image_dir),
        )

        assert report.passed, report.reason
        assert report.n_views >= MIN_VIEWS
        assert report.n_views <= view_count
        assert len(report.per_view_errors_px) == report.n_views
        assert report.max_error_px <= 2.0
        assert report.rms_px <= report.max_error_px
        assert report.reason.startswith("PASS")


def test_reprojection_threshold_failure() -> None:
    """极小门槛应触发重投影失败。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        intrinsics_path, image_dir = build_synthetic_dataset(
            root,
            view_count=MIN_VIEWS + 5,
        )

        report = verify_reprojection(
            str(intrinsics_path),
            str(image_dir),
            max_threshold_px=0.0001,
        )

        assert not report.passed
        assert report.max_error_px > 0.0001
        assert report.reason.startswith("FAIL")


def test_reprojection_invalid_threshold() -> None:
    """非正重投影门槛应抛 ValueError。"""

    try:
        verify_reprojection(
            "not_used.npz",
            "not_used_dir",
            max_threshold_px=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("非正重投影门槛未被拒绝")


def test_reprojection_missing_intrinsics() -> None:
    """内参文件缺失应抛不可恢复异常。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        _intrinsics_path, image_dir = build_synthetic_dataset(
            root,
            view_count=MIN_VIEWS,
        )

        try:
            verify_reprojection(
                str(root / "not_exists.npz"),
                str(image_dir),
            )
        except CalibrationError as error:
            assert not error.recoverable
        else:
            raise AssertionError("缺失内参文件未被拒绝")


def test_reprojection_empty_directory() -> None:
    """空图片目录应抛不可恢复异常。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        intrinsics_path, image_dir = build_synthetic_dataset(
            root,
            view_count=MIN_VIEWS,
        )

        empty_dir = root / "empty"
        empty_dir.mkdir()

        try:
            verify_reprojection(
                str(intrinsics_path),
                str(empty_dir),
            )
        except CalibrationError as error:
            assert not error.recoverable
        else:
            raise AssertionError("空图片目录未被拒绝")


def test_reprojection_insufficient_views() -> None:
    """有效视图不足应抛可恢复异常。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)
        intrinsics_path, image_dir = build_synthetic_dataset(
            root,
            view_count=MIN_VIEWS - 1,
        )

        try:
            verify_reprojection(
                str(intrinsics_path),
                str(image_dir),
            )
        except CalibrationError as error:
            assert error.recoverable
        else:
            raise AssertionError("有效视图不足未被拒绝")


def test_reprojection_detection_failure() -> None:
    """全部图片检测失败应抛可恢复异常。"""

    with tempfile.TemporaryDirectory() as temporary_dir:

        root = Path(temporary_dir)

        camera_matrix = np.array(
            [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((1, 5), dtype=np.float64)
        image_size = (1280, 720)

        intrinsics_path = root / "intrinsics.npz"
        write_intrinsics(
            intrinsics_path,
            camera_matrix,
            dist_coeffs,
            image_size,
        )

        image_dir = root / "images"
        image_dir.mkdir()

        blank = np.full(
            (image_size[1], image_size[0], 3),
            255,
            dtype=np.uint8,
        )

        for index in range(MIN_VIEWS):
            cv2.imwrite(str(image_dir / f"{index:03d}.png"), blank)

        try:
            verify_reprojection(
                str(intrinsics_path),
                str(image_dir),
            )
        except CalibrationError as error:
            assert error.recoverable
        else:
            raise AssertionError("棋盘格检测失败未被拒绝")


# ============================================================
# 4. 主入口
# ============================================================

def main() -> int:
    """依次运行全部离线测试。"""

    tests = [
        test_transform_pass,
        test_transform_max_failure,
        test_transform_mean_failure,
        test_transform_invalid_inputs,
        test_reprojection_pass,
        test_reprojection_threshold_failure,
        test_reprojection_invalid_threshold,
        test_reprojection_missing_intrinsics,
        test_reprojection_empty_directory,
        test_reprojection_insufficient_views,
        test_reprojection_detection_failure,
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
