"""标定结果质量验证与独立精度验证模块。

职责：
    1. 验证手眼标定矩阵 T_base_camera 对工作空间测试点的转换精度
    2. 验证相机内参在棋盘格图片上的重投影误差
    3. 输出结构化验证报告，供调用方判断 PASS/FAIL

说明：
    本模块只做验证，不做手眼标定求解，也不保存或修改 T_base_camera。
    坐标转换验证与重投影验证相互独立，分别对应 TransformVerifyReport
    与 ReprojectionReport。
"""

import os

import cv2
import numpy as np

from calibration.camera_calibration import (
    create_object_points,
    detect_chessboard,
    get_image_paths,
)
from calibration.handeye_calibration import load_intrinsics, validate_transform
from common.errors import CalibrationError
from common.types import ReprojectionReport, TransformVerifyReport


# ============================================================
# 1. 默认参数
# ============================================================

# 桌面点验证至少需要的测试点数量
MIN_POINTS = 6

# 默认平均/最大位置误差门槛，单位 mm
DEFAULT_MEAN_THRESHOLD_MM = 3.0
DEFAULT_MAX_THRESHOLD_MM = 5.0

# 重投影验证至少需要的有效视图数量
MIN_VIEWS = 10

# 默认棋盘格参数
DEFAULT_PATTERN_SIZE = (9, 6)
DEFAULT_SQUARE_SIZE_MM = 25.0

# 默认最大重投影误差门槛，单位 pixel
DEFAULT_MAX_THRESHOLD_PX = 2.0


# ============================================================
# 2. 输入校验
# ============================================================

def _validate_point_arrays(
    points_camera_mm: np.ndarray,
    points_robot_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """校验相机系与机器人系测试点数组。

    Args:
        points_camera_mm: 相机坐标系测试点，shape 应为 (N, 3)，单位 mm
        points_robot_mm: 机器人基座坐标系测试点，shape 应为 (N, 3)，单位 mm

    Returns:
        转换后的两个 float64 数组

    Raises:
        ValueError: 形状、点数、数值非法
    """

    camera_points = np.asarray(points_camera_mm, dtype=np.float64)
    robot_points = np.asarray(points_robot_mm, dtype=np.float64)

    if camera_points.ndim != 2 or camera_points.shape[1] != 3:
        raise ValueError(
            f"相机系测试点形状非法：{camera_points.shape}，预期 (N, 3)"
        )

    if robot_points.ndim != 2 or robot_points.shape[1] != 3:
        raise ValueError(
            f"机器人系测试点形状非法：{robot_points.shape}，预期 (N, 3)"
        )

    if camera_points.shape != robot_points.shape:
        raise ValueError(
            "两组测试点数量或形状不一致："
            f"{camera_points.shape} 与 {robot_points.shape}"
        )

    if camera_points.shape[0] < MIN_POINTS:
        raise ValueError(
            f"测试点数量不足：{camera_points.shape[0]}，"
            f"至少需要 {MIN_POINTS}"
        )

    if not np.all(np.isfinite(camera_points)):
        raise ValueError("相机系测试点包含 NaN 或 Inf")

    if not np.all(np.isfinite(robot_points)):
        raise ValueError("机器人系测试点包含 NaN 或 Inf")

    return camera_points, robot_points


def _validate_position_thresholds(
    mean_threshold_mm: float,
    max_threshold_mm: float,
) -> None:
    """校验位置误差门槛。

    Args:
        mean_threshold_mm: 平均误差门槛，mm
        max_threshold_mm: 最大误差门槛，mm

    Raises:
        ValueError: 门槛非正或最大门槛小于平均门槛
    """

    if not np.isfinite(mean_threshold_mm) or mean_threshold_mm <= 0.0:
        raise ValueError(
            f"平均误差门槛必须为有限正数：{mean_threshold_mm}"
        )

    if not np.isfinite(max_threshold_mm) or max_threshold_mm <= 0.0:
        raise ValueError(
            f"最大误差门槛必须为有限正数：{max_threshold_mm}"
        )

    if max_threshold_mm < mean_threshold_mm:
        raise ValueError(
            "最大误差门槛不能小于平均误差门槛："
            f"{max_threshold_mm} < {mean_threshold_mm}"
        )


def _validate_pixel_threshold(max_threshold_px: float) -> None:
    """校验像素误差门槛。

    Args:
        max_threshold_px: 最大重投影误差门槛，pixel

    Raises:
        ValueError: 门槛非正
    """

    if not np.isfinite(max_threshold_px) or max_threshold_px <= 0.0:
        raise ValueError(
            f"最大重投影误差门槛必须为有限正数：{max_threshold_px}"
        )


# ============================================================
# 3. 坐标变换验证
# ============================================================

def _transform_points(
    transform: np.ndarray,
    points_camera_mm: np.ndarray,
) -> np.ndarray:
    """将相机坐标系点转换到机器人基座坐标系。

    Args:
        transform: 4x4 T_base_camera
        points_camera_mm: (N, 3) 相机系点，mm

    Returns:
        (N, 3) 基座坐标系点，mm
    """

    homogeneous = np.hstack(
        [points_camera_mm, np.ones((points_camera_mm.shape[0], 1))]
    )

    transformed = (transform @ homogeneous.T).T

    return transformed[:, :3]


def _build_transform_reason(
    passed: bool,
    mean_error_mm: float,
    max_error_mm: float,
    mean_threshold_mm: float,
    max_threshold_mm: float,
) -> str:
    """生成坐标变换验证说明。

    Returns:
        PASS/FAIL 说明字符串
    """

    if passed:
        return (
            f"PASS：平均误差 {mean_error_mm:.4f} mm，"
            f"最大误差 {max_error_mm:.4f} mm"
        )

    failures = []

    if mean_error_mm > mean_threshold_mm:
        failures.append(
            f"平均误差 {mean_error_mm:.4f} mm "
            f"超过门槛 {mean_threshold_mm:.4f} mm"
        )

    if max_error_mm > max_threshold_mm:
        failures.append(
            f"最大误差 {max_error_mm:.4f} mm "
            f"超过门槛 {max_threshold_mm:.4f} mm"
        )

    return "FAIL：" + "；".join(failures)


def verify_transform(
    T_base_camera: np.ndarray,
    points_camera_mm: np.ndarray,
    points_robot_mm: np.ndarray,
    mean_threshold_mm: float = DEFAULT_MEAN_THRESHOLD_MM,
    max_threshold_mm: float = DEFAULT_MAX_THRESHOLD_MM,
) -> TransformVerifyReport:
    """验证 T_base_camera 对工作空间测试点的转换精度。

    Args:
        T_base_camera: 4x4 手眼变换矩阵，方向为相机 -> 基座，平移单位 mm
        points_camera_mm: 相机坐标系测试点，shape (N, 3)，单位 mm
        points_robot_mm: 独立测量的机器人基座坐标系参考点，
            shape (N, 3)，单位 mm
        mean_threshold_mm: 平均误差门槛，mm
        max_threshold_mm: 最大误差门槛，mm

    Returns:
        TransformVerifyReport：逐点误差、平均误差、最大误差与 PASS/FAIL

    Raises:
        CalibrationError: 变换矩阵非法
        ValueError: 点形状、点数、数值或门槛非法
    """

    transform = np.asarray(T_base_camera, dtype=np.float64)
    validate_transform(transform, name="T_base_camera")

    camera_points, robot_points = _validate_point_arrays(
        points_camera_mm,
        points_robot_mm,
    )
    _validate_position_thresholds(mean_threshold_mm, max_threshold_mm)

    transformed_points = _transform_points(transform, camera_points)

    errors = np.linalg.norm(
        transformed_points - robot_points,
        axis=1,
    )

    mean_error_mm = float(np.mean(errors))
    max_error_mm = float(np.max(errors))

    passed = (
        mean_error_mm <= mean_threshold_mm
        and max_error_mm <= max_threshold_mm
    )

    reason = _build_transform_reason(
        passed,
        mean_error_mm,
        max_error_mm,
        mean_threshold_mm,
        max_threshold_mm,
    )

    return TransformVerifyReport(
        n_points=int(camera_points.shape[0]),
        per_point_errors_mm=tuple(float(error) for error in errors),
        mean_error_mm=mean_error_mm,
        max_error_mm=max_error_mm,
        passed=passed,
        reason=reason,
    )


# ============================================================
# 4. 重投影验证
# ============================================================

def _load_image_paths(image_dir: str) -> list[str]:
    """读取图片目录中的标定图片路径。

    Args:
        image_dir: 图片目录

    Returns:
        按文件名排序的图片路径列表

    Raises:
        CalibrationError: 目录不存在或没有图片
    """

    if not os.path.isdir(image_dir):
        raise CalibrationError(
            f"图片目录不存在：{image_dir}",
            recoverable=False,
        )

    try:
        return get_image_paths(image_dir)
    except FileNotFoundError as error:
        raise CalibrationError(
            str(error),
            recoverable=False,
        ) from error


def _check_image_resolution(
    image: np.ndarray,
    metadata: dict,
    image_path: str,
) -> None:
    """校验图片分辨率与相机内参元数据一致。

    Args:
        image: BGR 图像
        metadata: load_intrinsics() 返回的元数据
        image_path: 当前图片路径，用于错误信息

    Raises:
        CalibrationError: 分辨率与内参记录不一致
    """

    expected_width = metadata.get("image_width")
    expected_height = metadata.get("image_height")

    if expected_width is None or expected_height is None:
        return

    height, width = image.shape[:2]

    if (width, height) != (expected_width, expected_height):
        raise CalibrationError(
            f"图像分辨率与内参不一致：{image_path} 为 "
            f"{(width, height)}，预期 "
            f"{(expected_width, expected_height)}",
            recoverable=False,
        )


def _estimate_view_reprojection(
    image: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[float, float, np.ndarray] | None:
    """计算单张棋盘格图片的重投影误差。

    Args:
        image: BGR 图像
        object_points: 棋盘格三维物点，shape (N, 3)
        camera_matrix: 3x3 相机内参 K
        dist_coeffs: 畸变系数 D
        pattern_size: 棋盘格内角点数量 (列, 行)

    Returns:
        (该视图 RMS 误差, 该视图最大角点误差, 逐角点误差)；
        检测或求解失败返回 None
    """

    success, corners, _gray = detect_chessboard(image, pattern_size)

    if not success:
        return None

    try:
        solved, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not solved:
            return None

        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            rotation_vector,
            translation_vector,
        )

        projected_points, _jacobian = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )

    except cv2.error:
        return None

    projected_points = projected_points.reshape(-1, 2)
    actual_points = corners.reshape(-1, 2)

    point_errors = np.linalg.norm(
        projected_points - actual_points,
        axis=1,
    )

    view_rms_px = float(np.sqrt(np.mean(np.square(point_errors))))
    view_max_px = float(np.max(point_errors))

    return view_rms_px, view_max_px, point_errors


def _build_reprojection_reason(
    passed: bool,
    n_views: int,
    rms_px: float,
    max_error_px: float,
    max_threshold_px: float,
    skipped_count: int,
) -> str:
    """生成重投影验证说明。

    Returns:
        PASS/FAIL 说明字符串
    """

    summary = (
        f"视图 {n_views}，跳过 {skipped_count}，"
        f"RMS {rms_px:.4f} px，最大误差 {max_error_px:.4f} px"
    )

    if passed:
        return f"PASS：{summary}"

    return (
        f"FAIL：最大误差 {max_error_px:.4f} px "
        f"超过门槛 {max_threshold_px:.4f} px；{summary}"
    )


def verify_reprojection(
    intrinsics_path: str,
    image_dir: str,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
    max_threshold_px: float = DEFAULT_MAX_THRESHOLD_PX,
) -> ReprojectionReport:
    """验证相机内参在棋盘格图片上的重投影误差。

    Args:
        intrinsics_path: camera_intrinsics.npz 路径
        image_dir: 棋盘格图片目录
        pattern_size: 棋盘格内角点数量 (列, 行)
        square_size_mm: 方格边长，mm
        max_threshold_px: 最大重投影误差门槛，pixel

    Returns:
        ReprojectionReport：整体 RMS、最大误差、逐视图 RMS 与 PASS/FAIL

    Raises:
        CalibrationError: 内参文件缺失/非法、图片目录非法、
            分辨率不一致或有效视图不足
        ValueError: 门槛或棋盘格参数非法
    """

    _validate_pixel_threshold(max_threshold_px)

    camera_matrix, dist_coeffs, metadata = load_intrinsics(intrinsics_path)

    image_paths = _load_image_paths(image_dir)
    object_points = create_object_points(pattern_size, square_size_mm)

    per_view_errors_px = []
    all_point_errors = []
    max_error_px = 0.0
    skipped_count = 0

    for image_path in image_paths:

        image = cv2.imread(image_path)

        if image is None:
            skipped_count += 1
            continue

        _check_image_resolution(image, metadata, image_path)

        estimate = _estimate_view_reprojection(
            image,
            object_points,
            camera_matrix,
            dist_coeffs,
            pattern_size,
        )

        if estimate is None:
            skipped_count += 1
            continue

        view_rms_px, view_max_px, point_errors = estimate

        per_view_errors_px.append(view_rms_px)
        all_point_errors.append(point_errors)
        max_error_px = max(max_error_px, view_max_px)

    n_views = len(per_view_errors_px)

    if n_views < MIN_VIEWS:
        raise CalibrationError(
            f"有效视图不足：{n_views}，至少需要 {MIN_VIEWS}；"
            f"跳过 {skipped_count} 张",
            recoverable=True,
        )

    concatenated_errors = np.concatenate(all_point_errors)
    rms_px = float(np.sqrt(np.mean(np.square(concatenated_errors))))

    passed = max_error_px <= max_threshold_px

    reason = _build_reprojection_reason(
        passed,
        n_views,
        rms_px,
        max_error_px,
        max_threshold_px,
        skipped_count,
    )

    return ReprojectionReport(
        n_views=n_views,
        rms_px=rms_px,
        max_error_px=max_error_px,
        per_view_errors_px=tuple(
            float(error) for error in per_view_errors_px
        ),
        passed=passed,
        reason=reason,
    )
