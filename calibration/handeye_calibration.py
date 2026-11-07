"""手眼标定模块（eye-to-hand）。

功能：
    1. 读取 pose_recorder 生成的 poses.json（机器人末端位姿 + 棋盘格图像）
    2. 读取相机内参 K/D
    3. 逐图检测棋盘格并用 solvePnP 求 T_cam_board
    4. 将机器人位姿转 T_base_gripper，并求逆得到 T_gripper_base
    5. 用 cv2.calibrateHandEye 的 eye-to-hand 形式求解 T_base_camera
    6. 评估刚性一致性与重投影误差，通过后原子保存 T_base_camera.npy

坐标与单位：
    相机固定在工作台，棋盘格随机器人末端运动。
    T_base_camera 方向为相机 -> 基座，旋转在左上 3x3，平移在右上 3x1，单位 mm。
    机器人位姿为固定轴 XYZ（roll-pitch-yaw），位置 mm、姿态 deg。

说明：
    eye-to-hand 与 eye-in-hand 的 AX=XB 输入不同：
    eye-to-hand 需要把机器人位姿取逆后作为 gripper2base 传入
    cv2.calibrateHandEye，返回值即 T_base_camera，无需再做二次合成。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 兼容 python -m 与直接执行文件两种方式，确保能导入仓库内模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibration.camera_calibration import create_object_points, detect_chessboard
from common.errors import CalibrationError, Nova2ProjectError
from common.types import HandEyeReport


# ============================================================
# 1. 默认标定参数
# ============================================================

# 默认位姿文件（pose_recorder 的输出）
DEFAULT_POSE_FILE = "data/handeye/poses.json"

# 默认图像目录（仅作为 image_path 解析失败时的回退）
DEFAULT_IMAGE_DIR = "data/handeye/images"

# 默认相机内参文件
DEFAULT_INTRINSICS_PATH = "config/camera_intrinsics.npz"

# 默认手眼矩阵输出路径
DEFAULT_OUTPUT_PATH = "config/T_base_camera.npy"

# 棋盘格内角点数量：(列数, 行数)
DEFAULT_PATTERN_SIZE = (9, 6)

# 棋盘格方格边长，mm
DEFAULT_SQUARE_SIZE_MM = 25.0

# OpenCV 求解要求的最少姿态数
MIN_SAMPLES = 3

# 建议的姿态数，低于该值仅告警
RECOMMENDED_SAMPLES = 10

# 姿态多样性检查：最大相对旋转角下限，deg
MIN_RELATIVE_ROTATION_DEG = 5.0

# 姿态多样性检查：旋转轴第二/第一奇异值比例下限
MIN_AXIS_DIVERSITY = 0.1

# 刚性一致性门槛：棋盘格在末端坐标系下平移标准差，mm
TRANSLATION_STD_MAX_MM = 2.0

# 刚性一致性门槛：棋盘格在末端坐标系下旋转标准差，deg
ROTATION_STD_MAX_DEG = 0.5

# 重投影 RMS 门槛，pixel
REPROJECTION_MAX_PX = 2.0

# OpenCV 手眼求解方法
HANDEYE_METHOD = cv2.CALIB_HAND_EYE_TSAI

# 机器人位姿字典必需字段
POSE_KEYS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "rx_deg",
    "ry_deg",
    "rz_deg",
)

# 运行日志目录（锚定仓库根目录，避免受当前工作目录影响）
LOG_DIR = os.path.join(_ROOT, "data", "logs")


# ============================================================
# 2. 日志
# ============================================================

def setup_logger() -> logging.Logger:
    """创建手眼标定日志记录器。

    Returns:
        同时输出到控制台与 data/logs/YYYYmmdd_HHMMSS_handeye_calibration.log 的 Logger
    """

    logger = logging.getLogger("handeye_calibration")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(
        LOG_DIR,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_handeye_calibration.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# 3. 位姿与矩阵工具
# ============================================================

def euler_fixed_xyz_to_matrix(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """将固定轴 XYZ（roll-pitch-yaw）欧拉角转换为旋转矩阵。

    Args:
        rx_deg: 绕固定 X 轴旋转，deg
        ry_deg: 绕固定 Y 轴旋转，deg
        rz_deg: 绕固定 Z 轴旋转，deg

    Returns:
        3x3 旋转矩阵，R = Rz(rz) * Ry(ry) * Rx(rx)

    Raises:
        CalibrationError: 角度非法
    """

    angles = np.asarray([rx_deg, ry_deg, rz_deg], dtype=np.float64)

    if not np.all(np.isfinite(angles)):
        raise CalibrationError(
            "机器人姿态包含非有限值",
            recoverable=False,
        )

    rx, ry, rz = np.deg2rad(angles)

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x],
            [0.0, sin_x, cos_x],
        ],
        dtype=np.float64,
    )

    rotation_y = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float64,
    )

    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0.0],
            [sin_z, cos_z, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rotation_z @ rotation_y @ rotation_x


def pose_dict_to_matrix(pose: dict) -> np.ndarray:
    """将 poses.json 中的机器人位姿转换为 4x4 齐次矩阵。

    Args:
        pose: 包含 x_mm/y_mm/z_mm/rx_deg/ry_deg/rz_deg 的字典

    Returns:
        T_base_gripper，4x4，平移 mm

    Raises:
        CalibrationError: 位姿字段缺失或数值非法
    """

    if not isinstance(pose, dict):
        raise CalibrationError(
            f"机器人位姿类型非法：{type(pose).__name__}",
            recoverable=False,
        )

    missing = [key for key in POSE_KEYS if key not in pose]

    if missing:
        raise CalibrationError(
            f"机器人位姿缺少字段：{missing}",
            recoverable=False,
        )

    try:
        values = [float(pose[key]) for key in POSE_KEYS]
    except (TypeError, ValueError) as error:
        raise CalibrationError(
            f"机器人位姿数值非法：{error}",
            recoverable=False,
        )

    if not np.all(np.isfinite(values)):
        raise CalibrationError(
            "机器人位姿包含非有限值",
            recoverable=False,
        )

    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = values

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_fixed_xyz_to_matrix(rx_deg, ry_deg, rz_deg)
    transform[:3, 3] = [x_mm, y_mm, z_mm]

    return transform


def transform_inverse(transform: np.ndarray) -> np.ndarray:
    """求刚体变换的逆矩阵。

    Args:
        transform: 4x4 齐次变换

    Returns:
        4x4 逆变换

    Raises:
        CalibrationError: 矩阵形状或数值非法
    """

    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise CalibrationError(
            "变换矩阵非法，无法求逆",
            recoverable=False,
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation

    return inverse


def validate_transform(transform: np.ndarray, name: str = "变换矩阵") -> None:
    """校验 4x4 刚体变换矩阵。

    Args:
        transform: 待校验矩阵
        name: 日志中使用的名称

    Raises:
        CalibrationError: 形状、末行、正交性或行列式非法
    """

    if transform.shape != (4, 4):
        raise CalibrationError(
            f"{name}形状非法：{transform.shape}，预期 (4, 4)",
            recoverable=False,
        )

    if not np.all(np.isfinite(transform)):
        raise CalibrationError(
            f"{name}包含非有限值",
            recoverable=False,
        )

    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise CalibrationError(
            f"{name}最后一行非法：{transform[3]}",
            recoverable=False,
        )

    rotation = transform[:3, :3]
    orthogonality_error = np.max(
        np.abs(rotation.T @ rotation - np.eye(3))
    )
    determinant = float(np.linalg.det(rotation))

    if orthogonality_error > 1e-6 or abs(determinant - 1.0) > 1e-6:
        raise CalibrationError(
            f"{name}旋转部分非法：正交误差={orthogonality_error:.3e}，"
            f"det={determinant:.6f}",
            recoverable=False,
        )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """计算旋转矩阵对应的旋转角。

    Args:
        rotation: 3x3 旋转矩阵

    Returns:
        旋转角，deg
    """

    rotation_vector, _ = cv2.Rodrigues(rotation)
    return float(np.degrees(np.linalg.norm(rotation_vector)))


def average_rotation_matrices(rotations: list[np.ndarray]) -> np.ndarray:
    """用 SVD 求一组旋转矩阵的平均旋转。

    Args:
        rotations: 3x3 旋转矩阵列表

    Returns:
        3x3 平均旋转矩阵

    Raises:
        ValueError: 列表为空
    """

    if not rotations:
        raise ValueError("旋转矩阵列表不能为空")

    matrix_sum = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    left, _singular, right_t = np.linalg.svd(matrix_sum)

    correction = np.diag(
        [1.0, 1.0, np.linalg.det(left @ right_t)]
    )

    return left @ correction @ right_t


# ============================================================
# 4. 标定数据加载与校验
# ============================================================

def load_intrinsics(
    intrinsics_path: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取相机内参文件。

    Args:
        intrinsics_path: camera_intrinsics.npz 路径

    Returns:
        (K, D, metadata)，metadata 可能包含 rms/image_width/image_height 等

    Raises:
        CalibrationError: 文件缺失、字段缺失或数值非法
    """

    if not os.path.isfile(intrinsics_path):
        raise CalibrationError(
            f"相机内参文件不存在：{intrinsics_path}",
            recoverable=False,
        )

    try:
        with np.load(intrinsics_path) as data:
            for key in ("K", "D", "rms"):
                if key not in data:
                    raise CalibrationError(
                        f"相机内参文件缺少字段：{key}",
                        recoverable=False,
                    )

            camera_matrix = np.asarray(data["K"], dtype=np.float64)
            dist_coeffs = np.asarray(data["D"], dtype=np.float64)
            rms_values = np.asarray(data["rms"], dtype=np.float64).reshape(-1)

            if rms_values.size != 1:
                raise CalibrationError(
                    f"相机内参 rms 字段非法：{rms_values}",
                    recoverable=False,
                )

            metadata = {
                "rms": float(rms_values[0]),
            }

            for key in ("image_width", "image_height"):
                if key in data:
                    metadata[key] = int(np.asarray(data[key]).reshape(-1)[0])

    except (OSError, ValueError, EOFError) as error:
        raise CalibrationError(
            f"相机内参文件读取失败：{intrinsics_path}：{error}",
            recoverable=False,
        )

    if camera_matrix.shape != (3, 3):
        raise CalibrationError(
            f"相机内参 K 形状非法：{camera_matrix.shape}",
            recoverable=False,
        )

    if dist_coeffs.ndim > 2 or dist_coeffs.size < 4:
        raise CalibrationError(
            f"相机畸变 D 形状非法：{dist_coeffs.shape}",
            recoverable=False,
        )

    if not np.all(np.isfinite(camera_matrix)):
        raise CalibrationError(
            "相机内参 K 包含非有限值",
            recoverable=False,
        )

    if not np.all(np.isfinite(dist_coeffs)):
        raise CalibrationError(
            "相机畸变 D 包含非有限值",
            recoverable=False,
        )

    return camera_matrix, dist_coeffs, metadata


def load_pose_payload(pose_file: str) -> dict:
    """读取 poses.json。

    Args:
        pose_file: poses.json 路径

    Returns:
        JSON 解析后的字典

    Raises:
        CalibrationError: 文件缺失、JSON 非法或顶层不是字典
    """

    if not os.path.isfile(pose_file):
        raise CalibrationError(
            f"位姿文件不存在：{pose_file}",
            recoverable=False,
        )

    try:
        with open(pose_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(
            f"位姿文件读取失败：{pose_file}：{error}",
            recoverable=False,
        )

    if not isinstance(payload, dict):
        raise CalibrationError(
            "位姿文件顶层结构必须是对象",
            recoverable=False,
        )

    return payload


def validate_pose_payload(
    payload: dict,
    pattern_size: tuple[int, int],
) -> list[dict]:
    """校验 poses.json 契约并返回样本列表。

    Args:
        payload: poses.json 解析结果
        pattern_size: 本次标定使用的棋盘格内角点数量

    Returns:
        samples 列表

    Raises:
        CalibrationError: 版本、棋盘格、欧拉角、单位或样本非法
    """

    if payload.get("version") != 1:
        raise CalibrationError(
            f"位姿文件版本不支持：{payload.get('version')}",
            recoverable=False,
        )

    file_pattern = payload.get("pattern_size")

    if not isinstance(file_pattern, (list, tuple)) or len(file_pattern) != 2:
        raise CalibrationError(
            f"位姿文件 pattern_size 非法：{file_pattern}",
            recoverable=False,
        )

    try:
        parsed_pattern = tuple(int(value) for value in file_pattern)
    except (TypeError, ValueError) as error:
        raise CalibrationError(
            f"位姿文件 pattern_size 非法：{file_pattern}：{error}",
            recoverable=False,
        )

    if parsed_pattern != tuple(pattern_size):
        raise CalibrationError(
            f"棋盘格参数不一致：文件 {file_pattern}，"
            f"本次 {tuple(pattern_size)}",
            recoverable=False,
        )

    if payload.get("euler_convention") != "fixed_xyz_rpy":
        raise CalibrationError(
            "位姿文件欧拉角约定不是 fixed_xyz_rpy",
            recoverable=False,
        )

    units = payload.get("robot_pose_units", {})

    if not isinstance(units, dict):
        raise CalibrationError(
            "位姿文件 robot_pose_units 非法",
            recoverable=False,
        )

    if units.get("position") != "mm" or units.get("rotation") != "deg":
        raise CalibrationError(
            "位姿文件单位不是 mm/deg",
            recoverable=False,
        )

    samples = payload.get("samples")

    if not isinstance(samples, list) or not samples:
        raise CalibrationError(
            "位姿文件 samples 为空或不是列表",
            recoverable=False,
        )

    return samples


def resolve_image_path(
    pose_file: str,
    image_dir: str,
    image_path: str,
) -> Path:
    """解析样本图像路径。

    优先相对 poses.json 所在目录解析；找不到时回退到 image_dir。

    Args:
        pose_file: poses.json 路径
        image_dir: 图像目录（回退）
        image_path: poses.json 中记录的相对路径

    Returns:
        存在的图像路径

    Raises:
        CalibrationError: 图像路径非法或文件不存在
    """

    if not isinstance(image_path, str) or not image_path:
        raise CalibrationError(
            "样本 image_path 非法",
            recoverable=False,
        )

    relative_path = Path(image_path)

    if relative_path.is_absolute():
        candidates = [relative_path]
    else:
        candidates = [Path(pose_file).resolve().parent / relative_path]

        if image_dir:
            candidates.append(Path(image_dir) / relative_path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    attempted = "，".join(str(candidate) for candidate in candidates)

    raise CalibrationError(
        f"标定图像不存在：{image_path}（已尝试：{attempted}）",
        recoverable=False,
    )


# ============================================================
# 5. 棋盘格位姿估计
# ============================================================

def estimate_board_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> tuple[np.ndarray, float] | None:
    """从单张图像估计棋盘格在相机中的位姿。

    Args:
        image: BGR 图像
        camera_matrix: 3x3 相机内参 K
        dist_coeffs: 畸变系数 D
        pattern_size: 棋盘格内角点数量 (列, 行)
        square_size_mm: 方格边长，mm

    Returns:
        (T_cam_board, reprojection_rms_px)；检测或求解失败返回 None
    """

    if image is None or getattr(image, "size", 0) == 0:
        return None

    success, corners, _gray = detect_chessboard(image, pattern_size)

    if not success:
        return None

    object_points = create_object_points(pattern_size, square_size_mm)

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

        rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)

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
    squared_errors = np.sum(
        (projected_points - actual_points) ** 2,
        axis=1,
    )
    reprojection_rms_px = float(np.sqrt(np.mean(squared_errors)))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector.ravel()

    return transform, reprojection_rms_px


# ============================================================
# 6. eye-to-hand 求解与质量评估
# ============================================================

def check_motion_diversity(
    base_gripper_transforms: list[np.ndarray],
) -> None:
    """检查机器人姿态是否具备足够的旋转多样性。

    Args:
        base_gripper_transforms: T_base_gripper 列表

    Raises:
        CalibrationError: 相对旋转角过小或旋转轴过于单一
    """

    rotations = [
        transform[:3, :3]
        for transform in base_gripper_transforms
    ]

    rotation_axes = []
    max_relative_angle_rad = 0.0

    for first in range(len(rotations)):
        for second in range(first + 1, len(rotations)):

            relative_rotation = rotations[first].T @ rotations[second]
            rotation_vector, _jacobian = cv2.Rodrigues(relative_rotation)
            angle_rad = float(np.linalg.norm(rotation_vector))

            max_relative_angle_rad = max(
                max_relative_angle_rad,
                angle_rad,
            )

            if angle_rad > 1e-6:
                rotation_axes.append(
                    rotation_vector.ravel() / angle_rad
                )

    if max_relative_angle_rad < np.deg2rad(MIN_RELATIVE_ROTATION_DEG):
        raise CalibrationError(
            f"姿态退化：最大相对旋转角 "
            f"{np.degrees(max_relative_angle_rad):.2f} deg，"
            f"低于 {MIN_RELATIVE_ROTATION_DEG:.1f} deg",
            recoverable=True,
        )

    if len(rotation_axes) < 2:
        raise CalibrationError(
            "姿态退化：有效相对旋转样本不足",
            recoverable=True,
        )

    singular_values = np.linalg.svd(
        np.asarray(rotation_axes, dtype=np.float64),
        compute_uv=False,
    )

    if singular_values[0] <= 0.0:
        raise CalibrationError(
            "姿态退化：旋转轴奇异值非法",
            recoverable=True,
        )

    if singular_values[1] / singular_values[0] < MIN_AXIS_DIVERSITY:
        raise CalibrationError(
            "姿态退化：旋转轴过于单一，请增加不同方向的姿态",
            recoverable=True,
        )


def solve_eye_to_hand(
    base_gripper_transforms: list[np.ndarray],
    cam_board_transforms: list[np.ndarray],
) -> np.ndarray:
    """求解 eye-to-hand 的 T_base_camera。

    Args:
        base_gripper_transforms: T_base_gripper 列表
        cam_board_transforms: T_cam_board 列表

    Returns:
        T_base_camera，4x4，平移 mm

    Raises:
        CalibrationError: 样本不足、姿态退化或求解失败
    """

    if len(base_gripper_transforms) != len(cam_board_transforms):
        raise CalibrationError(
            "机器人位姿与棋盘格位姿数量不一致",
            recoverable=False,
        )

    if len(base_gripper_transforms) < MIN_SAMPLES:
        raise CalibrationError(
            f"有效样本不足：{len(base_gripper_transforms)}，"
            f"至少需要 {MIN_SAMPLES}",
            recoverable=True,
        )

    check_motion_diversity(base_gripper_transforms)

    gripper_base_rotations = []
    gripper_base_translations = []
    cam_board_rotations = []
    cam_board_translations = []

    for base_gripper, cam_board in zip(
        base_gripper_transforms,
        cam_board_transforms,
    ):
        gripper_base = transform_inverse(base_gripper)

        gripper_base_rotations.append(gripper_base[:3, :3])
        gripper_base_translations.append(
            gripper_base[:3, 3].reshape(3, 1)
        )

        cam_board_rotations.append(cam_board[:3, :3])
        cam_board_translations.append(
            cam_board[:3, 3].reshape(3, 1)
        )

    try:
        rotation, translation = cv2.calibrateHandEye(
            gripper_base_rotations,
            gripper_base_translations,
            cam_board_rotations,
            cam_board_translations,
            method=HANDEYE_METHOD,
        )
    except cv2.error as error:
        raise CalibrationError(
            f"手眼求解失败：{error}",
            recoverable=True,
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.ravel()

    validate_transform(transform, "T_base_camera")

    return transform


def evaluate_rigidity(
    base_gripper_transforms: list[np.ndarray],
    cam_board_transforms: list[np.ndarray],
    base_camera_transform: np.ndarray,
) -> tuple[float, float]:
    """评估棋盘格在末端坐标系下的刚性一致性。

    Args:
        base_gripper_transforms: T_base_gripper 列表
        cam_board_transforms: T_cam_board 列表
        base_camera_transform: 求解得到的 T_base_camera

    Returns:
        (translation_std_mm, rotation_std_deg)
    """

    translations = []
    rotations = []

    for base_gripper, cam_board in zip(
        base_gripper_transforms,
        cam_board_transforms,
    ):
        gripper_board = (
            transform_inverse(base_gripper)
            @ base_camera_transform
            @ cam_board
        )

        translations.append(gripper_board[:3, 3])
        rotations.append(gripper_board[:3, :3])

    translations = np.asarray(translations, dtype=np.float64)
    translation_std_mm = float(
        np.linalg.norm(np.std(translations, axis=0))
    )

    mean_rotation = average_rotation_matrices(rotations)
    angle_errors = [
        rotation_angle_deg(mean_rotation.T @ rotation)
        for rotation in rotations
    ]
    rotation_std_deg = float(np.std(angle_errors))

    return translation_std_mm, rotation_std_deg


def save_transform(
    transform: np.ndarray,
    output_path: str,
) -> None:
    """原子保存手眼矩阵。

    Args:
        transform: 4x4 T_base_camera
        output_path: 输出路径

    Raises:
        CalibrationError: 目录创建或写入失败
    """

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    temporary_path = f"{output_path}.tmp"

    try:
        with open(temporary_path, "wb") as file:
            np.save(file, transform)

        os.replace(temporary_path, output_path)

    except OSError as error:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass

        raise CalibrationError(
            f"手眼矩阵写入失败：{output_path}：{error}",
            recoverable=False,
        )


# ============================================================
# 7. 标定主流程
# ============================================================

def calibrate_handeye(
    pose_file: str,
    image_dir: str,
    intrinsics_path: str,
    output_path: str,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    square_size_mm: float = DEFAULT_SQUARE_SIZE_MM,
) -> HandEyeReport:
    """执行 eye-to-hand 手眼标定并保存 T_base_camera。

    Args:
        pose_file: pose_recorder 生成的 poses.json
        image_dir: 图像目录（仅作路径解析回退）
        intrinsics_path: 相机内参 npz
        output_path: T_base_camera.npy 输出路径
        pattern_size: 棋盘格内角点数量 (列, 行)
        square_size_mm: 方格边长，mm

    Returns:
        HandEyeReport

    Raises:
        ValueError: 参数非法
        CalibrationError: 数据缺失、样本不足、求解失败或质量不达标
    """

    if not pose_file or not intrinsics_path or not output_path:
        raise ValueError("pose_file/intrinsics_path/output_path 不能为空")

    columns, rows = pattern_size

    if columns < 2 or rows < 2:
        raise ValueError("棋盘格行列内角点数量必须至少为 2")

    if square_size_mm <= 0:
        raise ValueError("square_size_mm 必须大于 0")

    logger = setup_logger()

    logger.info("=" * 60)
    logger.info("开始 eye-to-hand 手眼标定")
    logger.info(f"位姿文件：{pose_file}")
    logger.info(f"相机内参：{intrinsics_path}")
    logger.info(f"棋盘格内角点：{pattern_size}")
    logger.info(f"方格边长：{square_size_mm} mm")
    logger.info("=" * 60)

    camera_matrix, dist_coeffs, metadata = load_intrinsics(
        intrinsics_path
    )

    logger.info(f"相机内参 RMS：{metadata['rms']:.6f} pixel")

    payload = load_pose_payload(pose_file)
    samples = validate_pose_payload(payload, pattern_size)

    sample_count = len(samples)
    logger.info(f"位姿样本数量：{sample_count}")

    if sample_count < MIN_SAMPLES:
        raise CalibrationError(
            f"位姿样本不足：{sample_count}，至少需要 {MIN_SAMPLES}",
            recoverable=True,
        )

    expected_size = None

    if "image_width" in metadata and "image_height" in metadata:
        expected_size = (
            metadata["image_width"],
            metadata["image_height"],
        )

    base_gripper_transforms = []
    cam_board_transforms = []
    reprojection_errors = []
    skipped_count = 0

    for index, sample in enumerate(samples):

        if not isinstance(sample, dict):
            raise CalibrationError(
                f"样本 {index} 不是对象",
                recoverable=False,
            )

        base_gripper = pose_dict_to_matrix(sample.get("robot_pose"))

        image_path = resolve_image_path(
            pose_file,
            image_dir,
            sample.get("image_path"),
        )

        image = cv2.imread(str(image_path))

        if image is None:
            logger.warning(f"[跳过] 图像无法读取：{image_path}")
            skipped_count += 1
            continue

        if expected_size is not None:
            height, width = image.shape[:2]

            if (width, height) != expected_size:
                raise CalibrationError(
                    f"图像分辨率与内参不一致：{image_path} 为 "
                    f"{(width, height)}，预期 {expected_size}",
                    recoverable=False,
                )

        estimate = estimate_board_pose(
            image,
            camera_matrix,
            dist_coeffs,
            pattern_size,
            square_size_mm,
        )

        if estimate is None:
            logger.warning(f"[跳过] 未检测到棋盘格：{image_path}")
            skipped_count += 1
            continue

        cam_board, reprojection_rms_px = estimate

        base_gripper_transforms.append(base_gripper)
        cam_board_transforms.append(cam_board)
        reprojection_errors.append(reprojection_rms_px)

        logger.info(
            f"[成功] 样本 {index:03d}：重投影 "
            f"{reprojection_rms_px:.4f} pixel，图像={image_path.name}"
        )

    success_count = len(cam_board_transforms)

    logger.info(
        f"有效样本：{success_count} / {sample_count}，跳过 {skipped_count}"
    )

    if success_count < MIN_SAMPLES:
        raise CalibrationError(
            f"有效样本不足：{success_count}，至少需要 {MIN_SAMPLES}",
            recoverable=True,
        )

    if success_count < RECOMMENDED_SAMPLES:
        logger.warning(
            f"有效样本少于建议值 {RECOMMENDED_SAMPLES}，"
            "标定稳定性可能不足"
        )

    base_camera_transform = solve_eye_to_hand(
        base_gripper_transforms,
        cam_board_transforms,
    )

    translation_std_mm, rotation_std_deg = evaluate_rigidity(
        base_gripper_transforms,
        cam_board_transforms,
        base_camera_transform,
    )

    reprojection_rms_px = float(
        np.sqrt(np.mean(np.square(reprojection_errors)))
    )

    passed = (
        translation_std_mm <= TRANSLATION_STD_MAX_MM
        and rotation_std_deg <= ROTATION_STD_MAX_DEG
        and reprojection_rms_px <= REPROJECTION_MAX_PX
    )

    report = HandEyeReport(
        sample_count=sample_count,
        success_count=success_count,
        translation_std_mm=translation_std_mm,
        rotation_std_deg=rotation_std_deg,
        reprojection_rms_px=reprojection_rms_px,
        passed=passed,
    )

    logger.info("-" * 60)
    logger.info("T_base_camera：")
    logger.info(f"\n{np.array2string(base_camera_transform, precision=6)}")
    logger.info(
        f"平移标准差：{translation_std_mm:.4f} mm "
        f"(门槛 {TRANSLATION_STD_MAX_MM:.1f} mm)"
    )
    logger.info(
        f"旋转标准差：{rotation_std_deg:.4f} deg "
        f"(门槛 {ROTATION_STD_MAX_DEG:.1f} deg)"
    )
    logger.info(
        f"重投影 RMS：{reprojection_rms_px:.4f} pixel "
        f"(门槛 {REPROJECTION_MAX_PX:.1f} pixel)"
    )

    if not passed:
        raise CalibrationError(
            "手眼标定质量不达标："
            f"平移标准差 {translation_std_mm:.4f} mm，"
            f"旋转标准差 {rotation_std_deg:.4f} deg，"
            f"重投影 RMS {reprojection_rms_px:.4f} pixel；"
            "未写入 T_base_camera.npy",
            recoverable=True,
        )

    save_transform(base_camera_transform, output_path)

    logger.info("=" * 60)
    logger.info("手眼标定完成")
    logger.info(f"手眼矩阵已保存：{output_path}")
    logger.info("=" * 60)

    return report


# ============================================================
# 8. 命令行入口
# ============================================================

def main() -> int:
    """手眼标定命令行入口。

    Returns:
        进程退出码：0 成功，1 失败，130 用户中断
    """

    parser = argparse.ArgumentParser(
        description="Nova 2 eye-to-hand 手眼标定工具"
    )

    parser.add_argument(
        "--pose-file",
        default=DEFAULT_POSE_FILE,
        help="pose_recorder 生成的 poses.json",
    )

    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help="图像目录（image_path 解析失败时回退）",
    )

    parser.add_argument(
        "--intrinsics",
        default=DEFAULT_INTRINSICS_PATH,
        help="相机内参 npz 路径",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="T_base_camera.npy 输出路径",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_PATTERN_SIZE[0],
        help="棋盘格内角点列数",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_PATTERN_SIZE[1],
        help="棋盘格内角点行数",
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE_MM,
        help="棋盘格方格边长，单位 mm",
    )

    args = parser.parse_args()

    logger = setup_logger()

    try:
        report = calibrate_handeye(
            pose_file=args.pose_file,
            image_dir=args.image_dir,
            intrinsics_path=args.intrinsics,
            output_path=args.output,
            pattern_size=(args.cols, args.rows),
            square_size_mm=args.square_size,
        )

    except Nova2ProjectError as error:
        logger.error(f"手眼标定失败：{error}")
        return 1

    except KeyboardInterrupt:
        logger.warning("操作员中断手眼标定")
        return 130

    logger.info(
        f"标定通过：{report.success_count} / {report.sample_count} 样本"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
