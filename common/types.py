"""跨层共享类型。

功能：
    1. 定义项目内部统一的位姿类型 Pose（位置 mm，姿态 deg）
    2. 定义手眼标定报告类型 HandEyeReport
    3. 定义相机数据源协议 CameraSource
    4. 提供各层共用的最小数据结构，避免 robot/vision/task 相互依赖

说明：
    本模块不得依赖 robot / vision / task。
    其余共享类型（Point3D、TubeDetection 等）在对应模块实现时按需补充。
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


# ============================================================
# 1. 位姿类型
# ============================================================

@dataclass(frozen=True)
class Pose:
    """末端位姿，位置单位 mm，姿态单位 deg。

    Args:
        x_mm: X 位置，mm
        y_mm: Y 位置，mm
        z_mm: Z 位置，mm
        rx_deg: 绕固定轴 X 旋转，deg
        ry_deg: 绕固定轴 Y 旋转，deg
        rz_deg: 绕固定轴 Z 旋转，deg

    说明：
        项目内部统一采用固定轴 XYZ（roll-pitch-yaw）。
        TCP/User 坐标系由 Nova2Client 固定，上层不得切换。
    """

    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float


# ============================================================
# 2. 标定报告类型
# ============================================================

@dataclass(frozen=True)
class HandEyeReport:
    """手眼标定报告。

    Args:
        sample_count: poses.json 中的样本总数
        success_count: 成功检测棋盘格并参与求解的样本数
        translation_std_mm: 棋盘格在末端坐标系下平移离散度，mm
        rotation_std_deg: 棋盘格在末端坐标系下旋转离散度，deg
        reprojection_rms_px: 参与求解样本的整体重投影 RMS，pixel
        passed: 是否满足质量门槛
    """

    sample_count: int
    success_count: int
    translation_std_mm: float
    rotation_std_deg: float
    reprojection_rms_px: float
    passed: bool


# ============================================================
# 3. 相机数据源协议
# ============================================================

@runtime_checkable
class CameraSource(Protocol):
    """相机数据源协议：只约定读取单帧图像。"""

    def read(self) -> np.ndarray:
        """读取一帧 BGR 图像。

        Returns:
            shape 为 (H, W, 3) 的 BGR 图像数组；读取失败返回 None。
        """
        ...
