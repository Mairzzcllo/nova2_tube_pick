"""跨层共享类型。

功能：
    1. 定义项目内部统一的位姿类型 Pose（位置 mm，姿态 deg）
    2. 定义三维点 Point3D 与相机系平面 Plane
    3. 定义试管检测结果类型 TubeDetection、孔位像素 HolePixel
    4. 定义试管几何 TubeGeometry 与孔位位姿 HolePose
    5. 定义手眼标定报告类型 HandEyeReport
    6. 定义标定验证报告类型 TransformVerifyReport、ReprojectionReport
    7. 定义相机数据源协议 CameraSource
    8. 定义夹爪状态 GraspStatus 与夹爪配置 GripperConfig
    9. 提供各层共用的最小数据结构，避免 robot/vision/task 相互依赖

说明：
    本模块不得依赖 robot / vision / task。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


# ============================================================
# 1. 位姿与几何类型
# ============================================================

@dataclass(frozen=True)
class Point3D:
    """三维点，单位 mm。

    Args:
        x_mm: X 坐标，mm
        y_mm: Y 坐标，mm
        z_mm: Z 坐标，mm
    """

    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class Plane:
    """平面方程 a*x + b*y + c*z + d = 0。

    Args:
        a: X 法向分量
        b: Y 法向分量
        c: Z 法向分量
        d: 平面常数项
        frame: 平面所在坐标系，当前仅支持 `camera`

    说明：
        (a, b, c) 不要求归一化，使用前由消费方统一归一化。
    """

    a: float
    b: float
    c: float
    d: float
    frame: str = "camera"


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
# 2. 视觉检测类型
# ============================================================

@dataclass(frozen=True)
class TubeDetection:
    """单支试管检测结果。

    Args:
        tube_id: 试管编号；由检测层按图像内位置稳定排序生成
        center_u_px: 试管中心像素横坐标，pixel
        center_v_px: 试管中心像素纵坐标，pixel
        axis_u: 长轴单位向量 u 分量；方向由彩盖端指向管底端
        axis_v: 长轴单位向量 v 分量；方向由彩盖端指向管底端
        angle_deg: 长轴角度，范围 [0, 180)，图像坐标系（v 轴向下）
        confidence: 检测置信度，范围 [0, 1]

    说明：
        axis_u/axis_v 保留方向信息，angle_deg 仅表示无向长轴角度。
        下游计算抓取位姿时优先使用 axis_u/axis_v。
    """

    tube_id: str
    center_u_px: float
    center_v_px: float
    axis_u: float
    axis_v: float
    angle_deg: float
    confidence: float


@dataclass(frozen=True)
class TubeGeometry:
    """试管几何参数。

    Args:
        diameter_mm: 管身直径，mm
        length_mm: 管身长度，mm
        grasp_offset_mm: 夹持点相对检测中心的偏移，mm；
            正方向沿长轴由彩盖端指向管底端
    """

    diameter_mm: float
    length_mm: float
    grasp_offset_mm: float


@dataclass(frozen=True)
class HolePixel:
    """孔位像素坐标。

    Args:
        hole_id: 孔位编号
        u_px: 孔位中心像素横坐标，pixel
        v_px: 孔位中心像素纵坐标，pixel
    """

    hole_id: str
    u_px: float
    v_px: float


@dataclass(frozen=True)
class HolePose:
    """机器人基坐标系下的孔位位姿。

    Args:
        hole_id: 孔位编号
        pose_base: 基座位姿，位置 mm、姿态 deg
    """

    hole_id: str
    pose_base: Pose


# ============================================================
# 3. 标定报告类型
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
# 4. 标定验证报告类型
# ============================================================

@dataclass(frozen=True)
class TransformVerifyReport:
    """相机到基座坐标变换验证报告。

    Args:
        n_points: 参与验证的测试点数量
        per_point_errors_mm: 每个测试点的欧氏距离误差，mm
        mean_error_mm: 全部测试点的平均误差，mm
        max_error_mm: 全部测试点的最大误差，mm
        passed: 是否同时满足平均误差与最大误差门槛
        reason: PASS/FAIL 说明
    """

    n_points: int
    per_point_errors_mm: tuple[float, ...]
    mean_error_mm: float
    max_error_mm: float
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class ReprojectionReport:
    """相机内参重投影验证报告。

    Args:
        n_views: 有效棋盘格视图数量
        rms_px: 全部有效角点的整体 RMS 重投影误差，pixel
        max_error_px: 全部有效角点中的最大重投影误差，pixel
        per_view_errors_px: 每张有效视图的 RMS 重投影误差，pixel
        passed: 是否满足最大像素误差门槛
        reason: PASS/FAIL 说明
    """

    n_views: int
    rms_px: float
    max_error_px: float
    per_view_errors_px: tuple[float, ...]
    passed: bool
    reason: str = ""


# ============================================================
# 5. 相机数据源协议
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


# ============================================================
# 6. 夹爪类型
# ============================================================

class GraspStatus(Enum):
    """夹爪夹持状态。

    取值：
        GRASPED: 已确认夹住物体
        NOT_GRASPED: 已确认未夹住物体
        UNKNOWN: 无反馈或反馈不可信，不得当作成功
    """

    GRASPED = "grasped"
    NOT_GRASPED = "not_grasped"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GripperConfig:
    """夹爪配置。

    Args:
        control_mode: 控制方式，取值 `tool_do` / `dual_do` / `serial` / `unknown`
        open_do_index: 张开动作使用的末端 DO 编号（tool_do 模式）
        close_do_index: 闭合动作使用的末端 DO 编号（dual_do 模式）
        do_active_level: DO 有效电平，1 表示 ON，0 表示 OFF
        feedback_di_index: 夹持反馈 DI 编号；None 表示无反馈
        feedback_active_level: DI 反馈有效电平，1 表示 ON，0 表示 OFF
        feedback_timeout_s: 等待反馈到位的超时，s（预留）
        tool_power_required: 开合前是否需要打开末端工具电源

    说明：
        夹爪硬件协议尚未确定，`control_mode` 默认为 `unknown`。
        在协议确定前，`GripperController` 对开合动作抛出 `GripperError`，
        `is_grasped()` 返回 `GraspStatus.UNKNOWN`，不得假装成功。
    """

    control_mode: str = "unknown"
    open_do_index: int | None = None
    close_do_index: int | None = None
    do_active_level: int = 1
    feedback_di_index: int | None = None
    feedback_active_level: int = 1
    feedback_timeout_s: float = 2.0
    tool_power_required: bool = False
