"""应用级配置与标定聚合类型。

职责：
    1. 定义 `config/robot_config.yaml` 的解析结果 `AppConfig`
    2. 定义入口组装所需的标定聚合 `CalibrationBundle`
    3. 提供 robot / camera / calibration / task / acceptance 分段配置类型

说明：
    - 本模块只承载数据结构，不负责读取 YAML、不访问硬件。
    - 长度统一 mm，角度统一 deg，速度统一为 `velocity_ratio_pct`。
    - `CalibrationBundle.rack_model` 在阶段 8 前允许为 None。
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from common.types import GripperConfig


# ============================================================
# 1. 配置分段类型
# ============================================================

@dataclass(frozen=True)
class RobotConfig:
    """Nova 2 连接与默认速度配置。

    Args:
        ip: 控制柜 IP
        port: Dashboard 指令口
        feedback_port: 反馈口；None 表示关闭反馈
        velocity_ratio_pct: 默认运动速度比例，范围 1..100（不是 mm/s）
    """

    ip: str
    port: int
    feedback_port: int | None
    velocity_ratio_pct: int


@dataclass(frozen=True)
class CameraConfig:
    """相机运行配置。

    Args:
        intrinsics_file: 相机内参 npz 绝对路径
        width: 图像宽度，px
        height: 图像高度，px
    """

    intrinsics_file: str
    width: int
    height: int


@dataclass(frozen=True)
class CalibrationConfig:
    """标定文件与桌面参数配置。

    Args:
        T_base_camera_file: 相机 -> 基座变换文件绝对路径
        T_base_rack_file: 基座 -> 试管架变换文件绝对路径
        table_z_mm: 桌面在基座坐标系下的 Z 高度，mm
        max_age_days: 标定文件允许的最大天龄；None 表示不按天龄校验
    """

    T_base_camera_file: str
    T_base_rack_file: str
    table_z_mm: float
    max_age_days: float | None


@dataclass(frozen=True)
class TaskConfig:
    """任务运行配置。

    Args:
        tube_count: 连续任务试管数量
        approach_height_mm: 接近/抬升高度，mm
        insert_velocity_ratio_pct: 插入速度比例，范围 1..100
    """

    tube_count: int
    approach_height_mm: float
    insert_velocity_ratio_pct: int


@dataclass(frozen=True)
class AcceptanceConfig:
    """验收阈值配置，默认值来自 roadmap 初始门槛。

    Args:
        camera_rms_px: 相机整体重投影 RMS 门槛，pixel
        camera_max_error_px: 单张最大重投影误差门槛，pixel
        handeye_translation_std_mm: 手眼平移一致性门槛，mm
        handeye_rotation_std_deg: 手眼旋转一致性门槛，deg
        transform_mean_error_mm: 坐标转换平均误差门槛，mm
        transform_max_error_mm: 坐标转换最大误差门槛，mm
        insert_alignment_mm: 插入对准允许偏差，mm
    """

    camera_rms_px: float = 1.0
    camera_max_error_px: float = 2.0
    handeye_translation_std_mm: float = 2.0
    handeye_rotation_std_deg: float = 0.5
    transform_mean_error_mm: float = 3.0
    transform_max_error_mm: float = 5.0
    insert_alignment_mm: float = 2.0


# ============================================================
# 2. 入口聚合类型
# ============================================================

@dataclass(frozen=True)
class AppConfig:
    """`config/robot_config.yaml` 的完整解析结果。

    Args:
        robot: 机器人连接与速度
        gripper: 夹爪协议配置
        camera: 相机文件与分辨率
        calibration: 标定文件与桌面参数
        task: 任务参数
        acceptance: 验收阈值
        config_path: 配置文件绝对路径
        config_mtime_s: 配置文件修改时间，Unix 秒
    """

    robot: RobotConfig
    gripper: GripperConfig
    camera: CameraConfig
    calibration: CalibrationConfig
    task: TaskConfig
    acceptance: AcceptanceConfig
    config_path: str
    config_mtime_s: float


@dataclass(frozen=True)
class CalibrationBundle:
    """入口加载并校验后的标定集合。

    Args:
        K: 3x3 相机内参，单位 pixel
        D: 畸变系数
        T_base_camera_mm: 4x4 相机 -> 基座变换，平移 mm
        T_base_rack_mm: 4x4 基座 -> 试管架变换，平移 mm；未标定为 None
        rack_model: 试管架孔位模型；阶段 8 前为 None
        loaded_at: 加载时间（UTC）
        rms_px: 相机内参重投影 RMS，pixel
        source_mtimes: 各标定源文件修改时间，Unix 秒
    """

    K: np.ndarray
    D: np.ndarray
    T_base_camera_mm: np.ndarray
    T_base_rack_mm: np.ndarray | None
    rack_model: object | None
    loaded_at: datetime
    rms_px: float
    source_mtimes: dict[str, float] = field(default_factory=dict)
