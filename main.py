"""Nova 2 试管抓取项目入口。

职责：
    1. 加载并校验 `config/robot_config.yaml`
    2. 加载并校验相机内参与手眼标定
    3. 组装机器人、视觉、任务模块
    4. 连接机械臂并启动作业流程
    5. 统一日志、异常分类与资源释放

运行：
    ~/miniconda3/envs/Dobot-Sim/bin/python main.py
    ~/miniconda3/envs/Dobot-Sim/bin/python main.py --dry-run

说明：
    - 本模块只做入口组装与生命周期管理，不写识别/矩阵/运动算法。
    - `build_pipeline` 只组装；连接由 `Pipeline.connect()` / `main()` 负责。
    - 相机数据源未提供时直接抛 `FatalError`。
    - 标定文件缺失、过期或字段非法时抛 `FatalError`，不得使用占位矩阵。
"""

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calibration.handeye_calibration import load_intrinsics, validate_transform
from common.app import (
    AcceptanceConfig,
    AppConfig,
    CalibrationBundle,
    CalibrationConfig,
    CameraConfig,
    RobotConfig,
    TaskConfig,
)
from common.errors import (
    CalibrationError,
    FatalError,
    Nova2ProjectError,
    RecoverableError,
)
from common.types import CameraSource, GripperConfig

logger = logging.getLogger(__name__)


# ============================================================
# 1. 常量与默认参数
# ============================================================

# 默认配置文件
DEFAULT_CONFIG_PATH = "config/robot_config.yaml"

# 任务状态落盘文件
DEFAULT_STATE_PATH = "data/task_state.json"

# 运行日志目录
DEFAULT_LOG_DIR = "data/logs"

# 默认连续任务试管数量
DEFAULT_TUBE_COUNT = 3

# 速度比例合法范围（厂商百分比，不是 mm/s）
MIN_VELOCITY_RATIO_PCT = 1
MAX_VELOCITY_RATIO_PCT = 100

# 每日秒数
SECONDS_PER_DAY = 86400.0

# 已识别的夹爪控制方式
VALID_GRIPPER_MODES = frozenset({"tool_do", "dual_do", "serial", "unknown"})

# 进程退出码
EXIT_OK = 0
EXIT_RECOVERABLE = 1
EXIT_FATAL = 2
EXIT_INTERRUPTED = 130


# ============================================================
# 2. 路径与配置解析辅助
# ============================================================

def _resolve_path(path_value: str) -> str:
    """把配置中的相对路径解析为相对仓库根目录的绝对路径。

    Args:
        path_value: 配置中的路径字符串

    Returns:
        绝对路径字符串
    """

    path = Path(path_value)

    if path.is_absolute():
        return str(path)

    return str((_ROOT / path).resolve())


def _require_mapping(mapping: dict, key: str, source: str) -> dict:
    """读取并校验必填映射段。

    Args:
        mapping: 父级映射
        key: 段名
        source: 配置文件路径，用于错误信息

    Returns:
        子映射

    Raises:
        FatalError: 段缺失或不是映射
    """

    value = mapping.get(key)

    if not isinstance(value, dict):
        raise FatalError(f"配置 {source} 缺少映射段：{key}")

    return value


def _require_value(mapping: dict, key: str, source: str):
    """读取必填字段。

    Args:
        mapping: 映射
        key: 字段名
        source: 配置段名，用于错误信息

    Returns:
        字段原始值

    Raises:
        FatalError: 字段缺失
    """

    if key not in mapping:
        raise FatalError(f"配置 {source} 缺少字段：{key}")

    return mapping[key]


def _as_str(value, name: str) -> str:
    """校验并返回非空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise FatalError(f"配置 {name} 必须是非空字符串")

    return value.strip()


def _as_int(
    value,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """校验并返回整数，可限定范围。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise FatalError(f"配置 {name} 必须是整数，当前为 {value!r}")

    if minimum is not None and value < minimum:
        raise FatalError(f"配置 {name} 不能小于 {minimum}，当前为 {value}")

    if maximum is not None and value > maximum:
        raise FatalError(f"配置 {name} 不能大于 {maximum}，当前为 {value}")

    return value


def _as_optional_int(value, name: str, minimum: int | None = None) -> int | None:
    """校验并返回可空整数。"""

    if value is None:
        return None

    return _as_int(value, name, minimum=minimum)


def _as_float(
    value,
    name: str,
    positive: bool = False,
) -> float:
    """校验并返回有限浮点数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FatalError(f"配置 {name} 必须是数值，当前为 {value!r}")

    result = float(value)

    if not math.isfinite(result):
        raise FatalError(f"配置 {name} 必须是有限值")

    if positive and result <= 0.0:
        raise FatalError(f"配置 {name} 必须大于 0，当前为 {result}")

    return result


def _as_bool(value, name: str) -> bool:
    """校验并返回布尔值。"""

    if not isinstance(value, bool):
        raise FatalError(f"配置 {name} 必须是布尔值，当前为 {value!r}")

    return value


def _parse_gripper(raw: dict) -> GripperConfig:
    """解析夹爪配置段。

    Args:
        raw: `gripper` 映射

    Returns:
        GripperConfig

    Raises:
        FatalError: 字段非法
    """

    control_mode = _as_str(
        raw.get("control_mode", "unknown"),
        "gripper.control_mode",
    )

    if control_mode not in VALID_GRIPPER_MODES:
        raise FatalError(
            f"配置 gripper.control_mode 非法：{control_mode}，"
            f"可选 {sorted(VALID_GRIPPER_MODES)}"
        )

    return GripperConfig(
        control_mode=control_mode,
        open_do_index=_as_optional_int(
            raw.get("open_do_index"), "gripper.open_do_index", minimum=0
        ),
        close_do_index=_as_optional_int(
            raw.get("close_do_index"), "gripper.close_do_index", minimum=0
        ),
        do_active_level=_as_int(
            raw.get("do_active_level", 1), "gripper.do_active_level", 0, 1
        ),
        feedback_di_index=_as_optional_int(
            raw.get("feedback_di_index"), "gripper.feedback_di_index", minimum=0
        ),
        feedback_active_level=_as_int(
            raw.get("feedback_active_level", 1),
            "gripper.feedback_active_level",
            0,
            1,
        ),
        feedback_timeout_s=_as_float(
            raw.get("feedback_timeout_s", 2.0),
            "gripper.feedback_timeout_s",
            positive=True,
        ),
        tool_power_required=_as_bool(
            raw.get("tool_power_required", False),
            "gripper.tool_power_required",
        ),
    )


def _parse_acceptance(raw) -> AcceptanceConfig:
    """解析可选的验收阈值段，缺省使用 roadmap 初始门槛。

    Args:
        raw: `acceptance` 映射或 None

    Returns:
        AcceptanceConfig

    Raises:
        FatalError: 字段非法
    """

    defaults = AcceptanceConfig()

    if raw is None:
        return defaults

    if not isinstance(raw, dict):
        raise FatalError("配置 acceptance 必须是映射段")

    return AcceptanceConfig(
        camera_rms_px=_as_float(
            raw.get("camera_rms_px", defaults.camera_rms_px),
            "acceptance.camera_rms_px",
            positive=True,
        ),
        camera_max_error_px=_as_float(
            raw.get("camera_max_error_px", defaults.camera_max_error_px),
            "acceptance.camera_max_error_px",
            positive=True,
        ),
        handeye_translation_std_mm=_as_float(
            raw.get("handeye_translation_std_mm", defaults.handeye_translation_std_mm),
            "acceptance.handeye_translation_std_mm",
            positive=True,
        ),
        handeye_rotation_std_deg=_as_float(
            raw.get("handeye_rotation_std_deg", defaults.handeye_rotation_std_deg),
            "acceptance.handeye_rotation_std_deg",
            positive=True,
        ),
        transform_mean_error_mm=_as_float(
            raw.get("transform_mean_error_mm", defaults.transform_mean_error_mm),
            "acceptance.transform_mean_error_mm",
            positive=True,
        ),
        transform_max_error_mm=_as_float(
            raw.get("transform_max_error_mm", defaults.transform_max_error_mm),
            "acceptance.transform_max_error_mm",
            positive=True,
        ),
        insert_alignment_mm=_as_float(
            raw.get("insert_alignment_mm", defaults.insert_alignment_mm),
            "acceptance.insert_alignment_mm",
            positive=True,
        ),
    )


# ============================================================
# 3. 配置加载
# ============================================================

def load_config(path: str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """加载并校验 `config/robot_config.yaml`。

    Args:
        path: 配置文件路径；相对路径按仓库根目录解析

    Returns:
        AppConfig

    Raises:
        FatalError: 文件不存在、YAML 非法或字段缺失/越界
    """

    config_path = _resolve_path(path)

    if not os.path.isfile(config_path):
        raise FatalError(f"配置文件不存在：{config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise FatalError(f"配置文件读取失败：{config_path}：{error}") from error

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise FatalError(f"配置文件根节点必须是映射：{config_path}")

    robot_raw = _require_mapping(raw, "robot", config_path)
    gripper_raw = _require_mapping(raw, "gripper", config_path)
    camera_raw = _require_mapping(raw, "camera", config_path)
    calibration_raw = _require_mapping(raw, "calibration", config_path)
    task_raw = _require_mapping(raw, "task", config_path)

    robot = RobotConfig(
        ip=_as_str(_require_value(robot_raw, "ip", "robot"), "robot.ip"),
        port=_as_int(
            _require_value(robot_raw, "port", "robot"),
            "robot.port",
            minimum=1,
            maximum=65535,
        ),
        feedback_port=_as_optional_int(
            robot_raw.get("feedback_port"),
            "robot.feedback_port",
            minimum=1,
        ),
        velocity_ratio_pct=_as_int(
            robot_raw.get("speed", 50),
            "robot.speed",
            MIN_VELOCITY_RATIO_PCT,
            MAX_VELOCITY_RATIO_PCT,
        ),
    )

    camera = CameraConfig(
        intrinsics_file=_resolve_path(
            _as_str(
                _require_value(camera_raw, "intrinsics_file", "camera"),
                "camera.intrinsics_file",
            )
        ),
        width=_as_int(
            _require_value(camera_raw, "width", "camera"),
            "camera.width",
            minimum=1,
        ),
        height=_as_int(
            _require_value(camera_raw, "height", "camera"),
            "camera.height",
            minimum=1,
        ),
    )

    max_age_raw = calibration_raw.get("max_age_days")
    max_age_days = (
        None
        if max_age_raw is None
        else _as_float(max_age_raw, "calibration.max_age_days", positive=True)
    )

    calibration = CalibrationConfig(
        T_base_camera_file=_resolve_path(
            _as_str(
                _require_value(
                    calibration_raw,
                    "T_base_camera_file",
                    "calibration",
                ),
                "calibration.T_base_camera_file",
            )
        ),
        T_base_rack_file=_resolve_path(
            _as_str(
                _require_value(calibration_raw, "T_base_rack_file", "calibration"),
                "calibration.T_base_rack_file",
            )
        ),
        table_z_mm=_as_float(
            _require_value(calibration_raw, "table_z_mm", "calibration"),
            "calibration.table_z_mm",
        ),
        max_age_days=max_age_days,
    )

    task = TaskConfig(
        tube_count=_as_int(
            task_raw.get("tube_count", DEFAULT_TUBE_COUNT),
            "task.tube_count",
            minimum=1,
        ),
        approach_height_mm=_as_float(
            _require_value(task_raw, "approach_height_mm", "task"),
            "task.approach_height_mm",
            positive=True,
        ),
        insert_velocity_ratio_pct=_as_int(
            task_raw.get("insert_speed", 10),
            "task.insert_speed",
            MIN_VELOCITY_RATIO_PCT,
            MAX_VELOCITY_RATIO_PCT,
        ),
    )

    acceptance = _parse_acceptance(raw.get("acceptance"))

    try:
        config_mtime_s = os.path.getmtime(config_path)
    except OSError as error:
        raise FatalError(f"配置文件修改时间读取失败：{config_path}：{error}") from error

    return AppConfig(
        robot=robot,
        gripper=_parse_gripper(gripper_raw),
        camera=camera,
        calibration=calibration,
        task=task,
        acceptance=acceptance,
        config_path=config_path,
        config_mtime_s=config_mtime_s,
    )


# ============================================================
# 4. 标定加载与校验
# ============================================================

def _check_freshness(path: str, config: AppConfig, label: str) -> float:
    """校验标定文件存在、未超龄且不早于配置文件修改时间。

    Args:
        path: 标定文件路径
        config: 应用配置
        label: 日志标签

    Returns:
        文件修改时间，Unix 秒

    Raises:
        FatalError: 文件缺失、超龄或早于配置修改时间
    """

    if not os.path.isfile(path):
        raise FatalError(f"{label}文件不存在：{path}")

    try:
        mtime_s = os.path.getmtime(path)
    except OSError as error:
        raise FatalError(f"{label}修改时间读取失败：{path}：{error}") from error

    max_age_days = config.calibration.max_age_days

    if max_age_days is not None:
        age_s = time.time() - mtime_s

        if age_s > max_age_days * SECONDS_PER_DAY:
            raise FatalError(
                f"{label}已过期：{path}，"
                f"文件年龄 {age_s / SECONDS_PER_DAY:.1f} 天，"
                f"上限 {max_age_days:.1f} 天，请重新标定"
            )

    if mtime_s < config.config_mtime_s:
        raise FatalError(
            f"{label}早于配置文件修改时间：{path}，"
            f"文件 mtime={mtime_s:.0f}，"
            f"config mtime={config.config_mtime_s:.0f}，请重新标定"
        )

    return mtime_s


def _load_transform(path: str, label: str) -> np.ndarray:
    """读取并校验 4x4 刚体变换。

    Args:
        path: npy 文件路径
        label: 日志标签

    Returns:
        4x4 float64 变换矩阵

    Raises:
        FatalError: 文件缺失、读取失败或矩阵非法
    """

    if not os.path.isfile(path):
        raise FatalError(f"{label}文件不存在：{path}")

    try:
        transform = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise FatalError(f"{label}读取失败：{path}：{error}") from error

    if not isinstance(transform, np.ndarray):
        raise FatalError(f"{label}内容不是 ndarray：{path}")

    try:
        validate_transform(transform, label)
    except CalibrationError as error:
        raise FatalError(f"{label}校验失败：{error}") from error

    return np.asarray(transform, dtype=np.float64)


def _load_optional_transform(
    path: str,
    config: AppConfig,
    label: str,
) -> np.ndarray | None:
    """读取可选的试管架变换；缺失或空文件按未标定处理。

    Args:
        path: T_base_rack.npy 路径
        config: 应用配置
        label: 日志标签

    Returns:
        4x4 变换矩阵或 None
    """

    if not os.path.isfile(path):
        logger.warning("%s不存在，按未标定处理：%s", label, path)
        return None

    if os.path.getsize(path) == 0:
        logger.warning("%s为空文件，按未标定处理：%s", label, path)
        return None

    _check_freshness(path, config, label)
    return _load_transform(path, label)


def load_calibration(config: AppConfig) -> CalibrationBundle:
    """加载并校验相机内参与手眼标定。

    Args:
        config: AppConfig

    Returns:
        CalibrationBundle

    Raises:
        FatalError: 标定缺失、过期或字段非法
    """

    if not isinstance(config, AppConfig):
        raise FatalError(
            f"config 类型非法：{type(config).__name__}，预期 AppConfig"
        )

    intrinsics_path = config.camera.intrinsics_file
    intrinsics_mtime_s = _check_freshness(
        intrinsics_path,
        config,
        "相机内参",
    )

    try:
        camera_matrix, dist_coeffs, metadata = load_intrinsics(intrinsics_path)
    except CalibrationError as error:
        raise FatalError(f"相机内参加载失败：{error}") from error

    t_base_camera_path = config.calibration.T_base_camera_file
    t_base_camera_mtime_s = _check_freshness(
        t_base_camera_path,
        config,
        "T_base_camera",
    )
    T_base_camera_mm = _load_transform(t_base_camera_path, "T_base_camera")

    t_base_rack_path = config.calibration.T_base_rack_file
    T_base_rack_mm = _load_optional_transform(
        t_base_rack_path,
        config,
        "T_base_rack",
    )

    # 阶段 8 前 rack_model 允许为 None；插入任务必须由任务层重新校验。
    rack_model = None

    if T_base_rack_mm is None:
        logger.warning("T_base_rack 未标定，rack_model=None，仅允许执行抓取阶段")
    else:
        logger.warning("T_base_rack 已加载，但 rack_model 尚未实现，按 None 处理")

    source_mtimes = {
        intrinsics_path: intrinsics_mtime_s,
        t_base_camera_path: t_base_camera_mtime_s,
    }

    if T_base_rack_mm is not None:
        source_mtimes[t_base_rack_path] = os.path.getmtime(t_base_rack_path)

    return CalibrationBundle(
        K=camera_matrix,
        D=dist_coeffs,
        T_base_camera_mm=T_base_camera_mm,
        T_base_rack_mm=T_base_rack_mm,
        rack_model=rack_model,
        loaded_at=datetime.now(timezone.utc),
        rms_px=float(metadata.get("rms", float("nan"))),
        source_mtimes=source_mtimes,
    )


# ============================================================
# 5. 流水线组装
# ============================================================

class Pipeline:
    """已组装流水线，负责连接、运行与资源释放。

    Args:
        client: Nova2Client
        manager: TubeTaskManager
        camera: CameraSource
        config: AppConfig

    说明：
        `build_pipeline` 只组装，本类负责 `connect()` / `run()` / `close()`。
    """

    def __init__(self, client, manager, camera: CameraSource, config: AppConfig):
        self.client = client
        self.manager = manager
        self.camera = camera
        self.config = config

    def connect(self) -> None:
        """连接并使能机械臂。"""

        self.client.connect()

    def run(self, tube_count: int | None = None):
        """执行连续任务。

        Args:
            tube_count: 试管数量；None 时使用配置值

        Returns:
            TaskReport

        Raises:
            ValueError: 试管数量非法
        """

        count = self.config.task.tube_count if tube_count is None else tube_count

        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"tube_count 必须是正整数，当前为 {count!r}")

        return self.manager.run(tube_count=count)

    def emergency_stop(self) -> None:
        """尝试急停机械臂；失败只记录，不覆盖原始异常。"""

        try:
            self.client.stop()
        except Nova2ProjectError as error:
            logger.error("急停失败：%s", error)

    def close(self) -> None:
        """断开机械臂连接；幂等。"""

        try:
            self.client.disconnect()
        except Nova2ProjectError as error:
            logger.error("断开连接失败：%s", error)


def build_pipeline(
    config: AppConfig,
    calibration: CalibrationBundle,
    camera: CameraSource | None = None,
) -> Pipeline:
    """组装视觉、机器人、任务模块。

    Args:
        config: AppConfig
        calibration: CalibrationBundle
        camera: 相机数据源；None 直接抛 FatalError

    Returns:
        Pipeline

    Raises:
        FatalError: 相机缺失、依赖模块未实现或组装失败
    """

    if not isinstance(config, AppConfig):
        raise FatalError(
            f"config 类型非法：{type(config).__name__}，预期 AppConfig"
        )

    if not isinstance(calibration, CalibrationBundle):
        raise FatalError(
            f"calibration 类型非法：{type(calibration).__name__}，预期 CalibrationBundle"
        )

    if camera is None:
        raise FatalError("未提供相机数据源，无法构建流水线")

    try:
        from vision.pixel_to_robot import PixelToRobot, plane_from_base_table_z
    except ImportError as error:
        raise FatalError(f"视觉坐标转换模块加载失败：{error}") from error

    try:
        from task.state_store import StateStore
        from task.tube_task_manager import TaskContext, TubeTaskManager
    except ImportError as error:
        raise FatalError(f"任务层尚未实现，无法组装流水线：{error}") from error

    try:
        from vision.rack_detector import RackDetector
        from vision.tube_detector import TubeDetector
    except ImportError as error:
        raise FatalError(f"视觉检测门面尚未实现，无法组装流水线：{error}") from error

    try:
        from robot.gripper_controller import GripperController
        from robot.motion_controller import MotionController
        from robot.nova2_client import Nova2Client
    except ImportError as error:
        raise FatalError(f"机器人模块加载失败：{error}") from error

    client = Nova2Client(
        config.robot.ip,
        config.robot.port,
        config.robot.feedback_port,
    )
    motion = MotionController(client)
    gripper = GripperController(client, config.gripper)

    plane = plane_from_base_table_z(
        calibration.T_base_camera_mm,
        config.calibration.table_z_mm,
    )
    pixel_to_robot = PixelToRobot(
        (calibration.K, calibration.D),
        calibration.T_base_camera_mm,
        plane,
    )

    tube_detector = TubeDetector(camera)
    rack_detector = RackDetector(camera)
    state_store = StateStore(_resolve_path(DEFAULT_STATE_PATH))

    context = TaskContext(
        tube_detector=tube_detector,
        rack_detector=rack_detector,
        pixel_to_robot=pixel_to_robot,
        motion=motion,
        gripper=gripper,
        state_store=state_store,
        config=config,
    )
    manager = TubeTaskManager(context)

    return Pipeline(
        client=client,
        manager=manager,
        camera=camera,
        config=config,
    )


def _create_camera(config: AppConfig) -> CameraSource:
    """通过相机驱动工厂创建数据源。

    Args:
        config: AppConfig

    Returns:
        CameraSource

    Raises:
        FatalError: 相机驱动尚未实现
    """

    try:
        from camera.camera import create_camera
    except ImportError as error:
        raise FatalError(
            "相机驱动 camera/camera.py 尚未实现，无法创建相机数据源"
        ) from error

    return create_camera(config.camera)


# ============================================================
# 6. 日志与流程入口
# ============================================================

def setup_logging(log_level: int = logging.INFO) -> Path:
    """配置控制台与文件日志。

    Args:
        log_level: 日志级别

    Returns:
        日志文件路径
    """

    log_dir = _ROOT / DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_main.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError as error:
        root_logger.warning("日志文件创建失败，仅输出到控制台：%s", error)
    else:
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return log_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Nova 2 试管抓取项目入口")

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径，默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--tube-count",
        type=int,
        default=None,
        help="本次任务试管数量；缺省使用配置值",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只加载并校验配置与标定，不连接硬件",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别，默认 INFO",
    )

    return parser.parse_args(argv)


def _log_summary(config: AppConfig, calibration: CalibrationBundle) -> None:
    """输出配置与标定摘要。"""

    logger.info(
        "机器人：%s:%s feedback=%s speed=%s%%",
        config.robot.ip,
        config.robot.port,
        config.robot.feedback_port,
        config.robot.velocity_ratio_pct,
    )
    logger.info(
        "相机：%s，分辨率 %sx%s",
        config.camera.intrinsics_file,
        config.camera.width,
        config.camera.height,
    )
    logger.info("相机内参重投影 RMS：%.4f pixel", calibration.rms_px)
    logger.info(
        "T_base_camera 平移：[%.3f, %.3f, %.3f] mm",
        calibration.T_base_camera_mm[0, 3],
        calibration.T_base_camera_mm[1, 3],
        calibration.T_base_camera_mm[2, 3],
    )
    logger.info("rack_model：%s", calibration.rack_model)
    logger.info(
        "任务：试管数=%s，接近高度=%.1f mm，插入速度=%s%%",
        config.task.tube_count,
        config.task.approach_height_mm,
        config.task.insert_velocity_ratio_pct,
    )


def main(argv: list[str] | None = None) -> int:
    """程序主入口。

    Args:
        argv: 命令行参数；None 时读取 sys.argv

    Returns:
        进程退出码
    """

    args = _parse_args(argv)
    log_level = getattr(logging, args.log_level)
    log_path = setup_logging(log_level)
    logger.info("日志文件：%s", log_path)

    try:
        config = load_config(args.config)
        calibration = load_calibration(config)
    except FatalError as error:
        logger.error("启动失败：%s", error)
        return EXIT_FATAL
    except Nova2ProjectError as error:
        logger.error("启动异常：%s", error)
        return EXIT_FATAL if not error.recoverable else EXIT_RECOVERABLE

    logger.info("配置与标定加载完成")
    _log_summary(config, calibration)

    if args.dry_run:
        logger.info("dry-run 完成，未连接硬件")
        return EXIT_OK

    if args.tube_count is not None and args.tube_count <= 0:
        logger.error("--tube-count 必须为正整数，当前为 %s", args.tube_count)
        return EXIT_FATAL

    pipeline = None

    try:
        pipeline = build_pipeline(
            config,
            calibration,
            camera=_create_camera(config),
        )
        pipeline.connect()
        report = pipeline.run(tube_count=args.tube_count)
    except FatalError as error:
        logger.error("致命错误：%s", error)
        if pipeline is not None:
            pipeline.emergency_stop()
        return EXIT_FATAL
    except RecoverableError as error:
        logger.error("可恢复错误：%s", error)
        return EXIT_RECOVERABLE
    except Nova2ProjectError as error:
        logger.error("任务异常：%s", error)
        if pipeline is not None:
            pipeline.emergency_stop()
        return EXIT_FATAL if not error.recoverable else EXIT_RECOVERABLE
    except KeyboardInterrupt:
        logger.warning("收到用户中断，执行急停")
        if pipeline is not None:
            pipeline.emergency_stop()
        return EXIT_INTERRUPTED
    finally:
        if pipeline is not None:
            pipeline.close()

    if report is None:
        logger.error("任务层未返回报告")
        return EXIT_FATAL

    total = getattr(report, "total", None)
    success = getattr(report, "success", None)

    logger.info("任务完成：total=%s success=%s", total, success)

    if total is not None and success is not None and success < total:
        return EXIT_RECOVERABLE

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
