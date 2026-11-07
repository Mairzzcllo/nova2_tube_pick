"""机器人姿态采集模块

功能：
    1. 通过注入的 Nova2Client 读取机械臂实际末端位姿（mm/deg）
    2. 通过注入的相机数据源采集棋盘格图像
    3. 检测棋盘格，成功后将图像与对应位姿成对落盘
    4. 生成 data/handeye/poses.json，供 handeye_calibration 使用

使用方式：
    交互式手动采集：操作员示教机械臂到不同姿态，按 Enter 采集，输入 q 结束。
    命令行入口统一使用 python -m calibration.pose_recorder。

坐标与单位：
    位姿位置 mm、姿态 deg，固定轴 XYZ（roll-pitch-yaw）。
    TCP/User 坐标系由 Nova2Client 固定，本模块不做单位或欧拉角转换。
    Nova 2 实际欧拉角约定接入 SDK 时应在 nova2_client 中验证并转换。
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

# 兼容 python -m 与直接执行文件两种方式，确保能导入仓库内模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibration.camera_calibration import detect_chessboard
from common.errors import CalibrationError, FatalError, Nova2ProjectError
from common.types import CameraSource, Pose


# ============================================================
# 1. 默认采集参数
# ============================================================

# 默认输出目录（内部生成 images/ 与 poses.json）
DEFAULT_OUTPUT_DIR = "data/handeye"

# 默认有效样本数量
DEFAULT_COUNT = 15

# 棋盘格内角点数量：(列数, 行数)
DEFAULT_PATTERN_SIZE = (9, 6)

# 默认机器人配置
DEFAULT_CONFIG_PATH = "config/robot_config.yaml"

# 默认相机索引（临时 UVC 方案）
DEFAULT_CAMERA_INDEX = 0

# 等待机械臂停止的超时时间，s
DEFAULT_STOP_TIMEOUT_S = 5.0

# 失败尝试上限系数：最多允许 count * 该系数的采集尝试，防止无限循环
FAILED_ATTEMPT_FACTOR = 5

# 运行日志目录（锚定仓库根目录，避免受当前工作目录影响）
LOG_DIR = os.path.join(_ROOT, "data", "logs")


# ============================================================
# 2. 日志
# ============================================================

def setup_logger() -> logging.Logger:
    """创建姿态采集日志记录器。

    Returns:
        同时输出到控制台与 data/logs/YYYYmmdd_HHMMSS_pose_recorder.log 的 Logger
    """

    logger = logging.getLogger("pose_recorder")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(
        LOG_DIR,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pose_recorder.log"
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
# 3. 位姿序列化
# ============================================================

def pose_to_dict(pose: Pose) -> dict:
    """将 Pose 转换为 JSON 可写入的字典（mm/deg）。

    Args:
        pose: 末端位姿

    Returns:
        包含 x_mm/y_mm/z_mm/rx_deg/ry_deg/rz_deg 的字典

    Raises:
        FatalError: 位姿类型不符合契约
    """

    if not isinstance(pose, Pose):
        raise FatalError(
            f"机器人返回的位姿类型非法：{type(pose).__name__}，"
            "预期 common.types.Pose"
        )

    return {
        "x_mm": float(pose.x_mm),
        "y_mm": float(pose.y_mm),
        "z_mm": float(pose.z_mm),
        "rx_deg": float(pose.rx_deg),
        "ry_deg": float(pose.ry_deg),
        "rz_deg": float(pose.rz_deg),
    }


# ============================================================
# 4. 输出目录准备
# ============================================================

def clear_output_dir(output_dir: str) -> None:
    """清空既有采集产物（显式覆盖时调用）。

    Args:
        output_dir: 采集输出目录
    """

    output_path = Path(output_dir)
    pose_file = output_path / "poses.json"
    images_dir = output_path / "images"

    if pose_file.exists():
        pose_file.unlink()

    if images_dir.exists():
        shutil.rmtree(images_dir)


# ============================================================
# 5. 姿态采集器
# ============================================================

class PoseRecorder:
    """机器人位姿与棋盘格图像同步采集器。

    Args:
        robot: Nova2Client，需提供 get_pose() 与 wait_until_stopped(timeout_s)
        camera: CameraSource，需提供 read()
        output_dir: 输出目录，内部生成 images/ 与 poses.json
        pattern_size: 棋盘格内角点数量 (列数, 行数)

    说明：
        本类只负责采集与落盘，不控制机械臂运动。
        采集图像与机器人位姿必须成对保存，检测失败时跳过且不得写伪数据。
    """

    def __init__(
        self,
        robot: Any,
        camera: CameraSource,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    ):
        if robot is None:
            raise ValueError("robot 不能为空")

        if camera is None:
            raise ValueError("camera 不能为空")

        if not output_dir:
            raise ValueError("output_dir 不能为空")

        cols, rows = pattern_size

        if cols < 2 or rows < 2:
            raise ValueError("棋盘格行列内角点数量必须至少为 2")

        self.robot = robot
        self.camera = camera
        self.output_dir = Path(output_dir)
        self.pattern_size = (int(cols), int(rows))

        self.images_dir = self.output_dir / "images"
        self.pose_file = self.output_dir / "poses.json"

        self.samples: list[dict] = []
        self._created_at = ""

        self.logger = setup_logger()

    # --------------------------------------------------------
    # 5.1 采集主流程
    # --------------------------------------------------------

    def record(self, count: int = DEFAULT_COUNT) -> None:
        """采集指定数量的有效样本并保存。

        Args:
            count: 需要采集的有效样本数量

        Raises:
            ValueError: count 非法
            CalibrationError: 输出目录冲突、相机读取失败或有效样本不足
        """

        if count < 1:
            raise ValueError("count 必须大于 0")

        self._prepare_output_dir()

        self.samples = []
        self._created_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        max_attempts = count * FAILED_ATTEMPT_FACTOR

        self.logger.info("=" * 60)
        self.logger.info("开始机械臂位姿采集")
        self.logger.info(f"目标样本数量：{count}")
        self.logger.info(f"棋盘格内角点：{self.pattern_size}")
        self.logger.info(f"输出目录：{self.output_dir}")
        self.logger.info("=" * 60)

        attempts = 0

        while len(self.samples) < count:

            if attempts >= max_attempts:
                self.logger.warning(
                    f"采集尝试次数达到上限：{max_attempts}"
                )
                break

            if not self._wait_for_trigger():
                self.logger.info("操作员结束采集")
                break

            attempts += 1

            if self._capture_once():
                self.logger.info(
                    f"[进度] {len(self.samples)} / {count}"
                )

        if len(self.samples) < count:
            raise CalibrationError(
                f"有效样本不足：{len(self.samples)} / {count}，"
                "请增加不同姿态后重试",
                recoverable=True,
            )

        self.logger.info("=" * 60)
        self.logger.info(
            f"位姿采集完成：{len(self.samples)} 个样本"
        )
        self.logger.info(f"位姿文件：{self.pose_file}")
        self.logger.info(f"图像目录：{self.images_dir}")
        self.logger.info("=" * 60)

    # --------------------------------------------------------
    # 5.2 单次采集
    # --------------------------------------------------------

    def _capture_once(self) -> bool:
        """完成一次“停止检查 -> 读位姿 -> 读图 -> 检测 -> 落盘”。

        Returns:
            True 表示成功保存一个样本；False 表示本次跳过
        """

        if not self.robot.wait_until_stopped(
            timeout_s=DEFAULT_STOP_TIMEOUT_S
        ):
            self.logger.warning("[跳过] 机械臂未在超时时间内停止")
            return False

        pose = self.robot.get_pose()
        pose_dict = pose_to_dict(pose)

        frame = self.camera.read()

        if frame is None or getattr(frame, "size", 0) == 0:
            raise CalibrationError(
                "相机读取失败，无法采集标定图像",
                recoverable=False,
            )

        success, _corners, _gray = detect_chessboard(
            frame,
            self.pattern_size,
        )

        if not success:
            self.logger.warning("[跳过] 当前图像未检测到棋盘格")
            return False

        sample_id = len(self.samples)
        image_rel_path = f"images/{sample_id:03d}.png"
        image_path = self.output_dir / image_rel_path

        if not cv2.imwrite(str(image_path), frame):
            raise CalibrationError(
                f"标定图像写入失败：{image_path}",
                recoverable=False,
            )

        sample = {
            "sample_id": sample_id,
            "timestamp": datetime.now().timestamp(),
            "robot_pose": pose_dict,
            "image_path": image_rel_path,
        }

        self.samples.append(sample)
        self._write_pose_file()

        self.logger.info(
            f"[成功] 样本 {sample_id:03d} 已保存："
            f"pose=({pose_dict['x_mm']:.2f}, {pose_dict['y_mm']:.2f}, "
            f"{pose_dict['z_mm']:.2f}) mm，图像={image_rel_path}"
        )

        return True

    # --------------------------------------------------------
    # 5.3 落盘
    # --------------------------------------------------------

    def _prepare_output_dir(self) -> None:
        """检查并创建输出目录，拒绝覆盖既有采集产物。

        Raises:
            CalibrationError: 已存在 poses.json 或历史图像
        """

        if self.pose_file.exists():
            raise CalibrationError(
                f"输出目录已存在位姿文件：{self.pose_file}，"
                "请更换目录或使用 --overwrite",
                recoverable=False,
            )

        if self.images_dir.exists() and any(self.images_dir.iterdir()):
            raise CalibrationError(
                f"输出目录已存在历史图像：{self.images_dir}，"
                "请更换目录或使用 --overwrite",
                recoverable=False,
            )

        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _build_payload(self) -> dict:
        """构建 poses.json 内容。

        Returns:
            包含元信息与样本列表的字典
        """

        return {
            "version": 1,
            "created_at": self._created_at,
            "pattern_size": list(self.pattern_size),
            "euler_convention": "fixed_xyz_rpy",
            "robot_pose_units": {
                "position": "mm",
                "rotation": "deg",
            },
            "samples": self.samples,
        }

    def _write_pose_file(self) -> None:
        """原子写入 poses.json，避免中途崩溃导致文件损坏。"""

        tmp_file = self.pose_file.with_suffix(".json.tmp")

        with open(tmp_file, "w", encoding="utf-8") as file:
            json.dump(
                self._build_payload(),
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(tmp_file, self.pose_file)

    # --------------------------------------------------------
    # 5.4 采集触发
    # --------------------------------------------------------

    def _wait_for_trigger(self) -> bool:
        """等待操作员确认采集。

        Returns:
            True 表示继续采集；False 表示结束采集
        """

        try:
            text = input(
                "调整机械臂到新姿态后按 Enter 采集，输入 q 结束："
            ).strip().lower()
        except EOFError:
            return False

        return text != "q"


# ============================================================
# 6. 临时相机数据源
# ============================================================

class VideoCaptureSource:
    """基于 cv2.VideoCapture 的临时相机数据源。

    说明：
        仅用于相机驱动尚未实现阶段的手动标定联调，品牌无关。
        后续 camera/ 驱动落地后应替换为正式 CameraSource 实现。
    """

    def __init__(self, index: int = DEFAULT_CAMERA_INDEX):
        self._capture = cv2.VideoCapture(index)

        if not self._capture.isOpened():
            raise FatalError(f"无法打开相机：index={index}")

    def read(self):
        """读取一帧 BGR 图像。

        Returns:
            BGR 图像数组；读取失败返回 None
        """

        success, frame = self._capture.read()
        return frame if success else None

    def release(self) -> None:
        """释放相机资源。"""

        self._capture.release()


# ============================================================
# 7. 命令行入口
# ============================================================

def load_robot_config(config_path: str) -> dict:
    """读取机器人配置文件。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        配置字典；文件不存在时返回空字典
    """

    if not os.path.isfile(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> int:
    """姿态采集命令行入口。

    Returns:
        进程退出码：0 成功，1 失败，130 用户中断
    """

    parser = argparse.ArgumentParser(
        description="Nova 2 手眼标定位姿采集工具"
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="采集输出目录（生成 images/ 与 poses.json）",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="需要采集的有效样本数量",
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
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="机器人配置文件路径",
    )

    parser.add_argument(
        "--ip",
        default=None,
        help="Nova 2 控制柜 IP，缺省读取配置文件",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Nova 2 通信端口，缺省读取配置文件",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="临时 UVC 相机索引",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中已有的采集产物",
    )

    args = parser.parse_args()

    logger = setup_logger()

    if args.overwrite:
        clear_output_dir(args.output_dir)

    config = load_robot_config(args.config)
    robot_config = config.get("robot", {})

    ip = args.ip if args.ip is not None else robot_config.get("ip")
    port = args.port if args.port is not None else robot_config.get("port")

    if not ip or port is None:
        logger.error("缺少机器人连接参数：请检查配置文件或 --ip/--port")
        return 1

    from robot.nova2_client import Nova2Client

    camera = None
    client = Nova2Client(ip=ip, port=int(port))

    try:
        client.connect()
        camera = VideoCaptureSource(args.camera_index)

        recorder = PoseRecorder(
            robot=client,
            camera=camera,
            output_dir=args.output_dir,
            pattern_size=(args.cols, args.rows),
        )

        recorder.record(count=args.count)

    except Nova2ProjectError as error:
        logger.error(f"采集失败：{error}")
        return 1

    except NotImplementedError as error:
        logger.error(f"依赖模块尚未实现：{error}")
        return 1

    except KeyboardInterrupt:
        logger.warning("操作员中断采集")
        return 130

    finally:
        if camera is not None:
            camera.release()

        try:
            client.disconnect()
        except NotImplementedError:
            logger.warning("Nova2Client.disconnect 尚未实现，跳过断开")

    return 0


if __name__ == "__main__":
    sys.exit(main())
