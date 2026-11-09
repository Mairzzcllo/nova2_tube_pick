"""main.py 离线测试。

覆盖：
    1. 配置加载与字段映射（speed -> velocity_ratio_pct 等）
    2. 标定文件加载与聚合
    3. 标定文件早于配置文件修改时间时拒绝加载
    4. 标定文件超过 max_age_days 时拒绝加载
    5. 配置缺失、夹爪模式非法时抛 FatalError
    6. build_pipeline 在未提供相机时抛 FatalError

说明：
    不连接真实硬件；使用临时目录与合成标定数据验证入口逻辑。

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_main.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import FatalError
from main import build_pipeline, load_calibration, load_config


# ============================================================
# 1. 测试数据构造
# ============================================================

def _write_config(
    root: Path,
    max_age_days=30,
    speed=70,
    control_mode="unknown",
) -> tuple[Path, Path, Path, Path]:
    """生成临时配置与合成标定文件。

    Returns:
        (config_path, intrinsics_path, T_base_camera_path, T_base_rack_path)
    """

    config_path = root / "robot_config.yaml"
    intrinsics_path = root / "camera_intrinsics.npz"
    t_base_camera_path = root / "T_base_camera.npy"
    t_base_rack_path = root / "T_base_rack.npy"

    config = {
        "robot": {
            "ip": "192.168.1.100",
            "port": 29999,
            "feedback_port": None,
            "speed": speed,
        },
        "gripper": {
            "type": "parallel",
            "open_width_mm": 40.0,
            "close_width_mm": 10.0,
            "control_mode": control_mode,
            "open_do_index": None,
            "close_do_index": None,
            "do_active_level": 1,
            "feedback_di_index": None,
            "feedback_active_level": 1,
            "feedback_timeout_s": 2.0,
            "tool_power_required": False,
        },
        "camera": {
            "intrinsics_file": str(intrinsics_path),
            "width": 1280,
            "height": 720,
        },
        "calibration": {
            "T_base_camera_file": str(t_base_camera_path),
            "T_base_rack_file": str(t_base_rack_path),
            "table_z_mm": 0.0,
            "max_age_days": max_age_days,
        },
        "task": {
            "tube_count": 3,
            "approach_height_mm": 100.0,
            "insert_speed": 10,
        },
    }

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    camera_matrix = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)

    np.savez(
        intrinsics_path,
        K=camera_matrix,
        D=dist_coeffs,
        rms=np.array(0.5, dtype=np.float64),
        image_width=np.array(1280, dtype=np.int64),
        image_height=np.array(720, dtype=np.int64),
    )
    np.save(t_base_camera_path, np.eye(4, dtype=np.float64))

    return config_path, intrinsics_path, t_base_camera_path, t_base_rack_path


# ============================================================
# 2. 配置加载测试
# ============================================================

def test_load_config_mapping():
    """验证配置字段映射与默认值。"""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config_path, intrinsics_path, _, _ = _write_config(root)

        config = load_config(str(config_path))

        assert config.robot.velocity_ratio_pct == 70
        assert config.robot.feedback_port is None
        assert config.camera.intrinsics_file == str(intrinsics_path.resolve())
        assert config.task.insert_velocity_ratio_pct == 10
        assert config.calibration.max_age_days == 30.0
        assert config.acceptance.camera_rms_px == 1.0

    print("[PASS] 配置字段映射")


def test_load_config_missing_file():
    """验证配置文件缺失时抛 FatalError。"""

    try:
        load_config("/nonexistent/robot_config.yaml")
        raise AssertionError("缺失配置应抛 FatalError")
    except FatalError:
        pass

    print("[PASS] 配置缺失拒绝加载")


def test_load_config_invalid_gripper_mode():
    """验证非法夹爪模式被拒绝。"""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config_path, _, _, _ = _write_config(root, control_mode="bogus")

        try:
            load_config(str(config_path))
            raise AssertionError("非法夹爪模式应抛 FatalError")
        except FatalError:
            pass

    print("[PASS] 非法夹爪模式拒绝加载")


# ============================================================
# 3. 标定加载测试
# ============================================================

def test_load_calibration_valid():
    """验证合成标定文件可正常加载。"""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config_path, _, _, _ = _write_config(root)

        config = load_config(str(config_path))
        bundle = load_calibration(config)

        assert bundle.K.shape == (3, 3)
        assert bundle.D.size == 5
        assert bundle.rms_px == 0.5
        assert bundle.T_base_camera_mm.shape == (4, 4)
        assert np.allclose(bundle.T_base_camera_mm, np.eye(4))
        assert bundle.T_base_rack_mm is None
        assert bundle.rack_model is None

    print("[PASS] 标定加载与聚合")


def test_load_calibration_older_than_config():
    """验证标定文件早于配置文件修改时间时拒绝加载。"""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config_path, _, _, _ = _write_config(root)

        future_s = time.time() + 1000.0
        os.utime(config_path, (future_s, future_s))

        config = load_config(str(config_path))

        try:
            load_calibration(config)
            raise AssertionError("过期标定应抛 FatalError")
        except FatalError:
            pass

    print("[PASS] 标定早于配置文件拒绝加载")


def test_load_calibration_exceeds_max_age():
    """验证标定文件超过 max_age_days 时拒绝加载。"""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config_path, intrinsics_path, t_base_camera_path, _ = _write_config(
            root,
            max_age_days=0.0001,
        )

        now_s = time.time()
        os.utime(config_path, (now_s - 2000.0, now_s - 2000.0))
        os.utime(intrinsics_path, (now_s - 1000.0, now_s - 1000.0))
        os.utime(t_base_camera_path, (now_s - 1000.0, now_s - 1000.0))

        config = load_config(str(config_path))

        try:
            load_calibration(config)
            raise AssertionError("超龄标定应抛 FatalError")
        except FatalError:
            pass

    print("[PASS] 标定超龄拒绝加载")


# ============================================================
# 4. 流水线组装测试
# ============================================================

def test_build_pipeline_requires_camera():
    """验证未提供相机数据源时 build_pipeline 抛 FatalError。"""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config_path, _, _, _ = _write_config(root)

        config = load_config(str(config_path))
        bundle = load_calibration(config)

        try:
            build_pipeline(config, bundle, camera=None)
            raise AssertionError("camera=None 应抛 FatalError")
        except FatalError:
            pass

    print("[PASS] 缺少相机时拒绝组装")


# ============================================================
# 5. 入口
# ============================================================

def main() -> None:
    """依次执行全部离线测试。"""

    test_load_config_mapping()
    test_load_config_missing_file()
    test_load_config_invalid_gripper_mode()
    test_load_calibration_valid()
    test_load_calibration_older_than_config()
    test_load_calibration_exceeds_max_age()
    test_build_pipeline_requires_camera()
    print("main.py 离线测试全部通过")


if __name__ == "__main__":
    main()
