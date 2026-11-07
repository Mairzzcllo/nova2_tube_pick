# main.py 组装契约

本文件承接根 `AGENTS.md` 的入口组装契约。

```python
load_config(path="config/robot_config.yaml") -> AppConfig
load_calibration(config) -> CalibrationBundle   # K, D, T_base_camera_mm, T_base_rack_mm, RackModel
build_pipeline(config, calibration, camera: CameraSource | None = None) -> Pipeline
Pipeline.run(tube_count=3) -> TaskReport
```

## 职责与约束

- `load_calibration` 必须校验文件存在、npz `rms` 字段、修改时间；缺失/过期抛 `FatalError`，不得加载占位矩阵。
- 阶段 8 前 `T_base_rack` 可能不存在：此时 `CalibrationBundle.rack_model` 允许为 `None`，只能执行到阶段 7 的抓取测试；任何插入任务必须要求 `rack_model` 非空，否则抛 `FatalError`。
- `build_pipeline` 负责创建 `Nova2Client`、`MotionController`、`GripperController`、视觉模块、`PixelToRobot`、`StateStore`、`TubeTaskManager`；`main.py` 不写识别/矩阵/运动算法。
- `CameraSource` 协议（`read() -> np.ndarray`）计划定义在 `common/types.py`；具体相机驱动（建议 `camera/camera.py`）尚未实现，接入前不得假设品牌。
- `main()` 捕获 `Nova2ProjectError`：`FatalError` 立即 `stop()` 并记录；`RecoverableError` 记录并落盘状态。

## 配置与加载

- 配置来源：`config/robot_config.yaml`，长度统一 mm、角度 deg、速度用 `velocity_ratio_pct`。
- 标定产物：`config/camera_intrinsics.npz`、`config/T_base_camera.npy`、`config/T_base_rack.npy`；均已被 `.gitignore` 忽略，不要提交。
- 标定加载失败、文件过期、字段缺失：抛 `FatalError` 并中止，不得使用占位矩阵继续运行。
