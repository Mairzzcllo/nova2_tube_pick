# 共享类型与异常

本文件承接根 `AGENTS.md` 的类型与异常约束。跨层共享类型计划放在 `common/types.py`，项目异常计划放在 `common/errors.py`；`common` 不得依赖 `robot`/`vision`/`task`。

## 共享类型（计划新增 `common/types.py`）

字段为最小契约，可扩展，但已列字段不得改名：

```python
Point3D(x_mm, y_mm, z_mm)
Pose(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)   # 固定轴 XYZ；roll/pitch/yaw 同义
Plane(a, b, c, d, frame="camera")                 # a*x + b*y + c*z + d = 0
TubeDetection(tube_id, center_u_px, center_v_px, axis_u, axis_v, angle_deg, confidence)
HolePixel(hole_id, u_px, v_px)
RackDetection(valid, reason, center_u_px, center_v_px, angle_deg, holes: list[HolePixel])
HolePose(hole_id, pose_base: Pose)
TubeGeometry(diameter_mm, length_mm, grasp_offset_mm)
TransformVerifyReport(n_points, per_point_errors_mm, mean_error_mm, max_error_mm, passed, reason="")
ReprojectionReport(n_views, rms_px, max_error_px, per_view_errors_px, passed, reason="")
HandEyeReport(sample_count, success_count, translation_std_mm, rotation_std_deg, reprojection_rms_px, passed)
RackCalibrationReport(mean_error_mm, max_error_mm, passed)
TaskState(tube_id, status, hole_id, timestamp, attempts, last_error="")
TaskReport(total, success, failures, states, final_state, elapsed_s)
GraspStatus = Enum(GRASPED, NOT_GRASPED, UNKNOWN)
```

`TaskState.status` 取值：`detected` / `picked` / `inserted` / `failed`。`TaskReport.success` 为成功数量，`failures` 为失败列表，`states` 为 `dict[str, TaskState]`。

模块私有类（`StateStore`、`RackModel`）放各自模块，不强行放进 `common`。

## Pose 与坐标约定

- `Pose` 位置单位 mm，姿态单位 deg。
- 项目内部统一为固定轴 XYZ（roll-pitch-yaw）。
- TCP/User 坐标系由 `nova2_client` 固定，上层不得切换。
- Nova 2 实际欧拉角约定接入 SDK 时验证；若与项目约定不同，只在 `nova2_client` 转换，禁止散落到其他模块。
- 齐次变换 `T_base_camera` 方向为相机 -> 基座；旋转在左上 3x3，平移在右上 3x1。
- 相机内参 `K` 单位是 pixel，畸变 `D` 无量纲，与长度单位无关；`square_size` 的单位决定 `T` 的平移单位。

## 异常层级（计划新增 `common/errors.py`）

- `Nova2ProjectError(recoverable: bool)`：项目异常基类。
- `RecoverableError` / `FatalError`：可恢复与致命两类**语义类别**，不是业务异常的唯一父类。
- 业务异常（`CalibrationError`、`VisionError`、`MotionError`、`GripperError`、`TaskError`）继承 `Nova2ProjectError`，在抛出点设置 `recoverable`：
  - `CalibrationError`：标定文件缺失/过期 -> `recoverable=False`；样本不足/单次检测失败 -> `recoverable=True`。
  - `VisionError`：未检测到目标 -> `recoverable=True`；标定缺失/图像非法 -> `recoverable=False`。
  - `MotionError`：单次运动超时 -> `recoverable=True`；限位/未使能/急停/碰撞 -> `recoverable=False`。
  - `GripperError`：通信失败 -> `recoverable=False`；无反馈不是异常，返回 `GraspStatus.UNKNOWN`。
- 禁止使用裸 `Exception`。
- 任务层捕获 `Nova2ProjectError` 后读取 `recoverable`：可恢复进入 RECOVER 并落盘；致命错误停止并保存状态后抛出。
- **不要采用**“`CalibrationError < RecoverableError`、`MotionError < FatalError`”这种硬编码继承；同一业务异常在不同场景可恢复性不同。

## 其它相关类型

- `CameraSource` 协议：`read() -> np.ndarray`，计划定义在 `common/types.py`；具体相机驱动（建议 `camera/camera.py`）尚未实现，接入前不得假设品牌。
- `AppConfig`：`config/robot_config.yaml` 的解析结果，字段至少覆盖 robot/gripper/camera/calibration/task/acceptance。
- `CalibrationBundle`：至少包含 `K`、`D`、`T_base_camera_mm`、`T_base_rack_mm`、`RackModel`、加载时间。
