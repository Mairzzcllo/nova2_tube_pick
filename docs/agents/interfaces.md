# 公开接口契约（目标契约）

本文件承接根 `AGENTS.md` 的完整接口契约。**下表是实现目标，不是当前 stub 的现状。** 当前占位文件签名不一致，实现时必须先迁移签名，再写实现，并同步 `main.py`。

## 接口设计原则

- **构造依赖与调用参数分离**：硬件、标定、存储等长期依赖通过构造函数注入；方法只接收本次任务数据。禁止把 robot/camera/calibration 藏进函数内部全局变量。
- **类型与单位显式**：公开接口必须有类型注解；长度 `_mm`、角度 `_deg`、速度 `_pct` 后缀不得省略。跨层共享类型放 `common/types.py`（计划新增），项目异常放 `common/errors.py`（计划新增）；`common` 不得依赖 `robot`/`vision`/`task`。
- **Pose 约定**：`Pose` 位置 mm、姿态 deg，项目内部统一为固定轴 XYZ（roll-pitch-yaw）。TCP/User 坐标系由 `nova2_client` 固定，上层不得切换；Nova 2 实际欧拉角约定接入 SDK 时验证，若不同只在 `nova2_client` 转换。
- **异常分类**：`Nova2ProjectError(recoverable: bool)` 为基类；`RecoverableError` 表示可恢复/可重试（未检测到目标、单次超时、夹爪无反馈），`FatalError` 表示必须停机人工处理（未使能、标定缺失、SDK 断连、急停/碰撞）。禁止用裸 `Exception`。
- **当前 stub 不是契约**：占位文件里的旧签名是过渡态。实现时必须先按本节更新签名，再写实现，并同步 `main.py`。

## 类型引用

共享类型字段以 [types_and_errors.md](types_and_errors.md) 为准；本文件只定义接口。接口涉及：`Point3D`、`Pose`、`Plane`、`TubeDetection`、`HolePixel`、`RackDetection`、`HolePose`、`TubeGeometry`、`GraspStatus`、`TaskState`、`TaskReport`、`HandEyeReport`、`TransformVerifyReport`、`ReprojectionReport`、`RackModel`、`AppConfig`、`StateStore`、`TaskContext`。

## 标定层

| 符号 / 签名 | 输入 | 输出 / 单位 | 异常 |
|---|---|---|---|
| `calibrate_camera(image_dir, output_path, pattern_size=(9,6), square_size_mm=25.0, min_views=10, debug_dir=None) -> tuple[np.ndarray, np.ndarray, float]` | 图片目录；`square_size_mm` | `(K, D, rms_px)`；写 npz | `CalibrationError` |
| `PoseRecorder(robot, camera, output_dir, pattern_size=(9,6))`；`record(count=15) -> None` | 机器人、相机、棋盘格参数 | `images/*.png` + `poses.json`（mm/deg） | 检测失败跳过并记录；不足 `count` 抛 `CalibrationError`；不得写伪数据 |
| `calibrate_handeye(pose_file, image_dir, intrinsics_path, output_path, pattern_size=(9,6), square_size_mm=25.0) -> HandEyeReport` | 位姿 JSON、图片、K/D | `T_base_camera.npy`（平移 mm）+ 报告 | 样本不足/求解失败 `CalibrationError` |
| `verify_transform(T_base_camera, points_camera_mm, points_robot_mm, mean_threshold_mm=3.0, max_threshold_mm=5.0) -> TransformVerifyReport` | 4x4 变换 + 两组 Nx3 点（mm） | 逐点误差、均值、最大值、PASS/FAIL | 点数/形状不一致 `ValueError`；矩阵非法 `CalibrationError` |
| `verify_reprojection(intrinsics_path, image_dir, pattern_size=(9,6), square_size_mm=25.0, max_threshold_px=2.0) -> ReprojectionReport` | K/D + 棋盘格图片 | 重投影误差报告 | 图片不足/检测失败 `CalibrationError` |

说明：**重投影验证与 T_base_camera 验证分开**。`verify_transform` 不接收 `reproj_threshold_px`，避免职责混淆。

## 视觉层

| 符号 / 签名 | 输入 | 输出 / 单位 | 异常 |
|---|---|---|---|
| `detect_tubes(image) -> list[TubeDetection]` | BGR 图 | `TubeDetection` 列表；无目标返回 `[]` | `VisionError`；不得伪造 |
| `detect_rack(image) -> RackDetection` | BGR 图 | `valid=False` + `reason` 表示失败 | `VisionError`；不得返回假孔位 |
| `PixelToRobot(intrinsics, T_base_camera_mm, plane)` | K/D、T_BC、相机系桌面平面 | — | 标定缺失 `FatalError` |
| `PixelToRobot.pixel_to_point(u, v, depth_mm=None) -> Point3D` | 像素；`depth_mm` 为相机系 Z；`None` 时与 `plane` 求交 | 基座坐标 mm | 射线与平面不相交/深度非法 `ValueError` |
| `PixelToRobot.tube_to_grasp_pose(tube, tube_geometry) -> Pose` | `TubeDetection` + 管径/管长/夹持偏移 | 基座抓取位姿 mm/deg | 几何非法 `ValueError` |
| `PixelToRobot.hole_pixel_to_pose(hole: HolePixel, rack_model: RackModel) -> HolePose` | 孔位像素 + `RackModel` | 基座孔位 mm/deg | 标定缺失 `FatalError`；孔位不匹配 `ValueError` |

说明：`pixel_to_point` 只返回点，不返回姿态；抓取姿态由 `tube_to_grasp_pose` 负责。深度来源必须显式：外部深度给 `depth_mm`，否则用桌面平面。`hole_pixel_to_pose` 用于关闭“孔位像素 -> HolePose”的链路缺口；孔位朝向来自 `RackModel`，不在任务层计算矩阵。

## 机器人层

| 符号 / 签名 | 输入 | 输出 / 单位 | 异常 |
|---|---|---|---|
| `Nova2Client(ip, port)`；`connect()` / `disconnect()` / `get_pose() -> Pose` / `get_joint_angles() -> tuple[float, ...]` / `move_j(pose, velocity_ratio_pct)` / `move_l(pose, velocity_ratio_pct)` / `wait_until_stopped(timeout_s) -> bool` / `stop()` | 连接参数；位姿 mm/deg | 对外统一 mm/deg；关节角 deg | SDK 异常转 `FatalError`/`RecoverableError`；单位与欧拉角转换只在此层 |
| `MotionController(client)`；`move_to(pose, velocity_ratio_pct=50)` / `move_linear(pose, velocity_ratio_pct=10)` / `wait_arrival(timeout_s=10.0) -> bool` | `Pose` mm/deg | 到位 bool | `wait_arrival` 返回 bool；`move_to`/`move_linear` 超时抛 `RecoverableError`；限位/未使能抛 `FatalError` |
| `GripperController(client)`；`open()` / `close()` / `is_grasped() -> GraspStatus` | — | `GRASPED` / `NOT_GRASPED` / `UNKNOWN` | 无反馈返回 `UNKNOWN`，不得假装成功；通信失败 `GripperError` |

说明：`velocity_ratio_pct` 为厂商速度百分比 `1..100`，不是 mm/s。若要物理速度，必须在 `Nova2Client` 做映射并实测标定，业务层不得自行换算。

`Nova2Client.move_j` 参数语义**待 SDK 确认**：UML 提议 `move_j(joint_angles_deg, velocity_ratio_pct)`，当前契约保留 `move_j(pose, velocity_ratio_pct)`。若 Nova 2 SDK 的 MovJ 只接受关节角，则在 `Nova2Client` 内完成 Pose/关节角转换，或新增 `move_joints(...)`，不改变 `MotionController` 的 `Pose` 接口。

## 任务层

| 符号 / 签名 | 输入 | 输出 / 单位 | 异常 |
|---|---|---|---|
| `StateStore(path)`；`load() -> dict[str, TaskState]` / `save(states) -> None` | JSON 路径 | 状态字典 | 文件损坏 `RecoverableError`；写失败 `FatalError` |
| `TaskContext(tube_detector: TubeDetector, rack_detector: RackDetector, pixel_to_robot: PixelToRobot, motion: MotionController, gripper: GripperController, state_store: StateStore, config: AppConfig)` | 上述依赖 | — | — |
| `TubeTaskManager(context)`；`pick_tube(tube: TubeDetection) -> bool` / `insert_tube(tube: TubeDetection, hole: HolePose) -> bool` / `run(tube_count=3) -> TaskReport` | 检测结果、孔位 | 成功 bool / 报告 | 可恢复失败返回 False 并落盘；致命异常保存后抛出；不清空已完成状态 |

## 任务 FSM 契约

状态：`INIT`、`DETECT`、`SELECT`、`CALCULATE`、`APPROACH`、`GRASP`、`LIFT`、`REORIENT`、`ALIGN`、`INSERT`、`RELEASE`、`VERIFY`、`RECOVER`、`FATAL`、`FINISH`。

主流程：

```
INIT -> DETECT -> SELECT -> CALCULATE -> APPROACH -> GRASP -> LIFT
     -> REORIENT -> ALIGN -> INSERT -> RELEASE -> VERIFY
VERIFY -> DETECT   还有未归位试管
VERIFY -> FINISH   全部完成
```

异常分支：

- `GRASP -> RECOVER`：`NOT_GRASPED` / `UNKNOWN`。
- `DETECT -> RECOVER`：未检测到目标；默认中止当前循环并记录，**不得猜坐标继续运动**。
- `RECOVER -> DETECT`：仅在显式重试策略/人工确认后允许，必须有重试上限，禁止无限循环。
- `RECOVER -> FATAL`：不可恢复错误。
- `INIT -> FATAL`：标定缺失或机器人未使能。

持久化：每次状态迁移后通过 `StateStore.save()` 落盘 `TaskState`；恢复时 `load()` 并跳过已完成试管，禁止重复抓取。

修改任何公开签名时，必须同步更新本文件与 `main.py`。
