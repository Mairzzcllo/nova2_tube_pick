# AGENTS.md

Nova 2 机械臂试管抓取/归位项目。核心思想：标定、视觉、坐标转换、机器人控制、任务调度相互独立，任一环节可替换而不影响其他模块。

详细设计文档见 `ai_output.md`；阶段计划见 `note.md`（两者已被 `.git/info/exclude` 本地忽略，勿提交）。

## 目录与职责

```
config/        运行配置与标定产物
  camera_intrinsics.npz   K, D, image_width/height, rms, per_view_errors
  T_base_camera.npy       T_BC：相机 -> 机器人基座（4x4 齐次）
  T_base_rack.npy         T_BR：试管架 -> 机器人基座（4x4 齐次）
  robot_config.yaml       robot/gripper/camera/calibration/task 参数
calibration/   标定链路（camera_calibration -> pose_recorder -> handeye_calibration -> calibration_verify）
vision/        tube_detector / rack_detector / pixel_to_robot
robot/         nova2_client（仅此层可碰 SDK）/ motion_controller / gripper_controller
task/          tube_task_manager（FSM 调度，不写算法）
debug/         独立验证脚本（非 pytest）
main.py        入口：加载配置与标定 -> 组装各模块 -> 启动作业
```

数据流：`camera_calibration` 产 K/D → `handeye_calibration`；`pose_recorder` 产机器人位姿 → `handeye_calibration` → `T_base_camera` → `pixel_to_robot`；`tube_detector`/`rack_detector` 产像素结果 → `pixel_to_robot` → `tube_task_manager` → `motion_controller` → `nova2_client`；夹爪动作由任务层调用 `gripper_controller`。

## 当前实现状态

| 文件 | 状态 |
|---|---|
| `calibration/camera_calibration.py` | 已实现（可运行 CLI + 合成图自测） |
| `calibration/pose_recorder.py` | 占位（NotImplementedError） |
| `calibration/handeye_calibration.py` | 占位 |
| `calibration/calibration_verify.py` | 占位 |
| `vision/*.py` | 占位 |
| `robot/*.py` | 占位 |
| `task/tube_task_manager.py` | 占位 |
| `main.py` | 占位 |

开发优先级按 `note.md` 阶段 1→4：先标定并验证 `T_base_camera`，再做视觉、机器人控制，最后任务集成。

## 环境与命令

默认 `python` 是 conda base (3.13)，**未安装 cv2**。带 OpenCV 的环境：

```bash
# 任选其一（均已装 cv2/numpy/scipy/yaml）
~/miniconda3/envs/Dobot-Sim/bin/python
~/miniconda3/envs/leisaac/bin/python
~/miniconda3/envs/lerobot/bin/python
```

相机内参标定（需 `data/camera_calibration/` 下 20~30 张棋盘格图，默认 9x6、方格 25 mm）：

```bash
~/miniconda3/envs/Dobot-Sim/bin/python calibration/camera_calibration.py \
  --image-dir data/camera_calibration \
  --output config/camera_intrinsics.npz \
  --cols 9 --rows 6 --square-size 25
```

合成图自测（必须在仓库根目录运行，脚本内用相对路径 `calibration/camera_calibration.py`）：

```bash
~/miniconda3/envs/Dobot-Sim/bin/python debug/test_camera_calibration.py
```

无 requirements.txt / pytest；验证方式是脚本内 `assert`。改动后至少跑上面的自测，并确保其它模块仍可 import。

## 约定与注意事项

- 代码注释、docstring、打印信息均用中文；变量/函数用 snake_case，类用 PascalCase。
- 保持分层：`task` 不调 SDK、不算矩阵、不做识别；只有 `nova2_client` 接触 Nova 2 SDK；`vision` 不控制机械臂。
- **单位不统一**：标定链路使用 **mm**（内参、`square_size`、`T_*` 平移）；`robot_config.yaml` 使用 **m**（`open_width`、`approach_height` 等）。新增代码务必显式注明单位并在边界处转换。
- 齐次变换约定：`T_base_camera` 表示相机 -> 基座；矩阵为 4x4，旋转 R 在上左 3x3，平移 t 在右上 3x1。
- 欧拉角/位姿约定（TCP、User 坐标系、欧拉角顺序）必须集中封装，不要散落在各模块。
- 抓取/插入必须先经过安全接近点，禁止从当前位置直接直线冲向目标；插入用低速。
- 任务需持久化状态（哪支已抓/已插/占哪个孔），异常后不可盲目从头重跑。
- `config/T_base_camera.npy`、`config/T_base_rack.npy`、`config/camera_intrinsics.npz` 已被 `.gitignore` 忽略，标定产物不要提交。
- 占位模块实现时保留现有公开函数/类签名，避免破坏 `main.py` 与任务层的调用契约。
