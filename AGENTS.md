# AGENTS.md

Nova 2 机械臂试管抓取/归位项目。核心思想：标定、视觉、坐标转换、机器人控制、任务调度相互独立，任一环节可替换。

本文件是每次会话必须加载的项目级指令。详细阶段计划、接口契约、类型定义按需读取 `docs/agents/*.md`。

**当前 stub 不是契约。** 实现占位模块前必须先按目标契约更新签名，再写实现，并同步 `main.py`。

## 1. 核心分层原则

- 只有 `robot/nova2_client.py` 可以接触 Nova 2 SDK；其他模块不得直接调用 SDK。
- `task` 不调 SDK、不算矩阵、不做识别；`vision` 不控制机械臂。
- `common` 不得依赖 `robot` / `vision` / `task`。
- 硬件、标定、存储等长期依赖通过构造函数注入；方法只接收本次任务数据。
- 禁止裸 `Exception`。`RecoverableError` 表示可重试/可恢复；`FatalError` 表示必须停机人工处理。完整异常语义见 [types_and_errors.md](docs/agents/types_and_errors.md)。
- `config/*.npy`、`config/*.npz` 是标定产物，已被 `.gitignore` 忽略，不要提交。

## 2. 目录与职责

| 路径 | 职责 |
|---|---|
| `config/` | 运行配置与标定产物：`camera_intrinsics.npz`、`T_base_camera.npy`、`T_base_rack.npy`、`robot_config.yaml` |
| `calibration/` | 相机标定、位姿采集、手眼标定、标定验证；阶段 8 增加试管架标定 |
| `vision/` | `tube_detector`、`rack_detector`、`pixel_to_robot` |
| `robot/` | `nova2_client`（唯一 SDK 边界）、`motion_controller`、`gripper_controller` |
| `task/` | `tube_task_manager`（FSM 调度，不写算法） |
| `common/` | 跨层类型与异常（计划新增：`types.py`、`errors.py`） |
| `camera/` | 相机驱动（计划新增，品牌待定） |
| `third_party/` | 厂商 SDK；Nova 2 SDK 位于 `third_party/dobot_sdk/dobot_api.py` |
| `debug/` | 独立验证脚本，脚本内用 `assert`；非 pytest 用例 |
| `main.py` | 入口：加载配置与标定 -> 组装各模块 -> 启动作业 |
| `docs/agents/` | 按需读取的详细文档 |

## 3. 关键数据流（简版）

```
camera_calibration -> K/D
pose_recorder -> data/handeye/{images,poses.json}
handeye_calibration -> T_base_camera
calibration_verify -> PASS/FAIL 报告
detectors + T_base_camera + 桌面平面 -> pixel_to_robot -> Point3D / Pose
rack_calibration(计划) -> T_base_rack -> RackModel -> HolePose
tube_task_manager -> motion_controller -> nova2_client -> Nova 2
tube_task_manager -> gripper_controller -> 夹爪
状态 -> data/task_state.json；日志 -> data/logs/
```

完整 10 步流水线见 [dataflow.md](docs/agents/dataflow.md)。`T_base_camera` 的生产者是 `handeye_calibration`，`camera_calibration` 只生产 `K/D`。

## 4. 环境与统一命令

- 统一环境：`~/miniconda3/envs/Dobot-Sim`（Python 3.11.17）。
- 默认 `python` / conda base（3.13）**没有 cv2**，不要用它跑本项目。
- 依赖：`requirements.txt`；新机器执行 `pip install -r requirements.txt`。
- Nova 2 SDK 位于 `third_party/dobot_sdk/dobot_api.py`，版本与连接方式待确认；只允许在 `robot/nova2_client.py` 引用，并在该层完成单位转换。

```bash
# 相机内参标定
~/miniconda3/envs/Dobot-Sim/bin/python calibration/camera_calibration.py \
  --image-dir data/camera_calibration \
  --output config/camera_intrinsics.npz \
  --cols 9 --rows 6 --square-size 25

# 全部离线冒烟测试（仓库根目录；当前只有相机标定一项）
~/miniconda3/envs/Dobot-Sim/bin/python debug/run_smoke_tests.py

# 最小静态检查（语法/导入级）
~/miniconda3/envs/Dobot-Sim/bin/python -m compileall -q calibration vision robot task main.py
```

- 新增模块必须补 `debug/test_<module>.py`；需要硬件的测试命名为 `debug/test_<module>_hw.py`，默认不跑。
- 仓库未配置 ruff/black/mypy/CI，**不要臆造 lint 命令或自行引入工具**。

## 5. 单位、坐标与接口原则

- 全项目长度统一 **mm**，角度统一 **deg**；变量/配置键带单位后缀（`_mm`、`_deg`、`_pct`）。
- 相机内参 `K` 的单位是 **pixel**，畸变 `D` 无量纲，与长度单位无关；`square_size` 的单位决定 `T` 的平移单位。
- 速度接口统一用 `velocity_ratio_pct`，范围 `1..100`，**不是 mm/s**。物理速度必须在 `Nova2Client` 映射并实测后才能暴露；业务层不得自行换算。
- `T_base_camera` 方向为 **相机 -> 基座**；旋转在左上 3x3，平移在右上 3x1。
- `Pose` 位置 mm、姿态 deg，项目内部统一为固定轴 XYZ（roll-pitch-yaw）。TCP/User/欧拉角约定集中封装，禁止散落；Nova 2 实际欧拉角约定接入 SDK 时验证，若不同只在 `nova2_client` 转换。
- 公开接口签名与单位见 [interfaces.md](docs/agents/interfaces.md)。修改签名必须同步该文件与 `main.py`。
- **`T_base_rack` 当前不可用**，不得假设已标定。阶段 8 前 `rack_model` 可为 `None`；任何插入任务必须要求 `rack_model` 非空，否则抛 `FatalError`。

## 6. 安全与失败处理

| 失败模式 | 核心处理 |
|---|---|
| 连接/使能失败、急停、限位、碰撞 | 立即停止，记录日志，等待人工复位；不得自动继续 |
| 视觉未检测到试管/孔位 | 中止当前循环并记录；禁止猜坐标继续运动 |
| 夹爪无反馈 | `is_grasped` 返回 `GraspStatus.UNKNOWN`；抬升后视觉复检，不得假装成功 |
| 运动超时 | 抛 `RecoverableError`，任务进入 RECOVER，不盲目重发 |
| 标定文件缺失/过期 | 抛 `FatalError` 并中止；校验文件存在、`rms` 字段、修改时间，不加载占位矩阵 |
| 任务中断后重启 | 读取 `data/task_state.json`，跳过已完成试管；禁止从头重跑重复抓取 |
| 插入失败 | 退回安全点，记录孔位/试管状态，人工确认后再试 |

- 抓取/插入必须先经过安全接近点，禁止从当前位置直线冲向目标；插入用低速。
- 每个状态迁移后落盘 `data/task_state.json`：试管 id、状态（detected/picked/inserted）、占用孔位、时间戳。
- 运行日志：`data/logs/YYYYmmdd_HHMMSS_<module>.log`；调试产物：`data/debug/`，**不提交**。

## 7. 当前实现状态与占位

- `calibration/camera_calibration.py` 已实现（CLI + 合成图自测）。
- `calibration/pose_recorder.py`、`handeye_calibration.py`、`calibration_verify.py`、`vision/*.py`、`robot/*.py`、`task/tube_task_manager.py`、`main.py` 仍为占位。
- `common/types.py`、`common/errors.py`、`calibration/rack_calibration.py`、`calibration/rack_model.py` 为计划新增；`camera/` 驱动尚未实现。
- 完整状态表与约定见 [status.md](docs/agents/status.md)。

## 8. 按需参考

| 文档 | 内容 |
|---|---|
| [roadmap.md](docs/agents/roadmap.md) | 10 阶段计划、完成定义、验收阈值及依据 |
| [interfaces.md](docs/agents/interfaces.md) | 标定/视觉/机器人/任务层完整接口契约 |
| [types_and_errors.md](docs/agents/types_and_errors.md) | 共享类型字段、`GraspStatus`、异常层级 |
| [rack_calibration.md](docs/agents/rack_calibration.md) | `T_base_rack` 坐标系、生成与加载契约 |
| [main_assembly.md](docs/agents/main_assembly.md) | `main.py` 组装、标定加载校验、错误处理 |
| [status.md](docs/agents/status.md) | 当前实现状态、日志、代码约定 |
| [dataflow.md](docs/agents/dataflow.md) | 完整 10 步数据流 |

## 9. 未定义事项

- 提交规范、分支策略、CI、许可证当前未定义；不要自行补充或引入。
- ruff/black/mypy 未配置；不要臆造 lint 命令。
- Nova 2 SDK 版本与连接方式、相机品牌与驱动接口待确认。

## 10. Python 代码风格

本项目采用 **工程化模块化 Python 风格（Engineering-Oriented Modular Python Style）**。

编写和修改代码时，参考项目现有的模块，保持以下风格：

1. **功能分区**：使用带编号的分隔注释，将文件按功能划分为清晰的区域。
2. **函数拆分**：按照功能职责拆分函数，避免单个函数承载过多逻辑。
3. **中文文档**：文件头说明模块职责和功能，核心函数使用 `Args`、`Returns` 格式的 Docstring。
4. **配置集中**：默认参数集中定义在文件顶部，使用常量管理，避免散落魔法数字。
5. **防御式检查**：对关键输入、数据尺寸、文件状态和计算条件进行检查，使用明确的异常类型。
6. **流程分离**：主流程负责组织执行顺序，具体算法由独立函数实现，命令行入口统一使用 `main()`。
7. **过程可见**：在关键执行节点输出中文状态信息、计算结果和必要的中间指标。

### 执行要求

- 新代码优先沿用现有文件的组织方式和注释习惯。
- 修改代码时保持原有接口、数据格式和项目架构。
- 不为追求简洁而牺牲可读性，也不为遵循模板而过度拆分函数。
- 不机械复制示例代码结构，根据实际模块职责灵活组织。

**核心原则：结构清晰、职责明确、过程可追踪、便于维护。**
