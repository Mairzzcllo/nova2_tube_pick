# 当前实现状态与代码约定

本文件承接根 `AGENTS.md` 的实现状态、日志与代码约定细节。

## 实现状态

| 文件 | 状态 |
|---|---|
| `calibration/camera_calibration.py` | 已实现（CLI + 合成图自测） |
| `calibration/pose_recorder.py` | 占位（NotImplementedError） |
| `calibration/handeye_calibration.py` | 占位 |
| `calibration/calibration_verify.py` | 占位 |
| `vision/*.py` | 占位 |
| `robot/*.py` | 占位 |
| `task/tube_task_manager.py` | 占位 |
| `common/types.py`、`common/errors.py` | 计划新增 |
| `calibration/rack_calibration.py`、`calibration/rack_model.py` | 计划新增（阶段 8） |
| `third_party/dobot_sdk/dobot_api.py` | 已迁入（厂商 SDK，版本待确认） |
| `main.py` | 占位 |

## 代码约定

- 代码注释、docstring、打印信息用中文；变量/函数 snake_case，类 PascalCase。
- 保持分层：`task` 不调 SDK、不算矩阵、不做识别；只有 `nova2_client` 接触 SDK；`vision` 不控制机械臂。
- `common` 不得依赖 `robot` / `vision` / `task`。
- 实现占位模块时必须按 [interfaces.md](interfaces.md) 的目标契约更新签名，再写实现；当前 stub 签名仅作过渡参考。不要在 `main.py` 里写算法。
- `config/T_base_camera.npy`、`config/T_base_rack.npy`、`config/camera_intrinsics.npz` 已被 `.gitignore` 忽略，标定产物不要提交。

## 日志与调试产物

- 运行日志：`data/logs/YYYYmmdd_HHMMSS_<module>.log`。
- 调试图像/中间结果：`data/debug/`，**不提交**。
- 任务状态：`data/task_state.json`，**不提交**。
- 以上路径已在 `.gitignore` 忽略；运行时优先 `logging`，调试脚本可 `print`。

## 未定义事项

- 提交规范、分支策略、CI、许可证当前未定义；不要自行补充或引入。
- ruff/black/mypy 未配置；不要臆造 lint 命令。
- Nova 2 SDK 位于 `third_party/dobot_sdk/dobot_api.py`，版本与连接方式待确认；只有 `robot/nova2_client.py` 可 import，其他模块禁止直接依赖。
- 相机品牌与驱动接口待确认。
