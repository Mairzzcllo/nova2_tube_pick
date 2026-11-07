# 完整数据流

本文件承接根 `AGENTS.md` 的完整 10 步数据流。

```
1. camera_calibration.py
   -> config/camera_intrinsics.npz (K, D)
2. pose_recorder.py (机器人多姿态 + 同步棋盘格图)
   -> data/handeye/images/*.png + data/handeye/poses.json
3. handeye_calibration.py (读 1 + 2)
   -> config/T_base_camera.npy (T_BC)
4. calibration_verify.py (读 1 + 3 + 验证点)
   -> PASS/FAIL 报告
5. tube_detector / rack_detector (图像)
   -> 像素坐标 + 长轴/孔位
6. pixel_to_robot (1 + 3 + 桌面平面 + 5 的像素结果)
   -> 机器人基坐标下的点/抓取位姿 (mm, deg)
7. rack_calibration.py (计划新增，阶段 8)
   -> config/T_base_rack.npy (T_BR)
8. rack_model.py (计划新增)
   -> HolePose (基座 mm/deg)
9. tube_task_manager (FSM)
   -> motion_controller -> nova2_client -> Nova 2
   -> gripper_controller -> 夹爪
10. 任务状态
   -> data/task_state.json + data/logs/
```

注意：`handeye_calibration` 才是 `T_base_camera` 的生产者；`camera_calibration` 只生产 `K/D`。

## 各环节数据归属

| 数据 | 产生模块 | 使用模块 |
|---|---|---|
| 相机内参 `K`、`D` | `camera_calibration` | `handeye_calibration`、`vision` |
| 机器人姿态 | `pose_recorder` | `handeye_calibration` |
| 棋盘格相机位姿 | `handeye_calibration` | 手眼矩阵求解 |
| 相机到基座矩阵 `T_base_camera` | `handeye_calibration` | `pixel_to_robot` |
| 试管像素位置 | `tube_detector` | `pixel_to_robot` |
| 试管架孔位像素 | `rack_detector` | `pixel_to_robot` |
| 试管机器人位姿 | `pixel_to_robot` | `tube_task_manager` |
| 机器人运动指令 | `motion_controller` | `nova2_client` |
| 夹爪动作 | `tube_task_manager` | `gripper_controller` |
| 任务状态 | `tube_task_manager` | `main.py` / 日志 |
