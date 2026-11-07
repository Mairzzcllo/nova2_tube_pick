# T_base_rack 标定与加载契约

本文件承接根 `AGENTS.md` 的 `T_base_rack` 详细契约。

## 现状

- `config/T_base_rack.npy` 没有生成脚本；`T_BR` 当前不可用，不要假设已标定。
- 阶段 8 前 `rack_model` 可为 `None`；任何插入任务必须要求 `rack_model` 非空，否则抛 `FatalError`。

## 坐标系约定（mm）

- 原点：`hole_1` 孔中心。
- X 轴：三孔排列方向。
- Y 轴：架平面内垂直 X 轴的方向。
- Z 轴：架平面法向；实现时固定朝上/朝外并写入注释。

## 孔位局部坐标

- `hole_1=[0,0,0]`
- `hole_2=[pitch,0,0]`
- `hole_3=[2*pitch,0,0]`

`pitch` 为实测孔间距，单位 mm。

## 生成方式（二选一，计划 `calibration/rack_calibration.py`）

1. 示教：`calibrate_rack(taught_holes_base_mm, hole_layout_mm, output_path) -> RackCalibrationReport`。机器人触碰/对准三个孔中心，记录基座位姿并拟合 `T_BR`。
2. 视觉：`rack_detector` 孔位像素 + `T_base_camera` 反投影到架平面后拟合。

生成结果保存为 `config/T_base_rack.npy`，平移单位 mm。

## 加载方式（计划 `calibration/rack_model.py`）

```python
load_rack_model(T_base_rack_path, hole_layout_mm) -> RackModel
RackModel.hole_pose(hole_id) -> HolePose
```

- `HolePose` 在机器人基坐标系下，位置 mm、姿态 deg。
- 任务层只通过 `RackModel` 取孔位，**禁止直接读 `T_base_rack.npy` 或手写孔位**。

## 验证

- 用 `T_BR` 预测三孔基座坐标，与实际示教点比较。
- 平均误差 `<= 3 mm`，最大误差 `<= 5 mm`；超出则重标。
