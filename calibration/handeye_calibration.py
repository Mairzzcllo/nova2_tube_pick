"""手眼标定模块（占位）。

职责：
    根据机器人末端位姿与标定板在相机中的位姿，求解
    base 与 camera 之间的变换矩阵 ``T_base_camera``，
    结果保存为 ``config/T_base_camera.npy``。

验收（阶段 3）：
    得到稳定的 T_BC 变换。

说明：
    当前仅提供占位骨架，具体 AX=XB 求解逻辑待实现。
"""


def calibrate_handeye(pose_file: str, output_path: str) -> None:
    """执行手眼标定并保存 T_base_camera（占位）。"""
    raise NotImplementedError("手眼标定尚未实现")


if __name__ == "__main__":
    # TODO: 解析参数，调用 calibrate_handeye()
    pass
