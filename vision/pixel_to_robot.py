"""像素坐标到机器人坐标转换模块（占位）。

职责：
    结合相机内参与手眼标定矩阵 T_base_camera，将图像中的像素坐标
    转换为机器人基坐标系下的三维坐标。

说明：
    本模块是视觉与运动控制之间的桥梁，供 task / robot 层调用。
    当前仅提供占位骨架，具体反投影与坐标系变换逻辑待实现。
"""


def pixel_to_robot(u: float, v: float, depth: float):
    """将像素坐标 (u, v) 与深度转换为机器人基坐标（占位）。"""
    raise NotImplementedError("像素到机器人坐标转换尚未实现")


if __name__ == "__main__":
    # TODO: 加载内参与外参，调用 pixel_to_robot() 做自检
    pass
