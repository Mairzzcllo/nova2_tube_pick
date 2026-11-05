"""标定与坐标转换验证模块（占位）。

职责：
    检验手眼标定结果，将相机坐标系下的若干桌面测试点转换到机器人
    基坐标系，与机器人实际到达位置比对。

验收（阶段 4）：
    桌面多个点转换准确。

说明：
    当前仅提供占位骨架，具体误差统计逻辑待实现。
"""


def verify_transform(points_camera: list, points_robot: list) -> float:
    """计算坐标转换误差并返回平均误差（占位）。"""
    raise NotImplementedError("标定验证尚未实现")


if __name__ == "__main__":
    # TODO: 加载标定结果与测试点，调用 verify_transform()
    pass
