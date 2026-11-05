"""运动控制模块（占位）。

职责：
    在 Nova2Client 之上封装笛卡尔/关节空间运动、直线运动与到位判断，
    为任务层提供 move_to / approach / insert 等高层接口。

验收（阶段 6、9）：
    空载到达指定位置；单支低速插入并释放。

说明：
    当前仅提供占位骨架，具体运动规划与安全限位待实现。
"""


class MotionController:
    """机械臂运动控制器（占位）。"""

    def __init__(self, client):
        self.client = client

    def move_to(self, pose, speed: int = 50) -> None:
        """移动到目标位姿（占位）。"""
        raise NotImplementedError("位姿运动尚未实现")

    def move_linear(self, pose, speed: int = 10) -> None:
        """以直线方式移动到目标位姿（占位）。"""
        raise NotImplementedError("直线运动尚未实现")

    def wait_arrival(self, timeout: float = 10.0) -> bool:
        """等待运动到位（占位）。"""
        raise NotImplementedError("到位检测尚未实现")


if __name__ == "__main__":
    # TODO: 使用空载测试点位验证 move_to()
    pass
