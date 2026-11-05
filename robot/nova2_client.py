"""Nova 2 通信客户端（占位）。

职责：
    封装与 Nova 2 机械臂的底层通信（连接、位姿读写、状态查询），
    供上层 motion/gripper 控制器调用。

说明：
    当前仅提供占位骨架，具体协议与 SDK 调用待实现。
"""


class Nova2Client:
    """Nova 2 机械臂通信客户端（占位）。"""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def connect(self) -> None:
        """建立与机械臂的连接（占位）。"""
        raise NotImplementedError("连接 Nova 2 尚未实现")

    def disconnect(self) -> None:
        """断开连接（占位）。"""
        raise NotImplementedError("断开 Nova 2 尚未实现")

    def get_pose(self):
        """读取当前末端位姿（占位）。"""
        raise NotImplementedError("读取位姿尚未实现")


if __name__ == "__main__":
    # TODO: 从 config 读取参数并连接设 备做连通性测试
    pass
