"""机器人姿态采集模块（占位）。

职责：
    连接 Nova 2，按标定流程记录多组机器人实际末端位姿，
    供手眼标定使用。

验收（阶段 2）：
    机器人实际姿态记录正确。

说明：
    当前仅提供占位骨架，具体通信与落盘逻辑待实现。
"""


def record_poses(output_file: str, count: int = 15) -> None:
    """采集指定数量的机器人位姿并保存（占位）。"""
    raise NotImplementedError("姿态采集尚未实现")


if __name__ == "__main__":
    # TODO: 解析参数，调用 record_poses()
    pass
