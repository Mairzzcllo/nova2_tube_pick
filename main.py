"""Nova 2 试管抓取项目入口（占位）。

职责：
    加载配置，初始化相机/视觉、Nova 2 客户端、运动与夹爪控制器，
    构建 TubeTaskManager 并启动作业流程。

运行：
    python main.py

说明：
    当前仅提供占位流程，各模块实现完成后在此串联。
"""

def load_config():
    """加载 config/robot_config.yaml 等配置（占位）。"""
    # TODO: 解析 YAML 配置
    raise NotImplementedError("配置加载尚未实现")


def build_pipeline():
    """组装视觉 + 机器人 + 任务各模块（占位）。"""
    raise NotImplementedError("项目流水线组装尚未实现")


def main() -> None:
    """程序主入口（占位）。"""
    # TODO: pipeline = build_pipeline(); pipeline.run(tube_count=3)
    raise NotImplementedError("主流程尚未实现")


if __name__ == "__main__":
    main()
