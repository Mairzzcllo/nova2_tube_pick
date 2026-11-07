"""项目异常层级。

功能：
    1. 定义项目异常基类 Nova2ProjectError
    2. 提供可恢复/致命两类语义异常 RecoverableError、FatalError
    3. 提供标定层业务异常 CalibrationError
    4. 提供视觉层业务异常 VisionError

说明：
    禁止使用裸 Exception。
    recoverable 表示调用方是否可以重试/恢复，不等同于异常继承关系。
"""


# ============================================================
# 1. 项目异常基类
# ============================================================

class Nova2ProjectError(Exception):
    """项目异常基类。

    Args:
        message: 异常描述
        recoverable: True 表示可重试/可恢复；False 表示必须停机人工处理
    """

    def __init__(self, message: str, recoverable: bool):
        super().__init__(message)
        self.recoverable = recoverable


# ============================================================
# 2. 语义异常
# ============================================================

class RecoverableError(Nova2ProjectError):
    """可恢复异常：允许进入 RECOVER 并按策略重试。"""

    def __init__(self, message: str):
        super().__init__(message, recoverable=True)


class FatalError(Nova2ProjectError):
    """致命异常：必须停机并等待人工处理。"""

    def __init__(self, message: str):
        super().__init__(message, recoverable=False)


# ============================================================
# 3. 业务异常
# ============================================================

class CalibrationError(Nova2ProjectError):
    """标定业务异常。

    Args:
        message: 异常描述
        recoverable: 样本不足/单次检测失败为 True；文件缺失/数据非法为 False
    """

    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message, recoverable=recoverable)


class VisionError(Nova2ProjectError):
    """视觉业务异常。

    Args:
        message: 异常描述
        recoverable: 未检测到目标/单帧失败为 True；图像非法/标定缺失为 False
    """

    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message, recoverable=recoverable)
