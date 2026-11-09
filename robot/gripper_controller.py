"""夹爪控制模块。

职责：
    1. 封装夹爪张开、闭合动作
    2. 读取夹持反馈并转换为 `GraspStatus`
    3. 把底层 IO 异常转换为 `GripperError`
    4. 在夹爪协议未确定时安全降级，不假装夹持成功

说明：
    - 本模块不直接接触 Nova 2 SDK，所有 IO 读写经 `Nova2Client`。
    - 夹爪硬件协议尚未确定，`GripperConfig.control_mode` 默认为 `unknown`。
      协议确定前，开合动作抛出 `GripperError`，`is_grasped()` 返回
      `GraspStatus.UNKNOWN`。
    - 无反馈不是异常；只有通信/解析失败才抛 `GripperError`。
"""

import logging

from common.errors import GripperError, Nova2ProjectError
from common.types import GraspStatus, GripperConfig

logger = logging.getLogger(__name__)


# ============================================================
# 1. 常量与默认参数
# ============================================================

# 已识别的控制方式
MODE_TOOL_DO = "tool_do"
MODE_DUAL_DO = "dual_do"
MODE_SERIAL = "serial"
MODE_UNKNOWN = "unknown"

# 当前已实现的控制方式；其余值只保留占位
IMPLEMENTED_MODES = frozenset({MODE_TOOL_DO})

# 已识别但尚未实现/未配置的控制方式
PENDING_MODES = frozenset({MODE_DUAL_DO, MODE_SERIAL, MODE_UNKNOWN})

# DO 电平取值
ACTIVE_LEVEL_ON = 1


# ============================================================
# 2. 夹爪控制器
# ============================================================

class GripperController:
    """夹爪控制器。

    Args:
        client: `Nova2Client` 实例，提供末端 DO/DI 与工具电源接口
        config: 夹爪配置；None 时使用 `GripperConfig()`（`unknown` 模式）

    说明：
        上层接口固定为 `open()` / `close()` / `is_grasped()`，
        硬件协议确定后只需在 `_resolve_output()` 下补分支。
    """

    def __init__(self, client, config: GripperConfig | None = None):
        if config is not None and not isinstance(config, GripperConfig):
            raise ValueError(
                f"config 必须是 GripperConfig，当前为 {type(config).__name__}"
            )

        self.client = client
        self.config = config if config is not None else GripperConfig()
        self._tool_power_on = False

    # --------------------------------------------------------
    # 2.1 开合动作
    # --------------------------------------------------------

    def open(self) -> None:
        """张开夹爪。

        Raises:
            GripperError: 协议未配置/未实现、IO 编号缺失或通信失败
        """

        self._check_supported("张开夹爪")
        self._ensure_tool_power()
        index, level = self._resolve_output("open")
        self._set_output(index, level, "张开夹爪")

    def close(self) -> None:
        """闭合夹爪。

        Raises:
            GripperError: 协议未配置/未实现、IO 编号缺失或通信失败
        """

        self._check_supported("闭合夹爪")
        self._ensure_tool_power()
        index, level = self._resolve_output("close")
        self._set_output(index, level, "闭合夹爪")

    # --------------------------------------------------------
    # 2.2 夹持反馈
    # --------------------------------------------------------

    def is_grasped(self) -> GraspStatus:
        """读取夹持反馈状态。

        Returns:
            `GraspStatus.GRASPED` / `NOT_GRASPED`；无反馈通道返回 `UNKNOWN`

        Raises:
            GripperError: 读取反馈时通信或解析失败

        说明：
            - 无 `feedback_di_index` 时直接返回 `UNKNOWN`，不是异常。
            - DI 有效电平的物理含义（夹住/未夹住）以实机接线为准，
              当前假设 `feedback_active_level` 对应“已夹住”。
        """

        index = self.config.feedback_di_index

        if index is None:
            logger.debug("夹爪未配置反馈 DI，返回 UNKNOWN")
            return GraspStatus.UNKNOWN

        try:
            raw = self.client.read_tool_di(index)
        except Nova2ProjectError as exc:
            raise GripperError(
                f"读取夹爪反馈失败：{exc}", recoverable=exc.recoverable
            ) from exc

        active = self.config.feedback_active_level == ACTIVE_LEVEL_ON
        return GraspStatus.GRASPED if raw == active else GraspStatus.NOT_GRASPED

    # --------------------------------------------------------
    # 2.3 内部辅助
    # --------------------------------------------------------

    def _check_supported(self, action: str) -> None:
        """检查当前控制方式是否已实现。

        Raises:
            GripperError: 模式未知、未配置或尚未实现
        """

        mode = self.config.control_mode

        if mode in IMPLEMENTED_MODES:
            return

        if mode in PENDING_MODES:
            raise GripperError(
                f"{action}失败：夹爪控制方式 '{mode}' 尚未实现或未配置，"
                f"请先确认硬件协议",
                recoverable=False,
            )

        raise GripperError(
            f"{action}失败：未知夹爪控制方式 '{mode}'",
            recoverable=False,
        )

    def _resolve_output(self, action: str) -> tuple[int, bool]:
        """解析动作对应的 DO 编号与电平。

        Args:
            action: `open` 或 `close`

        Returns:
            `(do_index, on)`，`on=True` 表示输出 ON

        Raises:
            GripperError: 动作非法或缺少 DO 编号

        说明：
            - `tool_do`：单个 DO 电平切换，开/合使用相反电平。
            - `dual_do` / `serial`：协议未确定，暂不实现，见下方 TODO。
        """

        if action not in ("open", "close"):
            raise ValueError(f"未知夹爪动作：{action!r}")

        mode = self.config.control_mode
        active = self.config.do_active_level == ACTIVE_LEVEL_ON

        if mode == MODE_TOOL_DO:
            index = self.config.open_do_index
            if index is None:
                raise GripperError(
                    f"{action}失败：tool_do 模式缺少 open_do_index",
                    recoverable=False,
                )
            return index, active if action == "open" else not active

        # TODO: dual_do 使用 open_do_index / close_do_index 分别输出脉冲，
        #       具体脉冲宽度与间隔需实机确认后实现。
        # TODO: serial 需新增串口/总线驱动，协议与帧格式待厂商确认。
        raise GripperError(
            f"{action}失败：夹爪控制方式 '{mode}' 尚未实现",
            recoverable=False,
        )

    def _ensure_tool_power(self) -> None:
        """按配置打开末端工具电源。

        Raises:
            GripperError: 上电失败

        说明：
            标志位只用于避免重复上电；若实机会掉电，需要重新评估。
        """

        if not self.config.tool_power_required or self._tool_power_on:
            return

        try:
            self.client.set_tool_power(True)
        except Nova2ProjectError as exc:
            raise GripperError(
                f"夹爪工具上电失败：{exc}", recoverable=exc.recoverable
            ) from exc

        self._tool_power_on = True

    def _set_output(self, index: int, on: bool, action: str) -> None:
        """写入末端 DO 并转换底层异常。

        Raises:
            GripperError: 通信失败或指令返回错误
        """

        try:
            self.client.set_tool_do(index, on)
        except Nova2ProjectError as exc:
            raise GripperError(
                f"{action}失败：{exc}", recoverable=exc.recoverable
            ) from exc

        logger.info("%s：DO%d -> %s", action, index, "ON" if on else "OFF")


# ============================================================
# 3. 入口
# ============================================================

def main() -> None:
    """打印当前夹爪配置与状态（空载自检）。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = GripperConfig()
    controller = GripperController(client=None, config=config)

    logger.info("夹爪控制方式：%s", config.control_mode)
    logger.info("夹持状态：%s", controller.is_grasped())
    logger.info(
        "夹爪协议未确定，开合动作将抛出 GripperError；"
        "实机测试见 debug/test_gripper_controller_hw.py"
    )


if __name__ == "__main__":
    main()
