"""Nova 2 通信客户端。

功能：
    1. 封装 Nova 2 SDK 的连接、断开与自动使能
    2. 提供末端位姿、关节角读取
    3. 提供关节/直线运动指令（速度比例 1..100）
    4. 通过反馈口高频数据判断机械臂是否停止
    5. 提供末端 IO 读写（DO/DI 与工具电源），供夹爪层使用
    6. 把 SDK 原始返回与异常转换为项目统一类型和异常

说明：
    - 本模块是唯一允许 import `third_party/dobot_sdk/dobot_api.py` 的边界。
    - 对外位姿统一为 `Pose`（位置 mm、姿态 deg，固定轴 XYZ）。
    - TCP/User 坐标系在本层固定，上层不得切换。
    - Nova 2 实际欧拉角约定尚未实机验证；若与固定轴 XYZ 不同，只在本层转换。
    - SDK 返回字符串格式、错误码表、反馈字段含义均以实机为准，不确定处已在注释中标注。
"""

import logging
import math
import re
import threading
import time
from dataclasses import dataclass

from common.errors import FatalError, RecoverableError
from common.types import Pose
from third_party.dobot_sdk.dobot_api import DobotApiDashboard, DobotApiFeedBack

logger = logging.getLogger(__name__)


# ============================================================
# 1. 常量与默认参数
# ============================================================

# Dashboard 指令口：SDK 内部仅对 29999/30004/30005 建连，指令必须走 29999
DEFAULT_DASHBOARD_PORT = 29999

# 反馈口：30004 为实时反馈端口；传 None 可关闭反馈口
DEFAULT_FEEDBACK_PORT = 30004

# 速度比例合法范围（厂商百分比，不是 mm/s）
MIN_VELOCITY_RATIO_PCT = 1
MAX_VELOCITY_RATIO_PCT = 100

# 机械臂状态码，来自 SDK RobotMode() 文档
ROBOT_MODE_INIT = 1
ROBOT_MODE_BRAKE_OPEN = 2
ROBOT_MODE_POWEROFF = 3
ROBOT_MODE_DISABLED = 4
ROBOT_MODE_ENABLE = 5
ROBOT_MODE_BACKDRIVE = 6
ROBOT_MODE_RUNNING = 7
ROBOT_MODE_SINGLE_MOVE = 8
ROBOT_MODE_ERROR = 9
ROBOT_MODE_PAUSE = 10
ROBOT_MODE_COLLISION = 11

# 运动中的状态集合
MOVING_ROBOT_MODES = (
    ROBOT_MODE_RUNNING,
    ROBOT_MODE_SINGLE_MOVE,
)

# 必须停机人工处理的状态集合
FATAL_ROBOT_MODES = (
    ROBOT_MODE_ERROR,
    ROBOT_MODE_PAUSE,
    ROBOT_MODE_COLLISION,
)

# 停止判定：状态为空闲且 TCP 速度低于该阈值，单位 mm/s
STOP_SPEED_THRESHOLD_MM_S = 1.0

# 反馈口轮询等待的检查间隔，单位 s
WAIT_POLL_INTERVAL_S = 0.02

# 反馈快照超过该时间未更新视为反馈中断，单位 s
FEEDBACK_STALE_TIMEOUT_S = 1.0

# 连接后等待第一帧反馈的超时，单位 s
FEEDBACK_FIRST_TIMEOUT_S = 2.0

# 等待使能状态切换的超时，单位 s
ENABLE_TIMEOUT_S = 5.0

# 默认运动速度比例
DEFAULT_MOVE_JOINT_VELOCITY_PCT = 50
DEFAULT_MOVE_LINEAR_VELOCITY_PCT = 10

# 默认等待停止超时，单位 s
DEFAULT_STOP_TIMEOUT_S = 10.0

# 默认等待运动启动超时，单位 s
DEFAULT_MOVE_START_TIMEOUT_S = 2.0


# ============================================================
# 2. 内部反馈快照
# ============================================================

@dataclass(frozen=True)
class _FeedbackSnapshot:
    """反馈口单帧状态快照。

    Args:
        timestamp_s: 接收时刻（monotonic），s
        robot_mode: 机器人状态码
        enable_status: 使能状态
        running_status: 运行状态
        error_status: 错误状态
        tcp_speed_mm_s: TCP 速度三分量，mm/s
        tcp_speed_norm_mm_s: TCP 速度模长，mm/s
    """

    timestamp_s: float
    robot_mode: int
    enable_status: int
    running_status: int
    error_status: int
    tcp_speed_mm_s: tuple[float, float, float]
    tcp_speed_norm_mm_s: float


# ============================================================
# 3. 客户端主体
# ============================================================

class Nova2Client:
    """Nova 2 机械臂通信客户端。

    Args:
        ip: 控制柜 IP
        port: Dashboard 指令口，默认应为 29999
        feedback_port: 反馈口；默认 30004，传 None 关闭反馈口

    说明：
        - `connect()` 会自动调用 `EnableRobot` 使能机械臂。
        - 反馈口可用时，`wait_until_stopped` 使用反馈口高频数据；
          不可用时回退到 Dashboard 状态轮询。
    """

    def __init__(
        self,
        ip: str,
        port: int,
        feedback_port: int | None = DEFAULT_FEEDBACK_PORT,
    ):
        self.ip = ip
        self.port = port
        self.feedback_port = feedback_port

        self._dashboard: DobotApiDashboard | None = None
        self._feedback: DobotApiFeedBack | None = None
        self._connected = False

        self._stop_event = threading.Event()
        self._feedback_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._snapshot: _FeedbackSnapshot | None = None
        self._feedback_error: str | None = None

    # --------------------------------------------------------
    # 3.1 连接管理
    # --------------------------------------------------------

    def connect(self) -> None:
        """建立连接并自动使能机械臂。

        Raises:
            FatalError: 端口非法、连接失败、使能失败或状态异常
        """

        if self._connected:
            return

        if self.port != DEFAULT_DASHBOARD_PORT:
            raise FatalError(
                f"Dashboard 指令口应为 {DEFAULT_DASHBOARD_PORT}，当前为 {self.port}"
            )

        if self.feedback_port is not None and self.feedback_port == self.port:
            raise FatalError("反馈口不能与 Dashboard 指令口相同")

        self._stop_event.clear()
        self._snapshot = None
        self._feedback_error = None

        self._dashboard = self._create_dashboard()
        logger.info("Nova 2 Dashboard 已连接：%s:%s", self.ip, self.port)

        try:
            if self.feedback_port is not None:
                self._start_feedback()

            self._ensure_enabled()
        except Exception:
            # 连接/使能失败时释放已建立的 socket 与反馈线程
            self.disconnect()
            raise

        self._connected = True
        logger.info("Nova 2 已连接并使能")

    def disconnect(self) -> None:
        """断开连接，幂等。"""

        self._connected = False
        self._stop_event.set()

        thread = self._feedback_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=FEEDBACK_FIRST_TIMEOUT_S)

        if thread is not None and thread.is_alive():
            logger.warning("反馈线程未在超时内退出")

        if self._feedback is not None:
            try:
                self._feedback.close()
            except Exception as exc:
                logger.warning("关闭反馈口失败：%s", exc)
            self._feedback = None

        if self._dashboard is not None:
            try:
                self._dashboard.close()
            except Exception as exc:
                logger.warning("关闭 Dashboard 失败：%s", exc)
            self._dashboard = None

        self._feedback_thread = None
        with self._lock:
            self._snapshot = None
            self._feedback_error = None

        logger.info("Nova 2 已断开")

    @property
    def connected(self) -> bool:
        """当前是否已连接并使能。"""

        return self._connected

    def _create_dashboard(self) -> DobotApiDashboard:
        """创建 Dashboard 连接并检查底层 socket 是否建立。

        Raises:
            FatalError: 连接失败
        """

        try:
            dashboard = DobotApiDashboard(self.ip, self.port)
        except Exception as exc:
            # SDK 边界：第三方异常统一转为致命错误
            raise FatalError(f"连接 Nova 2 Dashboard 失败：{exc}") from exc

        # SDK 在 socket 连接失败时只打印错误并把 socket 置 0，不会抛异常
        if not getattr(dashboard, "socket_dobot", 0):
            raise FatalError(
                f"连接 Nova 2 Dashboard 失败：{self.ip}:{self.port}"
            )

        return dashboard

    def _ensure_enabled(self) -> None:
        """查询状态并在需要时自动使能。

        Raises:
            FatalError: 处于报警/碰撞/暂停状态，或使能后未进入空闲状态
        """

        mode = self._read_robot_mode()

        if mode in FATAL_ROBOT_MODES:
            raise FatalError(f"Nova 2 处于异常状态（RobotMode={mode}），需人工处理")

        if mode in (
            ROBOT_MODE_ENABLE,
            ROBOT_MODE_RUNNING,
            ROBOT_MODE_SINGLE_MOVE,
        ):
            return

        if mode in (
            ROBOT_MODE_INIT,
            ROBOT_MODE_BRAKE_OPEN,
            ROBOT_MODE_POWEROFF,
            ROBOT_MODE_DISABLED,
        ):
            self._check_reply(self._dashboard.EnableRobot(), "使能机械臂")
            self._wait_for_robot_mode(ROBOT_MODE_ENABLE, ENABLE_TIMEOUT_S)
            return

        raise FatalError(f"Nova 2 处于无法自动使能的状态（RobotMode={mode}）")

    def _wait_for_robot_mode(self, expected_mode: int, timeout_s: float) -> None:
        """轮询直到进入期望状态。

        Raises:
            FatalError: 超时或进入异常状态
        """

        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            mode = self._read_robot_mode()

            if mode == expected_mode:
                return

            if mode in FATAL_ROBOT_MODES:
                raise FatalError(
                    f"Nova 2 在等待状态 {expected_mode} 时进入异常状态（RobotMode={mode}）"
                )

            time.sleep(WAIT_POLL_INTERVAL_S)

        raise FatalError(f"等待 Nova 2 状态 {expected_mode} 超时")

    # --------------------------------------------------------
    # 3.2 反馈口
    # --------------------------------------------------------

    def _start_feedback(self) -> None:
        """创建反馈口连接并启动后台线程。

        Raises:
            FatalError: 反馈口连接失败或未收到第一帧数据
        """

        try:
            feedback = DobotApiFeedBack(self.ip, self.feedback_port)
        except Exception as exc:
            # SDK 边界：第三方异常统一转为致命错误
            raise FatalError(f"连接 Nova 2 反馈口失败：{exc}") from exc

        if not getattr(feedback, "socket_dobot", 0):
            raise FatalError(
                f"连接 Nova 2 反馈口失败：{self.ip}:{self.feedback_port}"
            )

        self._feedback = feedback
        self._feedback_thread = threading.Thread(
            target=self._feedback_loop,
            name="nova2-feedback",
            daemon=True,
        )
        self._feedback_thread.start()

        deadline = time.monotonic() + FEEDBACK_FIRST_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._lock:
                if self._snapshot is not None:
                    logger.info("Nova 2 反馈口已就绪：%s:%s", self.ip, self.feedback_port)
                    return
                if self._feedback_error is not None:
                    raise FatalError(f"Nova 2 反馈口异常：{self._feedback_error}")
            time.sleep(WAIT_POLL_INTERVAL_S)

        raise FatalError("等待 Nova 2 第一帧反馈数据超时")

    def _feedback_loop(self) -> None:
        """后台循环读取反馈帧，直到断开或发生错误。"""

        feedback = self._feedback
        if feedback is None:
            return

        while not self._stop_event.is_set():
            try:
                data = feedback.feedBackData()
            except Exception as exc:
                # SDK 边界：`feedBackData` 会抛裸 Exception，统一记录为反馈异常
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._feedback_error = message
                logger.error("Nova 2 反馈口读取失败：%s", message)
                return

            if data is None or len(data) == 0:
                continue

            try:
                snapshot = self._build_snapshot(data[0])
            except (KeyError, TypeError, ValueError) as exc:
                message = f"反馈字段解析失败：{type(exc).__name__}: {exc}"
                with self._lock:
                    self._feedback_error = message
                logger.error(message)
                return

            with self._lock:
                self._snapshot = snapshot

    @staticmethod
    def _build_snapshot(record) -> _FeedbackSnapshot:
        """把 SDK 结构化反馈帧转换为内部快照。

        Args:
            record: `DobotApiFeedBack.feedBackData()` 返回数组的单条记录

        Returns:
            `_FeedbackSnapshot`

        说明：
            `RobotMode/EnableStatus/RunningStatus/ErrorStatus/TCPSpeedActual`
            字段名取自 SDK 的 `MyType` 定义，含义以实机为准。
        """

        speed = record["TCPSpeedActual"]
        speed_tuple = (float(speed[0]), float(speed[1]), float(speed[2]))
        speed_norm = math.sqrt(sum(value * value for value in speed_tuple))

        return _FeedbackSnapshot(
            timestamp_s=time.monotonic(),
            robot_mode=int(record["RobotMode"]),
            enable_status=int(record["EnableStatus"]),
            running_status=int(record["RunningStatus"]),
            error_status=int(record["ErrorStatus"]),
            tcp_speed_mm_s=speed_tuple,
            tcp_speed_norm_mm_s=speed_norm,
        )

    def _require_fresh_snapshot(self) -> _FeedbackSnapshot | None:
        """读取最新反馈快照并检查新鲜度。

        Returns:
            最新快照；尚未收到数据时返回 None

        Raises:
            FatalError: 反馈线程已报错，或快照超时未更新
        """

        with self._lock:
            snapshot = self._snapshot
            feedback_error = self._feedback_error

        if feedback_error is not None:
            raise FatalError(f"Nova 2 反馈口异常：{feedback_error}")

        if snapshot is None:
            return None

        age_s = time.monotonic() - snapshot.timestamp_s
        if age_s > FEEDBACK_STALE_TIMEOUT_S:
            raise FatalError(f"Nova 2 反馈数据超时未更新（{age_s:.2f} s）")

        return snapshot

    # --------------------------------------------------------
    # 3.3 状态读取
    # --------------------------------------------------------

    def get_pose(self) -> Pose:
        """读取当前末端位姿。

        Returns:
            `Pose`，位置 mm、姿态 deg

        Raises:
            FatalError: 未连接、返回为空或格式无法解析
        """

        self._ensure_connected()
        payload = self._check_reply(self._dashboard.GetPose(), "读取末端位姿")
        values = self._extract_numbers(payload)

        if len(values) != 6:
            raise FatalError(f"末端位姿字段数量异常：{values}")

        if not all(math.isfinite(value) for value in values):
            raise FatalError(f"末端位姿包含非法数值：{values}")

        return Pose(*values)

    def get_joint_angles(self) -> tuple[float, ...]:
        """读取当前关节角。

        Returns:
            6 个关节角，单位 deg

        Raises:
            FatalError: 未连接、返回为空或格式无法解析
        """

        self._ensure_connected()
        payload = self._check_reply(self._dashboard.GetAngle(), "读取关节角")
        values = self._extract_numbers(payload)

        if len(values) != 6:
            raise FatalError(f"关节角字段数量异常：{values}")

        if not all(math.isfinite(value) for value in values):
            raise FatalError(f"关节角包含非法数值：{values}")

        return tuple(values)

    # --------------------------------------------------------
    # 3.4 运动指令
    # --------------------------------------------------------

    def move_j(
        self,
        pose: Pose,
        velocity_ratio_pct: int = DEFAULT_MOVE_JOINT_VELOCITY_PCT,
    ) -> None:
        """关节运动到目标位姿。

        Args:
            pose: 目标位姿，位置 mm、姿态 deg
            velocity_ratio_pct: 速度比例，范围 1..100（不是 mm/s）

        Raises:
            ValueError: 位姿或速度比例非法
            FatalError: 未连接、机械臂非空闲或指令失败
            RecoverableError: 指令返回超时
        """

        self._ensure_connected()
        values = self._pose_to_tuple(pose)
        velocity = self._validate_velocity_ratio(velocity_ratio_pct)
        self._require_motion_ready()

        reply = self._dashboard.MovJ(*values, 0, v=velocity)
        self._check_reply(reply, "move_j")

    def move_l(
        self,
        pose: Pose,
        velocity_ratio_pct: int = DEFAULT_MOVE_LINEAR_VELOCITY_PCT,
    ) -> None:
        """直线运动到目标位姿。

        Args:
            pose: 目标位姿，位置 mm、姿态 deg
            velocity_ratio_pct: 速度比例，范围 1..100（不是 mm/s）

        Raises:
            ValueError: 位姿或速度比例非法
            FatalError: 未连接、机械臂非空闲或指令失败
            RecoverableError: 指令返回超时
        """

        self._ensure_connected()
        values = self._pose_to_tuple(pose)
        velocity = self._validate_velocity_ratio(velocity_ratio_pct)
        self._require_motion_ready()

        reply = self._dashboard.MovL(*values, 0, v=velocity)
        self._check_reply(reply, "move_l")

    def wait_until_stopped(self, timeout_s: float = DEFAULT_STOP_TIMEOUT_S) -> bool:
        """等待机械臂停止。

        Args:
            timeout_s: 超时时间，s

        Returns:
            True 表示已停止；False 表示超时

        Raises:
            FatalError: 未连接、进入异常状态或反馈中断

        说明：
            反馈口可用时使用高频反馈快照；不可用时回退到 Dashboard 状态轮询。
            反馈口速度取 `TCPSpeedActual` 模长，阈值见 `STOP_SPEED_THRESHOLD_MM_S`。
        """

        self._ensure_connected()

        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError(f"timeout_s 必须为正数，当前为 {timeout_s!r}")

        deadline = time.monotonic() + float(timeout_s)

        while time.monotonic() < deadline:
            if self._feedback is not None:
                snapshot = self._require_fresh_snapshot()
                if snapshot is None:
                    time.sleep(WAIT_POLL_INTERVAL_S)
                    continue

                if snapshot.robot_mode in FATAL_ROBOT_MODES:
                    raise FatalError(
                        f"Nova 2 处于异常状态（RobotMode={snapshot.robot_mode}）"
                    )

                if snapshot.robot_mode == ROBOT_MODE_DISABLED:
                    raise FatalError("Nova 2 在运动过程中变为未使能状态")

                if (
                    snapshot.robot_mode == ROBOT_MODE_ENABLE
                    and snapshot.tcp_speed_norm_mm_s <= STOP_SPEED_THRESHOLD_MM_S
                ):
                    return True
            else:
                mode = self._read_robot_mode()

                if mode in FATAL_ROBOT_MODES:
                    raise FatalError(f"Nova 2 处于异常状态（RobotMode={mode}）")

                if mode == ROBOT_MODE_DISABLED:
                    raise FatalError("Nova 2 在运动过程中变为未使能状态")

                if mode == ROBOT_MODE_ENABLE:
                    return True

            time.sleep(WAIT_POLL_INTERVAL_S)

        return False

    def wait_until_moving(
        self,
        timeout_s: float = DEFAULT_MOVE_START_TIMEOUT_S,
    ) -> bool:
        """等待机械臂进入运动状态。

        Args:
            timeout_s: 超时时间，s

        Returns:
            True 表示已进入运动状态；False 表示超时仍处于空闲

        Raises:
            FatalError: 未连接、进入异常状态、未使能或反馈中断

        说明：
            该方法是 `wait_until_stopped` 的启动配对，用于避免运动指令刚下发、
            机械臂尚未启动时被误判为“已到位”。
            反馈口可用时依据 `RobotMode` 与 `TCPSpeedActual`；
            不可用时回退到 Dashboard 状态轮询。
        """

        self._ensure_connected()

        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError(f"timeout_s 必须为正数，当前为 {timeout_s!r}")

        deadline = time.monotonic() + float(timeout_s)

        while time.monotonic() < deadline:
            if self._feedback is not None:
                snapshot = self._require_fresh_snapshot()
                if snapshot is None:
                    time.sleep(WAIT_POLL_INTERVAL_S)
                    continue

                if snapshot.robot_mode in FATAL_ROBOT_MODES:
                    raise FatalError(
                        f"Nova 2 处于异常状态（RobotMode={snapshot.robot_mode}）"
                    )

                if snapshot.robot_mode == ROBOT_MODE_DISABLED:
                    raise FatalError("Nova 2 在运动过程中变为未使能状态")

                if (
                    snapshot.robot_mode in MOVING_ROBOT_MODES
                    or snapshot.tcp_speed_norm_mm_s > STOP_SPEED_THRESHOLD_MM_S
                ):
                    return True
            else:
                mode = self._read_robot_mode()

                if mode in FATAL_ROBOT_MODES:
                    raise FatalError(f"Nova 2 处于异常状态（RobotMode={mode}）")

                if mode == ROBOT_MODE_DISABLED:
                    raise FatalError("Nova 2 在运动过程中变为未使能状态")

                if mode in MOVING_ROBOT_MODES:
                    return True

            time.sleep(WAIT_POLL_INTERVAL_S)

        return False

    def stop(self) -> None:
        """停止当前运动。

        Raises:
            FatalError: 指令失败
        """

        if not self._connected or self._dashboard is None:
            return

        self._check_reply(self._dashboard.Stop(), "停止运动")

    # --------------------------------------------------------
    # 3.5 末端 IO（夹爪等）
    # --------------------------------------------------------

    def set_tool_do(self, index: int, on: bool, immediate: bool = True) -> None:
        """设置末端数字输出口状态。

        Args:
            index: 末端 DO 编号，非负整数
            on: True 表示 ON(1)，False 表示 OFF(0)
            immediate: True 使用立即指令，False 使用队列指令

        Raises:
            ValueError: DO 编号或电平非法
            FatalError: 未连接或指令失败
            RecoverableError: 指令返回超时

        说明：
            夹爪通常需要立即生效，默认使用 `ToolDOInstant`；
            若需要与运动队列严格同步，可传 `immediate=False`。
        """

        self._ensure_connected()
        do_index = self._validate_io_index(index, "tool_do")
        status = self._bool_to_level(on, "on")
        command = (
            self._dashboard.ToolDOInstant
            if immediate
            else self._dashboard.ToolDO
        )
        reply = command(do_index, status)
        self._check_reply(reply, "tool_do")

    def read_tool_do(self, index: int) -> bool:
        """读取末端数字输出口状态。

        Args:
            index: 末端 DO 编号，非负整数

        Returns:
            True 表示 ON，False 表示 OFF

        Raises:
            ValueError: DO 编号非法
            FatalError: 未连接、返回为空或格式无法解析
        """

        self._ensure_connected()
        do_index = self._validate_io_index(index, "read_tool_do")
        payload = self._check_reply(
            self._dashboard.GetToolDO(do_index), "读取末端 DO"
        )
        values = self._extract_numbers(payload)
        return values[0] != 0

    def read_tool_di(self, index: int) -> bool:
        """读取末端数字输入口状态。

        Args:
            index: 末端 DI 编号，非负整数

        Returns:
            True 表示 ON，False 表示 OFF

        Raises:
            ValueError: DI 编号非法
            FatalError: 未连接、返回为空或格式无法解析
        """

        self._ensure_connected()
        di_index = self._validate_io_index(index, "read_tool_di")
        payload = self._check_reply(
            self._dashboard.ToolDI(di_index), "读取末端 DI"
        )
        values = self._extract_numbers(payload)
        return values[0] != 0

    def set_tool_power(self, on: bool) -> None:
        """设置末端工具电源状态。

        Args:
            on: True 表示上电，False 表示断电

        Raises:
            ValueError: 电平非法
            FatalError: 未连接或指令失败
            RecoverableError: 指令返回超时

        说明：
            部分电动夹爪需要重新上电初始化；`identify` 参数暂用默认值，
            实机确认后再决定是否暴露。
        """

        self._ensure_connected()
        status = self._bool_to_level(on, "on")
        reply = self._dashboard.SetToolPower(status)
        self._check_reply(reply, "设置末端工具电源")

    # --------------------------------------------------------
    # 3.6 内部校验与解析
    # --------------------------------------------------------

    def _ensure_connected(self) -> None:
        """检查客户端是否已连接。"""

        if not self._connected or self._dashboard is None:
            raise FatalError("Nova 2 尚未连接")

    def _require_motion_ready(self) -> None:
        """运动前检查机械臂是否为空闲状态。

        Raises:
            RecoverableError: 正在运动
            FatalError: 状态异常或未使能
        """

        mode = self._read_robot_mode()

        if mode == ROBOT_MODE_ENABLE:
            return

        if mode in MOVING_ROBOT_MODES:
            raise RecoverableError(f"Nova 2 正在运动（RobotMode={mode}），拒绝新指令")

        if mode in FATAL_ROBOT_MODES:
            raise FatalError(f"Nova 2 处于异常状态（RobotMode={mode}）")

        raise FatalError(f"Nova 2 未就绪（RobotMode={mode}）")

    def _read_robot_mode(self) -> int:
        """读取机械臂状态码。

        Raises:
            FatalError: 返回为空或格式无法解析
        """

        payload = self._check_reply(self._dashboard.RobotMode(), "查询机器人状态")
        values = self._extract_numbers(payload)
        return int(round(values[0]))

    @staticmethod
    def _pose_to_tuple(pose: Pose) -> tuple[float, float, float, float, float, float]:
        """把 `Pose` 转换为 SDK 需要的 6 元组。

        Raises:
            ValueError: 入参不是 `Pose` 或包含非有限数值
        """

        if not isinstance(pose, Pose):
            raise ValueError(f"pose 必须是 Pose，当前为 {type(pose).__name__}")

        values = (
            float(pose.x_mm),
            float(pose.y_mm),
            float(pose.z_mm),
            float(pose.rx_deg),
            float(pose.ry_deg),
            float(pose.rz_deg),
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"pose 包含非法数值：{values}")

        return values

    @staticmethod
    def _validate_velocity_ratio(velocity_ratio_pct: int) -> int:
        """校验并转换速度比例。

        Raises:
            ValueError: 不是数值或超出 1..100
        """

        if not isinstance(velocity_ratio_pct, (int, float)):
            raise ValueError(
                f"velocity_ratio_pct 必须是数值，当前为 {velocity_ratio_pct!r}"
            )

        value = int(round(float(velocity_ratio_pct)))

        if not MIN_VELOCITY_RATIO_PCT <= value <= MAX_VELOCITY_RATIO_PCT:
            raise ValueError(
                f"velocity_ratio_pct 必须在 "
                f"{MIN_VELOCITY_RATIO_PCT}..{MAX_VELOCITY_RATIO_PCT}，当前为 {value}"
            )

        return value

    @staticmethod
    def _validate_io_index(index: int, name: str) -> int:
        """校验末端 IO 编号。

        Raises:
            ValueError: 不是非负整数
        """

        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{name} 的 IO 编号必须是非负整数，当前为 {index!r}")

        return index

    @staticmethod
    def _bool_to_level(value: bool, name: str) -> int:
        """把布尔电平转换为 SDK 需要的 1/0。

        Raises:
            ValueError: 入参不是 bool
        """

        if not isinstance(value, bool):
            raise ValueError(f"{name} 必须是 bool，当前为 {type(value).__name__}")

        return 1 if value else 0

    @staticmethod
    def _parse_reply(reply: str) -> tuple[int, str]:
        """解析 SDK 原始返回，拆出错误码与载荷。

        Args:
            reply: SDK `sendRecvMsg` 原始字符串

        Returns:
            `(error_code, payload)`

        Raises:
            FatalError: 返回为空或缺少错误码

        说明：
            典型格式为 `"0,{...},Command();"`；实际格式以实机为准，
            此处只提取开头整数并保留其余载荷供后续解析。
        """

        if not isinstance(reply, str) or not reply.strip():
            raise FatalError("Nova 2 返回为空，无法确认指令结果")

        text = reply.strip()
        match = re.match(r"\s*(-?\d+)", text)

        if match is None:
            raise FatalError(f"无法解析 Nova 2 返回内容：{reply!r}")

        error_code = int(match.group(1))
        payload = text[match.end():].lstrip(" ,")
        return error_code, payload

    @staticmethod
    def _extract_numbers(payload: str) -> list[float]:
        """从载荷中提取数值列表。

        Args:
            payload: `_parse_reply` 返回的载荷

        Returns:
            浮点数列表

        Raises:
            FatalError: 未提取到任何数值

        说明：
            优先取第一对花括号内的内容；无花括号时取分号前内容。
            实际格式待实机核对。
        """

        match = re.search(r"\{([^{}]*)\}", payload)

        if match is not None:
            body = match.group(1)
        else:
            body = payload.split(";")[0]

        values: list[float] = []
        for token in body.replace("\n", " ").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue

        if not values:
            raise FatalError(f"无法从 Nova 2 返回中提取数值：{payload!r}")

        return values

    def _check_reply(self, reply: str, action: str) -> str:
        """检查指令返回码，非零时转换为项目异常。

        Args:
            reply: SDK 原始返回
            action: 动作名，用于异常信息

        Returns:
            载荷字符串

        Raises:
            RecoverableError: 返回中提示超时
            FatalError: 其他非零错误码

        说明：
            该 SDK 版本未提供完整错误码表，除明确超时外一律按致命错误处理，
            避免报警、限位、碰撞等情况下继续运动。
        """

        error_code, payload = self._parse_reply(reply)

        if error_code == 0:
            return payload

        lowered = reply.lower()
        if "timeout" in lowered or "超时" in reply:
            raise RecoverableError(
                f"Nova 2 {action} 超时（code={error_code}）：{reply!r}"
            )

        raise FatalError(f"Nova 2 {action} 失败（code={error_code}）：{reply!r}")


# ============================================================
# 4. 连通性自测入口
# ============================================================

def main() -> None:
    """从配置读取连接参数并做连通性自检。"""

    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open("config/robot_config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    robot_config = config["robot"]
    client = Nova2Client(
        ip=robot_config["ip"],
        port=robot_config["port"],
        feedback_port=robot_config.get("feedback_port", DEFAULT_FEEDBACK_PORT),
    )

    client.connect()
    try:
        logger.info("当前位姿：%s", client.get_pose())
        logger.info("当前关节角：%s", client.get_joint_angles())
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
