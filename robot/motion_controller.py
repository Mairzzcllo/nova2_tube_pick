"""运动控制模块。

职责：
    1. 在 Nova2Client 之上封装笛卡尔/关节空间运动与到位判断
    2. 提供 move_to / move_linear / wait_arrival 三个高层接口
    3. 规避“运动指令刚下发、机械臂尚未启动就判定到位”的竞态
    4. 把运动超时转换为 RecoverableError，异常状态交由 Nova2Client 抛出

说明：
    - 本模块不直接接触 Nova 2 SDK，所有运动指令经 `Nova2Client`。
    - 对外位姿统一为 `Pose`（位置 mm、姿态 deg，固定轴 XYZ）。
    - 速度接口统一为 `velocity_ratio_pct`（厂商百分比 1..100，不是 mm/s）。
    - 运动超时、未启动、已停止但未到位均抛 `RecoverableError`；
      限位、未使能、急停、碰撞等由 `Nova2Client` 抛出 `FatalError`，本层不包装。
"""

import logging
import math

from common.errors import Nova2ProjectError, RecoverableError
from common.types import Pose

logger = logging.getLogger(__name__)


# ============================================================
# 1. 常量与默认参数
# ============================================================

# 默认关节运动速度比例
DEFAULT_JOINT_VELOCITY_PCT = 50

# 默认直线运动速度比例
DEFAULT_LINEAR_VELOCITY_PCT = 10

# 速度比例合法范围（厂商百分比，不是 mm/s）
MIN_VELOCITY_RATIO_PCT = 1
MAX_VELOCITY_RATIO_PCT = 100

# 默认等待到位超时，单位 s
DEFAULT_ARRIVAL_TIMEOUT_S = 10.0

# 等待运动启动超时，单位 s
MOVE_START_TIMEOUT_S = 2.0

# 到位判定容差：位置 mm，姿态 deg
ARRIVAL_POSITION_TOLERANCE_MM = 1.0
ARRIVAL_ROTATION_TOLERANCE_DEG = 0.5

# 角度归一化半周期，degree
ANGLE_HALF_PERIOD_DEG = 180.0


# ============================================================
# 2. 运动控制器主体
# ============================================================

class MotionController:
    """机械臂运动控制器。

    Args:
        client: `Nova2Client` 实例，提供 move_j / move_l / get_pose /
            wait_until_moving / wait_until_stopped / stop

    说明：
        上层只通过 `move_to` / `move_linear` / `wait_arrival` 调用；
        安全接近点、抓取/插入轨迹由任务层用这三个原语组合。
    """

    def __init__(self, client):
        if client is None:
            raise ValueError("client 不能为 None")

        self.client = client

    # --------------------------------------------------------
    # 2.1 公开运动接口
    # --------------------------------------------------------

    def move_to(
        self,
        pose: Pose,
        velocity_ratio_pct: int = DEFAULT_JOINT_VELOCITY_PCT,
    ) -> None:
        """关节运动到目标位姿，并等待到位。

        Args:
            pose: 目标位姿，位置 mm、姿态 deg
            velocity_ratio_pct: 速度比例，范围 1..100（不是 mm/s）

        Raises:
            ValueError: 位姿或速度比例非法
            RecoverableError: 未在超时内启动/停止，或停止后未到位
            FatalError: 未连接、限位、未使能、急停、碰撞等
        """

        self._validate_pose(pose)
        velocity = self._validate_velocity_ratio(velocity_ratio_pct)

        if self._is_at_pose(pose):
            logger.info("当前位姿已在目标容差内，跳过关节运动")
            return

        logger.info("关节运动到 %s，速度比例 %s%%", pose, velocity)
        self.client.move_j(pose, velocity)
        self._wait_for_arrival(pose)

    def move_linear(
        self,
        pose: Pose,
        velocity_ratio_pct: int = DEFAULT_LINEAR_VELOCITY_PCT,
    ) -> None:
        """直线运动到目标位姿，并等待到位。

        Args:
            pose: 目标位姿，位置 mm、姿态 deg
            velocity_ratio_pct: 速度比例，范围 1..100（不是 mm/s）

        Raises:
            ValueError: 位姿或速度比例非法
            RecoverableError: 未在超时内启动/停止，或停止后未到位
            FatalError: 未连接、限位、未使能、急停、碰撞等
        """

        self._validate_pose(pose)
        velocity = self._validate_velocity_ratio(velocity_ratio_pct)

        if self._is_at_pose(pose):
            logger.info("当前位姿已在目标容差内，跳过直线运动")
            return

        logger.info("直线运动到 %s，速度比例 %s%%", pose, velocity)
        self.client.move_l(pose, velocity)
        self._wait_for_arrival(pose)

    def wait_arrival(self, timeout_s: float = DEFAULT_ARRIVAL_TIMEOUT_S) -> bool:
        """等待机械臂停止。

        Args:
            timeout_s: 超时时间，s

        Returns:
            True 表示已停止；False 表示超时

        Raises:
            ValueError: 超时时间非法
            FatalError: 未连接、进入异常状态、未使能或反馈中断

        说明：
            直接委托 `Nova2Client.wait_until_stopped`；
            本方法只判断“是否停止”，不判断“是否到达目标位姿”。
        """

        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError(f"timeout_s 必须为正数，当前为 {timeout_s!r}")

        return self.client.wait_until_stopped(float(timeout_s))

    # --------------------------------------------------------
    # 2.2 到位流程
    # --------------------------------------------------------

    def _wait_for_arrival(self, pose: Pose) -> None:
        """确认运动已启动并等待停止，再校验目标位姿。

        Args:
            pose: 目标位姿

        Raises:
            RecoverableError: 未启动、未停止或停止后未到位
            FatalError: 底层异常状态
        """

        started = self.client.wait_until_moving(MOVE_START_TIMEOUT_S)

        if not started:
            # 目标与当前位置极近时机械臂可能不产生可观测运动
            if self._is_at_pose(pose):
                logger.info("机械臂未启动，但目标位姿已在容差内")
                return

            self._safe_stop()
            raise RecoverableError(
                f"运动指令已下发但机械臂未在 {MOVE_START_TIMEOUT_S:.1f} s 内启动"
            )

        stopped = self.client.wait_until_stopped(DEFAULT_ARRIVAL_TIMEOUT_S)

        if not stopped:
            self._safe_stop()
            raise RecoverableError(
                f"运动未在 {DEFAULT_ARRIVAL_TIMEOUT_S:.1f} s 内停止"
            )

        if not self._is_at_pose(pose):
            self._safe_stop()
            raise RecoverableError("机械臂已停止但未到达目标位姿")

        logger.info("已到达目标位姿")

    def _is_at_pose(self, pose: Pose) -> bool:
        """判断当前位姿是否在目标容差内。

        Args:
            pose: 目标位姿

        Returns:
            True 表示平移与旋转误差均在容差内

        Raises:
            FatalError: 读取当前位姿失败
        """

        current = self.client.get_pose()

        position_error_mm = math.dist(
            (current.x_mm, current.y_mm, current.z_mm),
            (pose.x_mm, pose.y_mm, pose.z_mm),
        )
        rotation_error_deg = max(
            abs(self._normalize_angle_deg(current.rx_deg - pose.rx_deg)),
            abs(self._normalize_angle_deg(current.ry_deg - pose.ry_deg)),
            abs(self._normalize_angle_deg(current.rz_deg - pose.rz_deg)),
        )

        return (
            position_error_mm <= ARRIVAL_POSITION_TOLERANCE_MM
            and rotation_error_deg <= ARRIVAL_ROTATION_TOLERANCE_DEG
        )

    def _safe_stop(self) -> None:
        """尽力停止当前运动，且不掩盖原始异常。"""

        try:
            self.client.stop()
        except Nova2ProjectError as exc:
            logger.error("到位失败后停止机械臂失败：%s", exc)

    # --------------------------------------------------------
    # 2.3 内部校验
    # --------------------------------------------------------

    @staticmethod
    def _validate_pose(pose: Pose) -> None:
        """校验位姿类型与数值。

        Raises:
            ValueError: 不是 Pose 或包含非有限数值
        """

        if not isinstance(pose, Pose):
            raise ValueError(f"pose 必须是 Pose，当前为 {type(pose).__name__}")

        values = (
            pose.x_mm,
            pose.y_mm,
            pose.z_mm,
            pose.rx_deg,
            pose.ry_deg,
            pose.rz_deg,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"pose 包含非法数值：{values}")

    @staticmethod
    def _validate_velocity_ratio(velocity_ratio_pct: int) -> int:
        """校验并归一化速度比例。

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
    def _normalize_angle_deg(angle_deg: float) -> float:
        """把角度归一化到 [-180, 180)，用于处理角度环绕。"""

        return (angle_deg + ANGLE_HALF_PERIOD_DEG) % (
            2 * ANGLE_HALF_PERIOD_DEG
        ) - ANGLE_HALF_PERIOD_DEG


# ============================================================
# 3. 入口
# ============================================================

def main() -> None:
    """打印模块说明；运动测试需连接真实硬件。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("MotionController 提供 move_to / move_linear / wait_arrival")
    logger.info("离线逻辑测试见 debug/test_motion_controller.py；空载运动需连接真实硬件")


if __name__ == "__main__":
    main()
