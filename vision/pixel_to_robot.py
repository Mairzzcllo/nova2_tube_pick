"""像素坐标到机器人坐标转换模块。

职责：
    1. 结合相机内参 K 与手眼标定矩阵 T_base_camera，把像素坐标反投影为
       机器人基坐标系下的三维点（Point3D）
    2. 根据试管像素检测结果与试管几何，计算基座坐标系下的抓取位姿（Pose）
    3. 根据孔位像素与试管架模型，计算基座坐标系下的孔位位姿（HolePose）

说明：
    - 本模块是视觉与运动控制之间的桥梁，只做几何换算，不识别、不控制机械臂
    - 输入图像默认已完成去畸变，因此反投影只使用 K，不再用 D 做去畸变
    - T_base_camera 方向为相机 -> 基座，平移单位 mm
    - 深度来源必须显式：外部深度给 `depth_mm`（相机系 Z），否则与桌面平面求交
    - 抓取姿态中的工具朝向为占位常量，需按 Nova 2 夹爪 TCP 安装实测后确认
"""

import math
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

# 兼容 python -m 与直接执行文件两种方式，确保能导入仓库内模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import FatalError
from common.types import HolePose, Plane, Point3D, Pose, TubeDetection, TubeGeometry

if TYPE_CHECKING:  # 仅用于类型标注，RackModel 暂未实现，避免运行时依赖
    from calibration.rack_model import RackModel


# ============================================================
# 1. 默认参数
# ============================================================

# 沿试管长轴取探测点的像素步长，pixel
DEFAULT_AXIS_PROBE_PX = 40.0

# 孔位像素反投影结果与 RackModel 孔位中心允许的最大偏差，mm
DEFAULT_HOLE_MATCH_TOLERANCE_MM = 10.0

# 抓取工具朝向占位常量（固定轴 XYZ，deg）
# TODO: 占位：Nova 2 夹爪 TCP 安装方式尚未确认，工具朝下的 roll/pitch 需实测后填写
TOOL_DOWN_ROLL_DEG = 180.0
TOOL_DOWN_PITCH_DEG = 0.0

# 数值容差
PARALLEL_EPS = 1e-9
MIN_DIRECTION_NORM = 1e-9
TRANSFORM_ATOL = 1e-6


# ============================================================
# 2. 输入校验与基础几何
# ============================================================

def _validate_intrinsics(intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """校验相机内参 (K, D)。

    Args:
        intrinsics: `(K, D)` 二元组

    Returns:
        (K, D) 的 float64 数组

    Raises:
        FatalError: 结构、形状或数值非法
    """

    try:
        camera_matrix, dist_coeffs = intrinsics
    except (TypeError, ValueError) as error:
        raise FatalError(
            f"相机内参格式非法：{intrinsics!r}，预期 (K, D)"
        ) from error

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    if camera_matrix.shape != (3, 3):
        raise FatalError(f"相机内参 K 形状非法：{camera_matrix.shape}，预期 (3, 3)")

    if not np.all(np.isfinite(camera_matrix)):
        raise FatalError("相机内参 K 包含非有限值")

    if abs(float(camera_matrix[0, 0])) < PARALLEL_EPS or abs(
        float(camera_matrix[1, 1])
    ) < PARALLEL_EPS:
        raise FatalError("相机内参 K 的焦距非法")

    if dist_coeffs.ndim > 2 or dist_coeffs.size < 4:
        raise FatalError(f"相机畸变 D 形状非法：{dist_coeffs.shape}")

    if not np.all(np.isfinite(dist_coeffs)):
        raise FatalError("相机畸变 D 包含非有限值")

    return camera_matrix, dist_coeffs


def _validate_transform(transform) -> np.ndarray:
    """校验 4x4 齐次变换矩阵。

    Args:
        transform: 待校验的变换矩阵

    Returns:
        4x4 float64 矩阵

    Raises:
        FatalError: 形状、末行、旋转正交性或数值非法
    """

    matrix = np.asarray(transform, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise FatalError(f"变换矩阵形状非法：{matrix.shape}，预期 (4, 4)")

    if not np.all(np.isfinite(matrix)):
        raise FatalError("变换矩阵包含非有限值")

    if not np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=TRANSFORM_ATOL):
        raise FatalError(f"变换矩阵末行非法：{matrix[3, :]}")

    rotation = matrix[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=TRANSFORM_ATOL,
    ):
        raise FatalError("变换矩阵旋转部分非正交")

    if abs(float(np.linalg.det(rotation)) - 1.0) > TRANSFORM_ATOL:
        raise FatalError("变换矩阵旋转部分行列式非法")

    return matrix


def _validate_plane(plane: Plane) -> tuple[np.ndarray, float]:
    """校验相机系平面并归一化法向。

    Args:
        plane: 平面 a*x + b*y + c*z + d = 0

    Returns:
        (单位法向, 归一化常数项 d)

    Raises:
        FatalError: 坐标系、法向或数值非法
    """

    if not isinstance(plane, Plane):
        raise FatalError(f"平面类型非法：{type(plane)!r}，预期 Plane")

    if plane.frame != "camera":
        raise FatalError(f"平面坐标系非法：{plane.frame}，当前仅支持 camera")

    normal = np.array([plane.a, plane.b, plane.c], dtype=np.float64)

    if not np.all(np.isfinite(normal)) or not math.isfinite(float(plane.d)):
        raise FatalError("平面参数包含非有限值")

    norm = float(np.linalg.norm(normal))

    if norm < MIN_DIRECTION_NORM:
        raise FatalError("平面法向为零向量")

    return normal / norm, float(plane.d) / norm


def _decompose_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """拆分 4x4 变换为旋转矩阵与平移向量。

    Args:
        transform: 4x4 齐次变换

    Returns:
        (3x3 旋转矩阵, 3x1 平移向量)
    """

    return transform[:3, :3].copy(), transform[:3, 3].copy()


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    """求 4x4 刚体变换的逆。

    Args:
        transform: 4x4 齐次变换

    Returns:
        逆变换，4x4
    """

    rotation, translation = _decompose_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _fixed_xyz_deg_to_matrix(pose: Pose) -> np.ndarray:
    """将固定轴 XYZ（roll-pitch-yaw）欧拉角转换为旋转矩阵。

    Args:
        pose: 位置 mm、姿态 deg 的位姿

    Returns:
        3x3 旋转矩阵，R = Rz(rz) * Ry(ry) * Rx(rx)
    """

    rx, ry, rz = np.deg2rad([pose.rx_deg, pose.ry_deg, pose.rz_deg])

    cos_x, sin_x = math.cos(float(rx)), math.sin(float(rx))
    cos_y, sin_y = math.cos(float(ry)), math.sin(float(ry))
    cos_z, sin_z = math.cos(float(rz)), math.sin(float(rz))

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x],
            [0.0, sin_x, cos_x],
        ],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0.0],
            [sin_z, cos_z, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rotation_z @ rotation_y @ rotation_x


def plane_from_base_table_z(
    T_base_camera_mm,
    table_z_mm: float,
) -> Plane:
    """由基座坐标系桌面高度构造相机系桌面平面。

    Args:
        T_base_camera_mm: 4x4 T_base_camera，平移 mm
        table_z_mm: 桌面在基座坐标系下的 Z 高度，mm

    Returns:
        相机系平面 Plane(a, b, c, d, frame="camera")

    Raises:
        FatalError: 变换矩阵或高度非法
    """

    transform = _validate_transform(T_base_camera_mm)

    if not math.isfinite(float(table_z_mm)):
        raise FatalError(f"桌面高度非法：{table_z_mm}")

    T_camera_base = _invert_transform(transform)
    rotation_camera_base, translation_camera_base = _decompose_transform(
        T_camera_base
    )

    # 桌面在基座系中为 z = table_z，法向取 (0, 0, 1)
    normal_base = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    point_base = np.array([0.0, 0.0, float(table_z_mm)], dtype=np.float64)

    normal_camera = rotation_camera_base @ normal_base
    point_camera = rotation_camera_base @ point_base + translation_camera_base
    constant = -float(np.dot(normal_camera, point_camera))

    return Plane(
        a=float(normal_camera[0]),
        b=float(normal_camera[1]),
        c=float(normal_camera[2]),
        d=constant,
        frame="camera",
    )


# ============================================================
# 3. 像素到机器人坐标转换
# ============================================================

class PixelToRobot:
    """像素坐标到机器人基坐标转换器。

    Args:
        intrinsics: `(K, D)`；K 为 3x3 相机内参，D 为畸变系数（输入图已去畸变，
            当前仅校验不使用）
        T_base_camera_mm: 4x4 相机 -> 基座变换，平移 mm
        plane: 相机系桌面平面，用于 `depth_mm=None` 时的反投影

    Raises:
        FatalError: 内参、手眼矩阵或平面非法
    """

    def __init__(self, intrinsics, T_base_camera_mm, plane: Plane):
        self._camera_matrix, self._dist_coeffs = _validate_intrinsics(intrinsics)
        self._camera_matrix_inverse = np.linalg.inv(self._camera_matrix)

        self._T_base_camera = _validate_transform(T_base_camera_mm)
        self._T_camera_base = _invert_transform(self._T_base_camera)

        self._rotation_base_camera, self._translation_base_camera = (
            _decompose_transform(self._T_base_camera)
        )
        self._rotation_camera_base, self._translation_camera_base = (
            _decompose_transform(self._T_camera_base)
        )

        self._plane_normal, self._plane_constant = _validate_plane(plane)

    # --------------------------------------------------------
    # 3.1 基础几何
    # --------------------------------------------------------

    def _pixel_ray(self, u: float, v: float) -> np.ndarray:
        """将像素坐标转换为相机系射线方向（Z 分量为 1）。

        Args:
            u: 像素横坐标，pixel
            v: 像素纵坐标，pixel

        Returns:
            相机系射线方向，shape (3,)

        Raises:
            ValueError: 像素坐标非法
        """

        pixel = np.array([u, v, 1.0], dtype=np.float64)

        if not np.all(np.isfinite(pixel)):
            raise ValueError(f"像素坐标非法：u={u}, v={v}")

        return self._camera_matrix_inverse @ pixel

    @staticmethod
    def _intersect_plane(
        ray: np.ndarray,
        normal: np.ndarray,
        constant: float,
        plane_name: str,
    ) -> np.ndarray:
        """求相机原点出发的射线与平面交点。

        Args:
            ray: 相机系射线方向
            normal: 单位法向
            constant: 平面常数项（已随法向归一化）
            plane_name: 平面名称，用于异常信息

        Returns:
            相机系交点，shape (3,)

        Raises:
            ValueError: 射线与平面平行或交点在相机后方
        """

        denominator = float(np.dot(normal, ray))

        if abs(denominator) < PARALLEL_EPS:
            raise ValueError(f"射线与{plane_name}平行，无法求交")

        distance = -constant / denominator

        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"射线与{plane_name}的交点位于相机后方")

        return distance * ray

    def _camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        """相机系点转换到机器人基座坐标系。

        Args:
            point_camera: 相机系点，shape (3,)

        Returns:
            基座系点，shape (3,)
        """

        return (
            self._rotation_base_camera @ point_camera
            + self._translation_base_camera
        )

    # --------------------------------------------------------
    # 3.2 公开接口
    # --------------------------------------------------------

    def pixel_to_point(
        self,
        u: float,
        v: float,
        depth_mm: float | None = None,
    ) -> Point3D:
        """将像素坐标转换为机器人基坐标系三维点。

        Args:
            u: 像素横坐标，pixel
            v: 像素纵坐标，pixel
            depth_mm: 相机系 Z 深度，mm；为 None 时与桌面平面求交

        Returns:
            基座坐标系三维点 Point3D，mm

        Raises:
            ValueError: 像素/深度非法，或射线与平面不相交
        """

        ray = self._pixel_ray(u, v)

        if depth_mm is None:
            point_camera = self._intersect_plane(
                ray,
                self._plane_normal,
                self._plane_constant,
                "桌面平面",
            )
        else:
            depth = float(depth_mm)

            if not math.isfinite(depth) or depth <= 0.0:
                raise ValueError(f"深度非法：{depth_mm}，必须为有限正值")

            point_camera = depth * ray

        point_base = self._camera_to_base(point_camera)

        return Point3D(
            x_mm=float(point_base[0]),
            y_mm=float(point_base[1]),
            z_mm=float(point_base[2]),
        )

    def tube_to_grasp_pose(
        self,
        tube: TubeDetection,
        tube_geometry: TubeGeometry,
    ) -> Pose:
        """计算试管抓取位姿。

        Args:
            tube: 试管检测结果，像素中心与长轴方向
            tube_geometry: 试管几何；`grasp_offset_mm` 正方向为彩盖端 -> 管底端

        Returns:
            基座坐标系抓取位姿 Pose，位置 mm、姿态 deg

        Raises:
            ValueError: 几何非法或长轴方向退化
        """

        if not isinstance(tube_geometry, TubeGeometry):
            raise ValueError(
                f"试管几何类型非法：{type(tube_geometry)!r}，预期 TubeGeometry"
            )

        diameter_mm = float(tube_geometry.diameter_mm)
        length_mm = float(tube_geometry.length_mm)
        offset_mm = float(tube_geometry.grasp_offset_mm)

        if not math.isfinite(diameter_mm) or diameter_mm <= 0.0:
            raise ValueError(f"试管直径非法：{tube_geometry.diameter_mm}")

        if not math.isfinite(length_mm) or length_mm <= 0.0:
            raise ValueError(f"试管长度非法：{tube_geometry.length_mm}")

        if not math.isfinite(offset_mm) or abs(offset_mm) > length_mm / 2.0:
            raise ValueError(
                f"夹持偏移非法：{tube_geometry.grasp_offset_mm}，"
                f"应在 ±{length_mm / 2.0:.3f} mm 内"
            )

        axis = np.array([tube.axis_u, tube.axis_v], dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))

        if not np.all(np.isfinite(axis)) or axis_norm < MIN_DIRECTION_NORM:
            raise ValueError("试管长轴方向退化，无法计算抓取姿态")

        axis /= axis_norm

        center = self.pixel_to_point(tube.center_u_px, tube.center_v_px)
        probe_plus = self.pixel_to_point(
            tube.center_u_px + axis[0] * DEFAULT_AXIS_PROBE_PX,
            tube.center_v_px + axis[1] * DEFAULT_AXIS_PROBE_PX,
        )
        probe_minus = self.pixel_to_point(
            tube.center_u_px - axis[0] * DEFAULT_AXIS_PROBE_PX,
            tube.center_v_px - axis[1] * DEFAULT_AXIS_PROBE_PX,
        )

        # 基座系长轴方向：由彩盖端指向管底端
        axis_base = np.array(
            [
                probe_plus.x_mm - probe_minus.x_mm,
                probe_plus.y_mm - probe_minus.y_mm,
            ],
            dtype=np.float64,
        )
        axis_base_norm = float(np.linalg.norm(axis_base))

        if axis_base_norm < MIN_DIRECTION_NORM:
            raise ValueError("试管长轴在基座 XY 平面内退化，无法计算抓取姿态")

        axis_base /= axis_base_norm

        grasp_x_mm = center.x_mm + axis_base[0] * offset_mm
        grasp_y_mm = center.y_mm + axis_base[1] * offset_mm
        grasp_z_mm = center.z_mm + diameter_mm / 2.0

        yaw_deg = math.degrees(math.atan2(axis_base[1], axis_base[0]))

        # TODO: 占位：工具朝下角度与夹爪手指开合方向需按实际 TCP 实测确认
        return Pose(
            x_mm=float(grasp_x_mm),
            y_mm=float(grasp_y_mm),
            z_mm=float(grasp_z_mm),
            rx_deg=TOOL_DOWN_ROLL_DEG,
            ry_deg=TOOL_DOWN_PITCH_DEG,
            rz_deg=float(yaw_deg),
        )

    def hole_pixel_to_pose(
        self,
        hole,
        rack_model: "RackModel",
    ) -> HolePose:
        """将孔位像素映射为基座坐标系孔位位姿。

        Args:
            hole: 孔位像素 HolePixel
            rack_model: 试管架模型，需提供 `hole_pose(hole_id) -> HolePose`

        Returns:
            基座坐标系孔位位姿 HolePose

        Raises:
            FatalError: 试管架模型缺失或不可用
            ValueError: 孔位不存在、像素非法或反投影结果与模型孔位不匹配
        """

        if rack_model is None:
            raise FatalError("试管架模型缺失，无法进行孔位转换")

        hole_pose_method = getattr(rack_model, "hole_pose", None)

        if not callable(hole_pose_method):
            raise FatalError("试管架模型未提供 hole_pose(hole_id) 接口")

        try:
            hole_pose = hole_pose_method(hole.hole_id)
        except KeyError as error:
            raise ValueError(f"试管架模型不存在孔位：{hole.hole_id}") from error

        pose_base = hole_pose.pose_base
        rotation_base = _fixed_xyz_deg_to_matrix(pose_base)
        normal_base = rotation_base[:, 2]
        point_base = np.array(
            [pose_base.x_mm, pose_base.y_mm, pose_base.z_mm],
            dtype=np.float64,
        )

        # 将架面（点 + 法向）转换到相机系
        normal_camera = self._rotation_camera_base @ normal_base
        point_camera = (
            self._rotation_camera_base @ point_base
            + self._translation_camera_base
        )
        constant_camera = -float(np.dot(normal_camera, point_camera))

        ray = self._pixel_ray(hole.u_px, hole.v_px)
        point_camera = self._intersect_plane(
            ray,
            normal_camera,
            constant_camera,
            "试管架平面",
        )
        point_base_projected = self._camera_to_base(point_camera)

        error_mm = float(np.linalg.norm(point_base_projected - point_base))

        if error_mm > DEFAULT_HOLE_MATCH_TOLERANCE_MM:
            raise ValueError(
                f"孔位 {hole.hole_id} 像素反投影与模型偏差 {error_mm:.3f} mm，"
                f"超过容差 {DEFAULT_HOLE_MATCH_TOLERANCE_MM:.3f} mm"
            )

        return hole_pose


# ============================================================
# 4. 自检入口
# ============================================================

def main() -> None:
    """使用合成标定数据做自检。"""

    camera_matrix = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)

    # 相机位于基座上方 500 mm，光轴朝下
    T_base_camera = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 500.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    table_plane = plane_from_base_table_z(T_base_camera, table_z_mm=0.0)
    converter = PixelToRobot(
        (camera_matrix, dist_coeffs),
        T_base_camera,
        table_plane,
    )

    point = converter.pixel_to_point(320.0, 240.0)
    print("合成数据自检：图像中心反投影结果")
    print(f"  x_mm = {point.x_mm:.3f}")
    print(f"  y_mm = {point.y_mm:.3f}")
    print(f"  z_mm = {point.z_mm:.3f}")

    assert abs(point.x_mm) < 1e-6
    assert abs(point.y_mm) < 1e-6
    assert abs(point.z_mm) < 1e-6

    print("合成数据自检通过")


if __name__ == "__main__":
    main()
