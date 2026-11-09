"""像素到机器人坐标转换离线测试。

覆盖：
    1. pixel_to_point 的深度路径与桌面平面路径往返精度
    2. 射线平行/交点在相机后方/参数非法等失败路径
    3. tube_to_grasp_pose 的位置、抬升高度、长轴偏航角
    4. hole_pixel_to_pose 的命中与孔位不匹配
    5. 构造期内参/手眼矩阵/平面校验

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_pixel_to_robot.py
"""

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import FatalError
from common.types import (
    HolePixel,
    HolePose,
    Plane,
    Pose,
    TubeDetection,
    TubeGeometry,
)
from vision.pixel_to_robot import (
    DEFAULT_HOLE_MATCH_TOLERANCE_MM,
    PixelToRobot,
    plane_from_base_table_z,
)


# ============================================================
# 1. 合成标定数据与测试工具
# ============================================================

CAMERA_MATRIX = np.array(
    [
        [800.0, 0.0, 320.0],
        [0.0, 820.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST_COEFFS = np.zeros(5, dtype=np.float64)

T_BASE_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0, 100.0],
        [0.0, -1.0, 0.0, -50.0],
        [0.0, 0.0, -1.0, 600.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

TABLE_Z_MM = -20.0

TABLE_PLANE = plane_from_base_table_z(T_BASE_CAMERA, TABLE_Z_MM)

CONVERTER = PixelToRobot(
    (CAMERA_MATRIX, DIST_COEFFS),
    T_BASE_CAMERA,
    TABLE_PLANE,
)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """求 4x4 刚体变换的逆。"""

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


T_CAMERA_BASE = invert_transform(T_BASE_CAMERA)


def base_point_to_camera(point_base_mm) -> np.ndarray:
    """基座系点转换到相机系。"""

    point = np.asarray(point_base_mm, dtype=np.float64)
    return T_CAMERA_BASE[:3, :3] @ point + T_CAMERA_BASE[:3, 3]


def base_point_to_pixel(point_base_mm) -> tuple[float, float]:
    """基座系点投影到像素坐标（针孔模型）。"""

    point_camera = base_point_to_camera(point_base_mm)

    assert point_camera[2] > 0.0, "测试点必须位于相机前方"

    projected = CAMERA_MATRIX @ point_camera
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def angle_difference_deg(first_deg: float, second_deg: float) -> float:
    """返回两个角度在 [-180, 180) 内的最小差值。"""

    return (first_deg - second_deg + 180.0) % 360.0 - 180.0


def expect_raises(exception_type, function, *args, **kwargs) -> None:
    """断言调用会抛出指定异常。"""

    try:
        function(*args, **kwargs)
    except exception_type:
        return

    raise AssertionError(f"预期 {exception_type.__name__}，但未抛出异常")


# ============================================================
# 2. pixel_to_point 测试
# ============================================================

def test_pixel_to_point_plane_roundtrip() -> None:
    """桌面平面路径应精确往返基座点。"""

    table_points = (
        (0.0, 0.0, TABLE_Z_MM),
        (150.0, 120.0, TABLE_Z_MM),
        (-200.0, 80.0, TABLE_Z_MM),
        (60.0, -180.0, TABLE_Z_MM),
    )

    for point_base in table_points:
        u_px, v_px = base_point_to_pixel(point_base)
        result = CONVERTER.pixel_to_point(u_px, v_px)

        assert abs(result.x_mm - point_base[0]) < 1e-6, (result, point_base)
        assert abs(result.y_mm - point_base[1]) < 1e-6, (result, point_base)
        assert abs(result.z_mm - point_base[2]) < 1e-6, (result, point_base)

    print("  pixel_to_point 桌面平面往返：通过")


def test_pixel_to_point_depth_roundtrip() -> None:
    """显式深度路径应精确往返基座点。"""

    base_points = (
        (30.0, 40.0, 100.0),
        (-120.0, 60.0, 250.0),
    )

    for point_base in base_points:
        point_camera = base_point_to_camera(point_base)
        u_px, v_px = base_point_to_pixel(point_base)
        result = CONVERTER.pixel_to_point(
            u_px,
            v_px,
            depth_mm=float(point_camera[2]),
        )

        assert abs(result.x_mm - point_base[0]) < 1e-6, (result, point_base)
        assert abs(result.y_mm - point_base[1]) < 1e-6, (result, point_base)
        assert abs(result.z_mm - point_base[2]) < 1e-6, (result, point_base)

    print("  pixel_to_point 深度往返：通过")


def test_pixel_to_point_failures() -> None:
    """非法像素/深度与几何退化应抛出 ValueError。"""

    expect_raises(ValueError, CONVERTER.pixel_to_point, 320.0, 240.0, 0.0)
    expect_raises(ValueError, CONVERTER.pixel_to_point, 320.0, 240.0, -5.0)
    expect_raises(ValueError, CONVERTER.pixel_to_point, 320.0, 240.0, float("nan"))
    expect_raises(ValueError, CONVERTER.pixel_to_point, float("nan"), 240.0)

    parallel_converter = PixelToRobot(
        (CAMERA_MATRIX, DIST_COEFFS),
        T_BASE_CAMERA,
        Plane(a=1.0, b=0.0, c=0.0, d=0.0, frame="camera"),
    )
    expect_raises(ValueError, parallel_converter.pixel_to_point, 320.0, 240.0)

    behind_converter = PixelToRobot(
        (CAMERA_MATRIX, DIST_COEFFS),
        T_BASE_CAMERA,
        Plane(a=0.0, b=0.0, c=-1.0, d=-500.0, frame="camera"),
    )
    expect_raises(ValueError, behind_converter.pixel_to_point, 320.0, 240.0)

    print("  pixel_to_point 失败路径：通过")


def test_constructor_validation() -> None:
    """非法内参/手眼矩阵/平面应在构造期抛 FatalError。"""

    expect_raises(
        FatalError,
        PixelToRobot,
        (np.eye(2), DIST_COEFFS),
        T_BASE_CAMERA,
        TABLE_PLANE,
    )
    expect_raises(
        FatalError,
        PixelToRobot,
        (CAMERA_MATRIX, DIST_COEFFS),
        np.eye(3),
        TABLE_PLANE,
    )
    expect_raises(
        FatalError,
        PixelToRobot,
        (CAMERA_MATRIX, DIST_COEFFS),
        T_BASE_CAMERA,
        Plane(a=0.0, b=0.0, c=0.0, d=1.0, frame="camera"),
    )
    expect_raises(
        FatalError,
        PixelToRobot,
        (CAMERA_MATRIX, DIST_COEFFS),
        T_BASE_CAMERA,
        Plane(a=0.0, b=0.0, c=1.0, d=0.0, frame="base"),
    )

    print("  构造期校验：通过")


# ============================================================
# 3. tube_to_grasp_pose 测试
# ============================================================

def make_tube_detection(
    center_base_mm,
    axis_base,
) -> TubeDetection:
    """由基座中心与长轴方向合成 TubeDetection。"""

    center_u, center_v = base_point_to_pixel(center_base_mm)

    axis = np.asarray(axis_base, dtype=np.float64)
    axis = axis / float(np.linalg.norm(axis))
    probe_base = np.asarray(center_base_mm) + axis * 10.0
    probe_u, probe_v = base_point_to_pixel(probe_base)

    axis_px = np.array([probe_u - center_u, probe_v - center_v], dtype=np.float64)
    axis_px /= float(np.linalg.norm(axis_px))

    return TubeDetection(
        tube_id="tube_0",
        center_u_px=center_u,
        center_v_px=center_v,
        axis_u=float(axis_px[0]),
        axis_v=float(axis_px[1]),
        angle_deg=float(math.degrees(math.atan2(axis_px[1], axis_px[0])) % 180.0),
        confidence=1.0,
    )


def test_tube_to_grasp_pose() -> None:
    """抓取位置、抬升高度与偏航角应符合约定。"""

    center_base = np.array([120.0, -60.0, TABLE_Z_MM], dtype=np.float64)
    axis_base = np.array([math.cos(math.radians(25.0)), math.sin(math.radians(25.0)), 0.0])

    tube = make_tube_detection(center_base, axis_base)
    geometry = TubeGeometry(diameter_mm=12.0, length_mm=100.0, grasp_offset_mm=20.0)

    pose = CONVERTER.tube_to_grasp_pose(tube, geometry)

    expected_axis = axis_base[:2] / float(np.linalg.norm(axis_base[:2]))
    expected_x = center_base[0] + expected_axis[0] * geometry.grasp_offset_mm
    expected_y = center_base[1] + expected_axis[1] * geometry.grasp_offset_mm
    expected_z = center_base[2] + geometry.diameter_mm / 2.0
    expected_yaw = math.degrees(math.atan2(expected_axis[1], expected_axis[0]))

    assert abs(pose.x_mm - expected_x) < 1e-3, (pose, expected_x)
    assert abs(pose.y_mm - expected_y) < 1e-3, (pose, expected_y)
    assert abs(pose.z_mm - expected_z) < 1e-3, (pose, expected_z)
    assert abs(angle_difference_deg(pose.rz_deg, expected_yaw)) < 1e-3, (
        pose.rz_deg,
        expected_yaw,
    )

    negative_pose = CONVERTER.tube_to_grasp_pose(
        tube,
        TubeGeometry(diameter_mm=12.0, length_mm=100.0, grasp_offset_mm=-20.0),
    )
    assert abs(
        negative_pose.x_mm - (center_base[0] - expected_axis[0] * 20.0)
    ) < 1e-3

    print("  tube_to_grasp_pose 位姿：通过")


def test_tube_to_grasp_pose_failures() -> None:
    """非法几何与退化长轴应抛出 ValueError。"""

    center_base = np.array([0.0, 0.0, TABLE_Z_MM], dtype=np.float64)
    tube = make_tube_detection(center_base, (1.0, 0.0, 0.0))

    expect_raises(
        ValueError,
        CONVERTER.tube_to_grasp_pose,
        tube,
        TubeGeometry(diameter_mm=0.0, length_mm=100.0, grasp_offset_mm=0.0),
    )
    expect_raises(
        ValueError,
        CONVERTER.tube_to_grasp_pose,
        tube,
        TubeGeometry(diameter_mm=10.0, length_mm=0.0, grasp_offset_mm=0.0),
    )
    expect_raises(
        ValueError,
        CONVERTER.tube_to_grasp_pose,
        tube,
        TubeGeometry(diameter_mm=10.0, length_mm=100.0, grasp_offset_mm=80.0),
    )

    degenerate_tube = TubeDetection(
        tube_id="tube_degenerate",
        center_u_px=tube.center_u_px,
        center_v_px=tube.center_v_px,
        axis_u=0.0,
        axis_v=0.0,
        angle_deg=0.0,
        confidence=1.0,
    )
    expect_raises(
        ValueError,
        CONVERTER.tube_to_grasp_pose,
        degenerate_tube,
        TubeGeometry(diameter_mm=10.0, length_mm=100.0, grasp_offset_mm=0.0),
    )

    print("  tube_to_grasp_pose 失败路径：通过")


# ============================================================
# 4. hole_pixel_to_pose 测试
# ============================================================

class FakeRackModel:
    """测试用试管架模型：只实现 hole_pose(hole_id)。"""

    def __init__(self, hole_poses: dict[str, HolePose]):
        self._hole_poses = hole_poses

    def hole_pose(self, hole_id: str) -> HolePose:
        return self._hole_poses[hole_id]


def test_hole_pixel_to_pose() -> None:
    """孔位像素命中模型孔位时返回标准 HolePose。"""

    hole_pose = HolePose(
        hole_id="hole_1",
        pose_base=Pose(
            x_mm=150.0,
            y_mm=-80.0,
            z_mm=TABLE_Z_MM,
            rx_deg=0.0,
            ry_deg=0.0,
            rz_deg=0.0,
        ),
    )
    rack_model = FakeRackModel({"hole_1": hole_pose})

    u_px, v_px = base_point_to_pixel((150.0, -80.0, TABLE_Z_MM))
    hole = HolePixel(hole_id="hole_1", u_px=u_px, v_px=v_px)

    result = CONVERTER.hole_pixel_to_pose(hole, rack_model)

    assert result is hole_pose
    assert result.hole_id == "hole_1"

    print("  hole_pixel_to_pose 命中：通过")


def test_hole_pixel_to_pose_failures() -> None:
    """模型缺失、孔位不存在与像素不匹配应正确报错。"""

    hole_pose = HolePose(
        hole_id="hole_1",
        pose_base=Pose(
            x_mm=150.0,
            y_mm=-80.0,
            z_mm=TABLE_Z_MM,
            rx_deg=0.0,
            ry_deg=0.0,
            rz_deg=0.0,
        ),
    )
    rack_model = FakeRackModel({"hole_1": hole_pose})

    expect_raises(FatalError, CONVERTER.hole_pixel_to_pose, HolePixel("hole_1", 0.0, 0.0), None)
    expect_raises(FatalError, CONVERTER.hole_pixel_to_pose, HolePixel("hole_1", 0.0, 0.0), object())

    expect_raises(
        ValueError,
        CONVERTER.hole_pixel_to_pose,
        HolePixel("hole_missing", 0.0, 0.0),
        rack_model,
    )

    u_px, v_px = base_point_to_pixel((150.0, -80.0, TABLE_Z_MM))
    offset_px = DEFAULT_HOLE_MATCH_TOLERANCE_MM * 10.0
    expect_raises(
        ValueError,
        CONVERTER.hole_pixel_to_pose,
        HolePixel("hole_1", u_px + offset_px, v_px),
        rack_model,
    )

    print("  hole_pixel_to_pose 失败路径：通过")


# ============================================================
# 5. 测试入口
# ============================================================

def main() -> int:
    """运行全部测试。"""

    print("开始 pixel_to_robot 离线测试")

    test_pixel_to_point_plane_roundtrip()
    test_pixel_to_point_depth_roundtrip()
    test_pixel_to_point_failures()
    test_constructor_validation()
    test_tube_to_grasp_pose()
    test_tube_to_grasp_pose_failures()
    test_hole_pixel_to_pose()
    test_hole_pixel_to_pose_failures()

    print("pixel_to_robot 离线测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
