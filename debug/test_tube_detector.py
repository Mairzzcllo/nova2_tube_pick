"""试管视觉检测离线测试。

覆盖：
    1. 合成平放试管（透明管身 + 彩色管盖）的数量、中心、长轴与方向
    2. 重复采集 10 帧中心抖动（阶段 5 门槛 < 5 px）
    3. 空场景、非法图像、确定性
    4. 无彩盖时的细长轮廓回退

用法：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/test_tube_detector.py
"""

import math
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import VisionError
from vision.tube_detector import detect_tubes


# ============================================================
# 1. 合成图像参数
# ============================================================

WIDTH = 640
HEIGHT = 480
BACKGROUND = (35, 35, 35)

TUBE_LENGTH_PX = 180.0
TUBE_WIDTH_PX = 18.0
CAP_RADIUS_FACTOR = 0.70

# (cap_u, cap_v, angle_deg, cap_bgr)
TUBE_SPECS = (
    (90.0, 250.0, 20.0, (0, 0, 255)),      # 红盖
    (340.0, 90.0, 100.0, (0, 180, 0)),     # 绿盖
    (520.0, 120.0, 155.0, (255, 0, 0)),    # 蓝盖
)


# ============================================================
# 2. 合成场景
# ============================================================

def _tube_ground_truth(spec):
    """由彩盖位置与角度计算试管真值。

    Returns:
        (center_u, center_v, axis_u, axis_v)
    """

    cap_u, cap_v, angle_deg, _ = spec
    angle_rad = math.radians(angle_deg)
    axis_u = math.cos(angle_rad)
    axis_v = math.sin(angle_rad)

    # 试管中心取彩盖圆心到管底端的中点
    center_offset = TUBE_LENGTH_PX * 0.5

    center_u = cap_u + axis_u * center_offset
    center_v = cap_v + axis_v * center_offset

    return center_u, center_v, axis_u, axis_v


def _draw_tube(image, spec):
    """绘制一支平放试管：半透明管身 + 一端彩色盖。"""

    cap_u, cap_v, angle_deg, cap_bgr = spec
    angle_rad = math.radians(angle_deg)

    axis = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)

    start = np.array([cap_u, cap_v], dtype=np.float32)
    end = start + axis * TUBE_LENGTH_PX
    half_width = TUBE_WIDTH_PX * 0.5

    corners = np.array(
        [
            start + normal * half_width,
            end + normal * half_width,
            end - normal * half_width,
            start - normal * half_width,
        ],
        dtype=np.int32,
    )

    # 管身填充与高光，模拟透明玻璃
    cv2.fillConvexPoly(image, corners, (185, 185, 185))
    highlight_offset = normal * half_width * 0.3
    cv2.line(
        image,
        tuple((start + highlight_offset).astype(int)),
        tuple((end + highlight_offset).astype(int)),
        (235, 235, 235),
        2,
    )

    # 管身轮廓，保证边缘证据稳定
    cv2.polylines(image, [corners], True, (30, 30, 30), 2)
    cv2.line(
        image,
        tuple((end - normal * half_width).astype(int)),
        tuple((end + normal * half_width).astype(int)),
        (30, 30, 30),
        2,
    )

    # 彩色管盖
    cap_radius = int(round(TUBE_WIDTH_PX * CAP_RADIUS_FACTOR))
    cv2.circle(image, (int(round(cap_u)), int(round(cap_v))), cap_radius, cap_bgr, -1)
    cv2.circle(
        image,
        (int(round(cap_u)), int(round(cap_v))),
        cap_radius,
        (20, 20, 20),
        1,
    )


def _make_scene(noise_seed=0, with_tubes=True, noise_sigma=3.0):
    """生成合成场景。"""

    rng = np.random.default_rng(noise_seed)
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)

    # 轻微亮度梯度
    ramp = np.linspace(0.9, 1.1, WIDTH, dtype=np.float32)
    image = np.clip(
        image.astype(np.float32) * ramp[None, :, None],
        0,
        255,
    ).astype(np.uint8)

    if with_tubes:
        for spec in TUBE_SPECS:
            _draw_tube(image, spec)

    # 干扰物：无彩盖的灰色矩形
    cv2.rectangle(image, (520, 20), (585, 70), (120, 120, 120), -1)

    noise = rng.normal(0.0, noise_sigma, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image


def _match_detections(detections, ground_truth):
    """将检测结果按最近中心匹配到真值。

    Returns:
        与 ground_truth 等长的检测列表；缺失位置为 None
    """

    matched = [None] * len(ground_truth)
    used = set()

    for index, (gt_u, gt_v, _, _) in enumerate(ground_truth):
        best_index = -1
        best_distance = float("inf")
        for det_index, detection in enumerate(detections):
            if det_index in used:
                continue
            distance = math.hypot(
                detection.center_u_px - gt_u,
                detection.center_v_px - gt_v,
            )
            if distance < best_distance:
                best_distance = distance
                best_index = det_index
        if best_index >= 0:
            matched[index] = detections[best_index]
            used.add(best_index)

    return matched


# ============================================================
# 3. 测试用例
# ============================================================

def test_detect_three_tubes():
    """三支试管数量、中心、长轴方向与角度。"""

    image = _make_scene(noise_seed=0)
    detections = detect_tubes(image)
    ground_truth = [_tube_ground_truth(spec) for spec in TUBE_SPECS]

    print("\n[测试] 三支试管检测")
    print(f"检测数量: {len(detections)}")
    for detection in detections:
        print(
            f"  {detection.tube_id}: "
            f"center=({detection.center_u_px:.2f}, {detection.center_v_px:.2f}), "
            f"axis=({detection.axis_u:.3f}, {detection.axis_v:.3f}), "
            f"angle={detection.angle_deg:.2f}, conf={detection.confidence:.3f}"
        )

    assert len(detections) == 3, f"应检测到 3 支试管，实际 {len(detections)}"

    matched = _match_detections(detections, ground_truth)
    for index, detection in enumerate(matched):
        assert detection is not None, f"第 {index} 支试管未匹配到检测结果"

        gt_u, gt_v, gt_axis_u, gt_axis_v = ground_truth[index]
        center_error = math.hypot(
            detection.center_u_px - gt_u,
            detection.center_v_px - gt_v,
        )
        dot = detection.axis_u * gt_axis_u + detection.axis_v * gt_axis_v

        print(
            f"  匹配 {index}: center_error={center_error:.2f}px, "
            f"axis_dot={dot:.4f}"
        )
        assert center_error < 8.0, f"第 {index} 支中心误差过大: {center_error:.2f}px"
        assert dot > 0.95, f"第 {index} 支长轴方向错误: dot={dot:.4f}"

        gt_angle = math.degrees(math.atan2(gt_axis_v, gt_axis_u)) % 180.0
        angle_error = abs(detection.angle_deg - gt_angle)
        angle_error = min(angle_error, 180.0 - angle_error)
        assert angle_error < 5.0, f"第 {index} 支角度误差过大: {angle_error:.2f}deg"


def test_center_jitter():
    """重复采集 10 帧，中心抖动应小于 5 px。"""

    ground_truth = [_tube_ground_truth(spec) for spec in TUBE_SPECS]
    per_tube_centers = [[] for _ in ground_truth]

    for frame in range(10):
        image = _make_scene(noise_seed=frame + 1)
        detections = detect_tubes(image)
        assert len(detections) == 3, f"第 {frame} 帧检测数量异常: {len(detections)}"

        matched = _match_detections(detections, ground_truth)
        for index, detection in enumerate(matched):
            assert detection is not None, f"第 {frame} 帧第 {index} 支未匹配"
            per_tube_centers[index].append(
                (detection.center_u_px, detection.center_v_px)
            )

    print("\n[测试] 连续 10 帧中心抖动")
    for index, centers in enumerate(per_tube_centers):
        array = np.array(centers, dtype=np.float32)
        mean = array.mean(axis=0)
        max_deviation = float(np.max(np.linalg.norm(array - mean, axis=1)))
        print(f"  第 {index} 支最大抖动: {max_deviation:.2f}px")
        assert max_deviation < 5.0, (
            f"第 {index} 支中心抖动超门槛: {max_deviation:.2f}px"
        )


def test_empty_scene():
    """空场景返回空列表。"""

    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    detections = detect_tubes(image)

    print(f"\n[测试] 空场景: 检测数量 {len(detections)}")
    assert detections == [], "空场景应返回空列表"


def test_invalid_images():
    """非法图像抛 VisionError。"""

    print("\n[测试] 非法图像")

    invalid_inputs = (
        None,
        np.zeros((HEIGHT, WIDTH), dtype=np.uint8),
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32),
    )

    for image in invalid_inputs:
        try:
            detect_tubes(image)
        except VisionError as error:
            print(f"  正确抛出 VisionError: {error}")
        else:
            raise AssertionError(f"非法图像未抛出 VisionError: {image is None}")


def test_deterministic():
    """同一图像重复检测结果与编号应完全一致。"""

    image = _make_scene(noise_seed=7)
    first = detect_tubes(image)
    second = detect_tubes(image)

    assert [item.tube_id for item in first] == [item.tube_id for item in second]
    for left, right in zip(first, second):
        assert left.center_u_px == right.center_u_px
        assert left.center_v_px == right.center_v_px
        assert left.axis_u == right.axis_u
        assert left.axis_v == right.axis_v
        assert left.confidence == right.confidence

    print("\n[测试] 确定性: 通过")


def test_elongated_contour_fallback():
    """无彩盖时按细长轮廓回退检测。"""

    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    center = (320, 240)
    rect = (center, (180.0, 22.0), 30.0)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, (150, 150, 150))
    cv2.polylines(image, [box], True, (30, 30, 30), 2)

    detections = detect_tubes(image)

    print(f"\n[测试] 无彩盖回退: 检测数量 {len(detections)}")
    for detection in detections:
        print(
            f"  {detection.tube_id}: "
            f"center=({detection.center_u_px:.2f}, {detection.center_v_px:.2f}), "
            f"conf={detection.confidence:.3f}"
        )

    assert len(detections) >= 1, "无彩盖细长目标未被回退检测到"

    best = min(
        detections,
        key=lambda item: math.hypot(
            item.center_u_px - center[0],
            item.center_v_px - center[1],
        ),
    )
    center_error = math.hypot(
        best.center_u_px - center[0],
        best.center_v_px - center[1],
    )
    assert center_error < 8.0, f"回退检测中心误差过大: {center_error:.2f}px"


# ============================================================
# 4. 主入口
# ============================================================

def main():
    """依次执行全部测试。"""

    test_detect_three_tubes()
    test_center_jitter()
    test_empty_scene()
    test_invalid_images()
    test_deterministic()
    test_elongated_contour_fallback()

    print("\n[PASS] 试管检测流程测试全部通过")


if __name__ == "__main__":
    main()
