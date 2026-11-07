"""试管视觉识别模块。

职责：
    1. 在已去畸变的 BGR 图像中检测平放桌面的试管
    2. 以彩色管盖为锚点，结合透明管身边缘证据估计中心与长轴
    3. 输出稳定的 TubeDetection 列表，供 pixel_to_robot 与任务层使用

说明：
    - 本模块不读取相机内参，输入图像默认已完成去畸变
    - 长轴方向定义为彩盖端 -> 管底端
    - 无目标返回空列表；非法图像抛 VisionError，不伪造检测结果
    - 透明管身轮廓较弱，算法同时使用彩盖色域、Canny 边缘与梯度证据
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, replace

import cv2
import numpy as np

# 兼容 python -m 与直接执行文件两种方式，确保能导入仓库内模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.errors import VisionError
from common.types import TubeDetection


# ============================================================
# 1. 默认参数
# ============================================================

# 彩盖 HSV 色域（OpenCV：H 0-179，S/V 0-255），按需增删
CAP_HSV_RANGES = (
    ((0, 110, 60), (12, 255, 255)),      # 红（低色相段）
    ((160, 110, 60), (179, 255, 255)),   # 红（高色相段）
    ((15, 110, 60), (38, 255, 255)),     # 橙 / 黄
    ((40, 90, 60), (85, 255, 255)),      # 绿
    ((90, 90, 60), (135, 255, 255)),     # 蓝
    ((140, 80, 60), (165, 255, 255)),    # 紫
)

# 彩盖面积过滤：相对整图面积与绝对像素下限
CAP_MIN_AREA_RATIO = 0.00012
CAP_MIN_AREA_PX = 20.0
CAP_MAX_AREA_RATIO = 0.03

# 彩盖最小外接矩形长宽比上限
CAP_MAX_ASPECT = 5.0

# 彩盖形态学核尺寸
CAP_OPEN_KERNEL = (3, 3)
CAP_CLOSE_KERNEL = (5, 5)

# 图像预处理参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BLUR_KSIZE = (5, 5)

# 边缘证据参数
CANNY_LOW = 40
CANNY_HIGH = 120
GRAD_LOW = 40.0
GRAD_HIGH = 200.0

# 管身几何约束：长度相对图像最长边，搜索半宽相对图像最短边
MIN_BODY_LENGTH_RATIO = 0.05
MIN_SEARCH_LENGTH_RATIO = 0.12
MAX_BODY_LENGTH_RATIO = 0.75
SEARCH_HALF_WIDTH_RATIO = 0.06
MAX_BODY_WIDTH_RATIO = 0.15
MIN_ASPECT_RATIO = 1.8

# 长轴搜索参数
AXIS_ANGLE_STEP_DEG = 3.0
AXIS_OFFSET_COUNT = 13
AXIS_MIN_SCORE = 0.05

# 边缘证据阈值
EVIDENCE_THRESHOLD = 0.35

# 置信度组成权重与门槛
CONFIDENCE_EDGE_WEIGHT = 0.45
CONFIDENCE_ASPECT_WEIGHT = 0.35
CONFIDENCE_LENGTH_WEIGHT = 0.20
CONFIDENCE_EDGE_FULL_SCORE = 0.20
CONFIDENCE_LENGTH_FULL_RATIO = 0.25
MIN_CONFIDENCE = 0.30

# 去重距离相对图像最短边
DEDUP_DISTANCE_RATIO = 0.04


# ============================================================
# 2. 内部数据结构
# ============================================================

@dataclass(frozen=True)
class _CapCandidate:
    """彩盖候选。"""

    u_px: float
    v_px: float
    radius_px: float
    area_px: float
    score: float


@dataclass(frozen=True)
class _BodyGeometry:
    """管身几何估计结果。"""

    center_u_px: float
    center_v_px: float
    axis_u: float
    axis_v: float
    length_px: float
    width_px: float
    edge_score: float


@dataclass(frozen=True)
class _BodyCandidate:
    """由轮廓得到的管身候选。"""

    center_u_px: float
    center_v_px: float
    axis_u: float
    axis_v: float
    length_px: float
    width_px: float
    confidence: float


# ============================================================
# 3. 输入校验与预处理
# ============================================================

def _validate_image(image) -> None:
    """校验输入图像。

    Args:
        image: 待检测图像

    Raises:
        VisionError: 图像为空、类型/尺寸/通道/dtype 非法
    """

    if image is None:
        raise VisionError("输入图像为空", recoverable=False)

    if not isinstance(image, np.ndarray):
        raise VisionError(
            f"输入图像类型非法: {type(image).__name__}",
            recoverable=False,
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise VisionError(
            f"输入图像必须为三通道 BGR，当前 shape={image.shape}",
            recoverable=False,
        )

    if image.dtype != np.uint8:
        raise VisionError(
            f"输入图像 dtype 必须为 uint8，当前 {image.dtype}",
            recoverable=False,
        )

    if image.size == 0:
        raise VisionError("输入图像尺寸为 0", recoverable=False)


def _preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """生成增强灰度图与 HSV 图。

    Args:
        image: 三通道 BGR 图像

    Returns:
        (gray, hsv)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

    return gray, hsv


# ============================================================
# 4. 边缘证据
# ============================================================

def _build_edge_evidence(gray: np.ndarray) -> np.ndarray:
    """构建浮点边缘证据图，范围 [0, 1]。

    Args:
        gray: 增强灰度图

    Returns:
        与 gray 同尺寸的 float32 证据图
    """

    canny = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    canny_score = canny.astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    grad_score = np.clip(
        (magnitude - GRAD_LOW) / (GRAD_HIGH - GRAD_LOW),
        0.0,
        1.0,
    )
    grad_score = cv2.GaussianBlur(grad_score, (3, 3), 0)

    return np.maximum(canny_score, grad_score)


def _bilinear_sample(
    evidence: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """在证据图上做双线性采样，越界返回 0。

    Args:
        evidence: 证据图
        u: 横坐标数组
        v: 纵坐标数组

    Returns:
        与 u/v 同形状的采样值
    """

    height, width = evidence.shape
    inside = (
        (u >= 0.0)
        & (u < width - 1.0)
        & (v >= 0.0)
        & (v < height - 1.0)
    )

    u_clip = np.clip(u, 0.0, width - 1.001)
    v_clip = np.clip(v, 0.0, height - 1.001)

    u0 = np.floor(u_clip).astype(np.int32)
    v0 = np.floor(v_clip).astype(np.int32)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)

    du = (u_clip - u0).astype(np.float32)
    dv = (v_clip - v0).astype(np.float32)

    value = (
        evidence[v0, u0] * (1.0 - du) * (1.0 - dv)
        + evidence[v0, u1] * du * (1.0 - dv)
        + evidence[v1, u0] * (1.0 - du) * dv
        + evidence[v1, u1] * du * dv
    )

    return value * inside.astype(np.float32)


# ============================================================
# 5. 彩盖检测
# ============================================================

def _detect_caps(hsv: np.ndarray) -> list[_CapCandidate]:
    """基于 HSV 色域检测彩色管盖。

    Args:
        hsv: HSV 图像

    Returns:
        按分数降序排列的彩盖候选列表
    """

    height, width = hsv.shape[:2]
    image_area = float(height * width)

    mask = np.zeros((height, width), dtype=np.uint8)
    for lower, upper in CAP_HSV_RANGES:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            ),
        )

    open_kernel = np.ones(CAP_OPEN_KERNEL, dtype=np.uint8)
    close_kernel = np.ones(CAP_CLOSE_KERNEL, dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = max(CAP_MIN_AREA_PX, CAP_MIN_AREA_RATIO * image_area)
    max_area = CAP_MAX_AREA_RATIO * image_area
    ref_area = max(min_area * 4.0, 1.0)

    candidates: list[_CapCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        (center_u, center_v), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        short_side = min(rect_w, rect_h)
        long_side = max(rect_w, rect_h)
        if short_side <= 0.0:
            continue

        aspect = long_side / short_side
        if aspect > CAP_MAX_ASPECT:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0
        if perimeter > 0.0:
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        area_score = min(1.0, area / ref_area)
        aspect_score = max(0.0, 1.0 - aspect / CAP_MAX_ASPECT)
        score = 0.5 * min(1.0, circularity) + 0.3 * area_score + 0.2 * aspect_score

        candidates.append(
            _CapCandidate(
                u_px=float(center_u),
                v_px=float(center_v),
                radius_px=0.5 * long_side,
                area_px=area,
                score=score,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return _deduplicate_caps(candidates)


def _deduplicate_caps(candidates: list[_CapCandidate]) -> list[_CapCandidate]:
    """合并距离过近的彩盖候选。

    Args:
        candidates: 已按分数降序排列的候选

    Returns:
        去重后的候选
    """

    kept: list[_CapCandidate] = []
    for candidate in candidates:
        duplicated = False
        for other in kept:
            distance = math.hypot(
                candidate.u_px - other.u_px,
                candidate.v_px - other.v_px,
            )
            if distance < max(candidate.radius_px, other.radius_px):
                duplicated = True
                break
        if not duplicated:
            kept.append(candidate)
    return kept


# ============================================================
# 6. 长轴搜索与几何精修
# ============================================================

def _search_axis(
    evidence: np.ndarray,
    cap_u: float,
    cap_v: float,
    cap_radius_px: float,
    max_length_px: float,
) -> tuple[float, float, float, float]:
    """以彩盖为锚点搜索管身方向与长度。

    Args:
        evidence: 边缘证据图
        cap_u: 彩盖中心 u
        cap_v: 彩盖中心 v
        cap_radius_px: 彩盖半径
        max_length_px: 最大搜索长度

    Returns:
        (score, axis_u, axis_v, body_length_px)；axis 指向管身方向
    """

    height, width = evidence.shape
    short_side = float(min(height, width))

    half_width = max(4.0, short_side * SEARCH_HALF_WIDTH_RATIO)
    offsets = np.linspace(
        -half_width,
        half_width,
        AXIS_OFFSET_COUNT,
        dtype=np.float32,
    )

    max_length = min(max_length_px, math.hypot(height, width))
    start = max(cap_radius_px * 0.9, 3.0)
    step = max(2.0, short_side * 0.006)
    distances = np.arange(start, max_length, step, dtype=np.float32)
    if distances.size < 5:
        return 0.0, 0.0, 0.0, 0.0

    # 至少连续覆盖一段最小管身长度，避免把彩盖自身边缘当成管身
    min_samples = max(
        5,
        int(round(MIN_SEARCH_LENGTH_RATIO * max(height, width) / step)),
    )

    best_score = 0.0
    best_u = 0.0
    best_v = 0.0
    best_length = 0.0

    for angle_deg in np.arange(0.0, 360.0, AXIS_ANGLE_STEP_DEG):
        angle_rad = math.radians(float(angle_deg))
        direction_u = math.cos(angle_rad)
        direction_v = math.sin(angle_rad)
        normal_u = -direction_v
        normal_v = direction_u

        sampled = []
        for offset in offsets:
            sample_u = cap_u + direction_u * distances + normal_u * offset
            sample_v = cap_v + direction_v * distances + normal_v * offset
            sampled.append(_bilinear_sample(evidence, sample_u, sample_v))

        per_distance = np.max(np.stack(sampled, axis=0), axis=0)

        # 用前缀均值寻找“从彩盖出发连续存在管身证据”的最优端点，
        # 避免远处其他试管把平均得分稀释或拉长
        prefix_mean = np.cumsum(per_distance) / np.arange(
            1,
            per_distance.size + 1,
            dtype=np.float32,
        )
        prefix_mean[: min_samples - 1] = -1.0
        best_index = int(np.argmax(prefix_mean))
        best_mean = float(prefix_mean[best_index])
        if best_mean <= 0.0:
            continue

        segment = per_distance[: best_index + 1]
        coverage = float(np.mean(segment > 0.15))
        score = best_mean * (0.5 + 0.5 * coverage)

        if score > best_score:
            best_score = score
            best_u = direction_u
            best_v = direction_v
            best_length = float(distances[best_index])

    return best_score, best_u, best_v, best_length


def _refine_geometry(
    evidence: np.ndarray,
    cap: _CapCandidate,
    axis_u: float,
    axis_v: float,
    edge_score: float,
    max_length_px: float,
    body_length_px: float,
) -> _BodyGeometry | None:
    """在长轴走廊内精修管身中心、方向与尺寸。

    Args:
        evidence: 边缘证据图
        cap: 彩盖候选
        axis_u: 初始长轴 u 分量
        axis_v: 初始长轴 v 分量
        edge_score: 长轴搜索得分
        max_length_px: 最大管身长度
        body_length_px: 长轴搜索得到的连续管身长度

    Returns:
        管身几何；证据不足返回 None
    """

    height, width = evidence.shape
    short_side = float(min(height, width))
    half_width = max(4.0, short_side * SEARCH_HALF_WIDTH_RATIO)
    max_length = min(
        max_length_px,
        max(body_length_px * 1.15 + cap.radius_px, cap.radius_px * 2.0),
        math.hypot(height, width),
    )

    axis = np.array([axis_u, axis_v], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None
    axis /= norm
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)

    v_coords, u_coords = np.nonzero(evidence > EVIDENCE_THRESHOLD)
    if u_coords.size < 20:
        return None

    relative = np.stack(
        [u_coords - cap.u_px, v_coords - cap.v_px],
        axis=1,
    ).astype(np.float32)

    along = relative @ axis
    across = relative @ normal

    min_along = cap.radius_px * 0.6
    corridor = (
        (along >= min_along)
        & (along <= max_length)
        & (np.abs(across) <= half_width)
    )
    if int(np.count_nonzero(corridor)) < 10:
        return None

    points = relative[corridor]
    weights = evidence[v_coords[corridor], u_coords[corridor]].astype(np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return None

    weighted_mean = (points * weights[:, None]).sum(axis=0) / weight_sum
    centered = points - weighted_mean
    covariance = (centered * weights[:, None]).T @ centered / weight_sum

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    refined_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    refined_axis /= float(np.linalg.norm(refined_axis))

    if float(np.dot(refined_axis, axis)) < 0.0:
        refined_axis = -refined_axis

    # 精修方向不应相对搜索结果发生大角度偏转
    if float(np.dot(refined_axis, axis)) < 0.9:
        return None

    refined_normal = np.array(
        [-refined_axis[1], refined_axis[0]],
        dtype=np.float32,
    )
    refined_along = relative @ refined_axis
    refined_across = relative @ refined_normal

    corridor = (
        (refined_along >= min_along)
        & (refined_along <= max_length)
        & (np.abs(refined_across) <= half_width)
    )
    if int(np.count_nonzero(corridor)) < 10:
        return None

    far_distance = float(np.percentile(refined_along[corridor], 98.0))
    width_px = 2.0 * float(np.percentile(np.abs(refined_across[corridor]), 95.0))

    if far_distance <= 0.0:
        return None

    length_px = far_distance
    center_along = 0.5 * far_distance

    center_u = cap.u_px + refined_axis[0] * center_along
    center_v = cap.v_px + refined_axis[1] * center_along

    return _BodyGeometry(
        center_u_px=float(center_u),
        center_v_px=float(center_v),
        axis_u=float(refined_axis[0]),
        axis_v=float(refined_axis[1]),
        length_px=float(length_px),
        width_px=float(max(width_px, 1.0)),
        edge_score=float(edge_score),
    )


# ============================================================
# 7. 候选过滤、置信度与去重
# ============================================================

def _compute_confidence(
    geometry: _BodyGeometry,
    cap: _CapCandidate,
    image_shape: tuple[int, int],
) -> float:
    """计算检测置信度。

    Args:
        geometry: 管身几何
        cap: 彩盖候选
        image_shape: (height, width)

    Returns:
        [0, 1] 置信度
    """

    height, width = image_shape
    short_side = float(min(height, width))

    aspect = geometry.length_px / max(geometry.width_px, 1.0)

    edge_score = min(
        1.0,
        geometry.edge_score / CONFIDENCE_EDGE_FULL_SCORE,
    )
    aspect_score = min(1.0, aspect / (2.0 * MIN_ASPECT_RATIO))
    length_score = min(
        1.0,
        geometry.length_px / (CONFIDENCE_LENGTH_FULL_RATIO * max(height, width)),
    )
    cap_score = min(1.0, cap.score)
    cap_size_score = min(1.0, cap.area_px / max(short_side * short_side * 0.001, 1.0))

    confidence = (
        CONFIDENCE_EDGE_WEIGHT * edge_score
        + CONFIDENCE_ASPECT_WEIGHT * aspect_score
        + CONFIDENCE_LENGTH_WEIGHT * length_score
    )
    confidence = 0.8 * confidence + 0.2 * (0.5 * cap_score + 0.5 * cap_size_score)

    return float(np.clip(confidence, 0.0, 1.0))


def _is_geometry_valid(
    geometry: _BodyGeometry,
    image_shape: tuple[int, int],
) -> bool:
    """检查管身几何是否满足基本约束。"""

    height, width = image_shape
    long_side = float(max(height, width))
    short_side = float(min(height, width))

    if geometry.length_px < MIN_BODY_LENGTH_RATIO * long_side:
        return False
    if geometry.width_px > MAX_BODY_WIDTH_RATIO * short_side:
        return False

    aspect = geometry.length_px / max(geometry.width_px, 1.0)
    if aspect < MIN_ASPECT_RATIO:
        return False

    return True


def _deduplicate_detections(
    detections: list[TubeDetection],
    image_shape: tuple[int, int],
) -> list[TubeDetection]:
    """按中心距离合并重复检测，保留高置信度结果。"""

    height, width = image_shape
    min_distance = DEDUP_DISTANCE_RATIO * float(min(height, width))

    kept: list[TubeDetection] = []
    for detection in sorted(
        detections,
        key=lambda item: item.confidence,
        reverse=True,
    ):
        duplicated = False
        for other in kept:
            distance = math.hypot(
                detection.center_u_px - other.center_u_px,
                detection.center_v_px - other.center_v_px,
            )
            if distance < min_distance:
                duplicated = True
                break
        if not duplicated:
            kept.append(detection)

    return kept


def _assign_ids(detections: list[TubeDetection]) -> list[TubeDetection]:
    """按图像位置稳定排序并生成 tube_id。"""

    ordered = sorted(
        detections,
        key=lambda item: (item.center_u_px, item.center_v_px),
    )
    return [
        replace(detection, tube_id=f"tube_{index:02d}")
        for index, detection in enumerate(ordered)
    ]


# ============================================================
# 8. 细长管身轮廓检测与彩盖关联
# ============================================================

def _detect_body_candidates(
    evidence: np.ndarray,
    image_shape: tuple[int, int],
) -> list[_BodyCandidate]:
    """从边缘证据中检测细长管身候选。

    Args:
        evidence: 边缘证据图
        image_shape: (height, width)

    Returns:
        管身候选列表
    """

    height, width = image_shape
    long_side = float(max(height, width))
    short_side = float(min(height, width))

    binary = (evidence > EVIDENCE_THRESHOLD).astype(np.uint8) * 255
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[_BodyCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < CAP_MIN_AREA_PX:
            continue

        (center_u, center_v), (rect_w, rect_h), rect_angle_deg = cv2.minAreaRect(
            contour
        )
        short_width = min(rect_w, rect_h)
        long_length = max(rect_w, rect_h)
        if short_width <= 0.0:
            continue

        aspect = long_length / short_width
        if aspect < MIN_ASPECT_RATIO:
            continue
        if long_length < MIN_BODY_LENGTH_RATIO * long_side:
            continue
        if short_width > MAX_BODY_WIDTH_RATIO * short_side:
            continue

        angle_rad = math.radians(rect_angle_deg)
        if rect_w >= rect_h:
            axis_u = math.cos(angle_rad)
            axis_v = math.sin(angle_rad)
        else:
            axis_u = math.cos(angle_rad + math.pi / 2.0)
            axis_v = math.sin(angle_rad + math.pi / 2.0)

        # 没有彩盖时无法判断首尾，统一符号保证确定性
        if axis_u < 0.0 or (abs(axis_u) < 1e-6 and axis_v < 0.0):
            axis_u = -axis_u
            axis_v = -axis_v

        confidence = min(
            1.0,
            0.5 * min(1.0, aspect / (2.0 * MIN_ASPECT_RATIO))
            + 0.5 * min(1.0, long_length / (0.3 * long_side)),
        )

        candidates.append(
            _BodyCandidate(
                center_u_px=float(center_u),
                center_v_px=float(center_v),
                axis_u=float(axis_u),
                axis_v=float(axis_v),
                length_px=float(long_length),
                width_px=float(short_width),
                confidence=float(confidence),
            )
        )

    return candidates


def _match_caps_to_bodies(
    bodies: list[_BodyCandidate],
    caps: list[_CapCandidate],
) -> tuple[list[TubeDetection], set[int], set[int]]:
    """将彩盖与最近的管身端点配对，并按彩盖方向确定首尾。

    Args:
        bodies: 管身候选
        caps: 彩盖候选

    Returns:
        (检测结果, 已使用管身索引集合, 已使用彩盖索引集合)
    """

    pair_candidates: list[tuple[float, int, int, bool]] = []

    for body_index, body in enumerate(bodies):
        axis = np.array([body.axis_u, body.axis_v], dtype=np.float32)
        half_length = body.length_px * 0.5
        center = np.array(
            [body.center_u_px, body.center_v_px],
            dtype=np.float32,
        )
        end_minus = center - axis * half_length
        end_plus = center + axis * half_length

        for cap_index, cap in enumerate(caps):
            cap_point = np.array([cap.u_px, cap.v_px], dtype=np.float32)
            distance_minus = float(np.linalg.norm(cap_point - end_minus))
            distance_plus = float(np.linalg.norm(cap_point - end_plus))
            distance = min(distance_minus, distance_plus)

            max_distance = max(3.0 * cap.radius_px, 0.15 * body.length_px)
            if distance <= max_distance:
                cap_at_plus = distance_plus <= distance_minus
                pair_candidates.append(
                    (distance, body_index, cap_index, cap_at_plus)
                )

    pair_candidates.sort(key=lambda item: item[0])

    detections: list[TubeDetection] = []
    used_bodies: set[int] = set()
    used_caps: set[int] = set()

    for _, body_index, cap_index, cap_at_plus in pair_candidates:
        if body_index in used_bodies or cap_index in used_caps:
            continue

        body = bodies[body_index]
        cap = caps[cap_index]

        cap_point = np.array([cap.u_px, cap.v_px], dtype=np.float32)
        center_point = np.array(
            [body.center_u_px, body.center_v_px],
            dtype=np.float32,
        )
        axis = np.array([body.axis_u, body.axis_v], dtype=np.float32)
        half_length = body.length_px * 0.5
        end_minus = center_point - axis * half_length
        end_plus = center_point + axis * half_length

        # 远离彩盖的一端为管底端，由此确定首尾方向
        far_end = end_minus if cap_at_plus else end_plus
        direction = far_end - cap_point
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-6:
            continue
        direction /= direction_norm
        axis_u = float(direction[0])
        axis_v = float(direction[1])

        # 以彩盖圆心到管底端的中心作为试管中心，避免彩盖是否
        # 参与边缘轮廓造成中心偏移
        center_u = cap.u_px + axis_u * direction_norm * 0.5
        center_v = cap.v_px + axis_v * direction_norm * 0.5

        angle_deg = math.degrees(math.atan2(axis_v, axis_u)) % 180.0
        confidence = min(
            1.0,
            0.85 * body.confidence + 0.15 * cap.score,
        )

        detections.append(
            TubeDetection(
                tube_id="",
                center_u_px=float(center_u),
                center_v_px=float(center_v),
                axis_u=axis_u,
                axis_v=axis_v,
                angle_deg=float(angle_deg),
                confidence=float(confidence),
            )
        )

        used_bodies.add(body_index)
        used_caps.add(cap_index)

    return detections, used_bodies, used_caps


# ============================================================
# 9. 主检测流程
# ============================================================

def detect_tubes(image: np.ndarray) -> list[TubeDetection]:
    """检测图像中的试管。

    Args:
        image: 已去畸变的三通道 BGR 图像

    Returns:
        TubeDetection 列表；无目标返回空列表

    Raises:
        VisionError: 图像非法或内部处理失败
    """

    _validate_image(image)

    try:
        gray, hsv = _preprocess(image)
        edge_map = _build_edge_evidence(gray)
    except cv2.error as error:
        raise VisionError(
            f"视觉预处理失败: {error}",
            recoverable=True,
        ) from error

    image_shape = gray.shape[:2]
    height, width = image_shape
    max_length_px = MAX_BODY_LENGTH_RATIO * float(max(height, width))

    caps = _detect_caps(hsv)
    bodies = _detect_body_candidates(edge_map, image_shape)

    # 主路径：细长轮廓提供管身几何，彩盖提供首尾方向
    detections, used_bodies, used_caps = _match_caps_to_bodies(bodies, caps)

    # 未关联彩盖的细长轮廓仍作为检测结果保留
    for body_index, body in enumerate(bodies):
        if body_index in used_bodies:
            continue

        angle_deg = math.degrees(
            math.atan2(body.axis_v, body.axis_u)
        ) % 180.0
        detections.append(
            TubeDetection(
                tube_id="",
                center_u_px=body.center_u_px,
                center_v_px=body.center_v_px,
                axis_u=body.axis_u,
                axis_v=body.axis_v,
                angle_deg=float(angle_deg),
                confidence=body.confidence,
            )
        )

    # 回退路径：轮廓不完整但能检测到彩盖时，用锚点搜索估计管身
    for cap_index, cap in enumerate(caps):
        if cap_index in used_caps:
            continue

        search_score, axis_u, axis_v, body_length_px = _search_axis(
            edge_map,
            cap.u_px,
            cap.v_px,
            cap.radius_px,
            max_length_px,
        )
        if search_score < AXIS_MIN_SCORE:
            continue

        geometry = _refine_geometry(
            edge_map,
            cap,
            axis_u,
            axis_v,
            search_score,
            max_length_px,
            body_length_px,
        )
        if geometry is None:
            continue
        if not _is_geometry_valid(geometry, image_shape):
            continue

        confidence = _compute_confidence(geometry, cap, image_shape)
        if confidence < MIN_CONFIDENCE:
            continue

        angle_deg = math.degrees(
            math.atan2(geometry.axis_v, geometry.axis_u)
        ) % 180.0

        detections.append(
            TubeDetection(
                tube_id="",
                center_u_px=geometry.center_u_px,
                center_v_px=geometry.center_v_px,
                axis_u=geometry.axis_u,
                axis_v=geometry.axis_v,
                angle_deg=float(angle_deg),
                confidence=confidence,
            )
        )

    detections = _deduplicate_detections(detections, image_shape)
    return _assign_ids(detections)


# ============================================================
# 10. 调试可视化与命令行入口
# ============================================================

def _draw_debug(image: np.ndarray, detections: list[TubeDetection]) -> np.ndarray:
    """在图像上绘制检测结果。"""

    canvas = image.copy()
    for detection in detections:
        center = (int(round(detection.center_u_px)), int(round(detection.center_v_px)))
        axis = np.array([detection.axis_u, detection.axis_v], dtype=np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-6)

        half_length = 60
        start = (
            int(round(center[0] - axis[0] * half_length)),
            int(round(center[1] - axis[1] * half_length)),
        )
        end = (
            int(round(center[0] + axis[0] * half_length)),
            int(round(center[1] + axis[1] * half_length)),
        )

        cv2.circle(canvas, center, 4, (0, 255, 0), -1)
        cv2.arrowedLine(canvas, start, end, (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(
            canvas,
            f"{detection.tube_id} {detection.confidence:.2f}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main() -> int:
    """命令行入口：读取图像并输出检测结果。"""

    parser = argparse.ArgumentParser(description="试管视觉检测")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument(
        "--debug-dir",
        default="data/debug",
        help="调试图像输出目录，默认 data/debug",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] 无法读取图像: {args.image}")
        return 1

    try:
        detections = detect_tubes(image)
    except VisionError as error:
        print(f"[ERROR] 检测失败: {error}")
        return 1

    print(f"检测到 {len(detections)} 支试管")
    for detection in detections:
        print(
            f"  {detection.tube_id}: "
            f"center=({detection.center_u_px:.1f}, {detection.center_v_px:.1f}) px, "
            f"axis=({detection.axis_u:.3f}, {detection.axis_v:.3f}), "
            f"angle={detection.angle_deg:.1f} deg, "
            f"confidence={detection.confidence:.2f}"
        )

    os.makedirs(args.debug_dir, exist_ok=True)
    debug_path = os.path.join(args.debug_dir, "tube_detector_debug.png")
    cv2.imwrite(debug_path, _draw_debug(image, detections))
    print(f"调试图像已保存: {debug_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
