
"""
相机内参标定模块

功能：
    1. 读取指定目录中的棋盘格图片
    2. 检测棋盘格内角点
    3. 亚像素级角点优化
    4. 使用 OpenCV calibrateCamera() 求解内参和畸变系数
    5. 计算整体及逐张图片的重投影误差
    6. 保存到 config/camera_intrinsics.npz

坐标单位：
    square_size 使用 mm 时，棋盘格三维坐标也使用 mm。
    相机内参 K 和畸变系数 D 不受长度单位影响。
"""

import argparse
import glob
import os

import cv2
import numpy as np


# ============================================================
# 1. 默认标定参数
# ============================================================

# 棋盘格内角点数量：(列数, 行数)
PATTERN_SIZE = (9, 6)

# 每个方格的实际边长，单位 mm
SQUARE_SIZE = 25.0

# 至少需要多少张成功检测到棋盘格的图片
MIN_VIEWS = 10

# 亚像素角点优化条件
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ============================================================
# 2. 创建棋盘格三维坐标
# ============================================================

def create_object_points(pattern_size, square_size):
    """
    创建棋盘格内角点对应的三维世界坐标。

    Args:
        pattern_size: (列数, 行数)
        square_size: 每个格子的实际边长

    Returns:
        shape = (N, 3) 的三维坐标数组
    """

    cols, rows = pattern_size

    if cols < 2 or rows < 2:
        raise ValueError("棋盘格行列内角点数量必须至少为 2")

    if square_size <= 0:
        raise ValueError("square_size 必须大于 0")

    object_points = np.zeros(
        (rows * cols, 3),
        dtype=np.float32
    )

    # 棋盘格位于 Z=0 平面
    object_points[:, :2] = (
        np.mgrid[0:cols, 0:rows]
        .T.reshape(-1, 2)
        * square_size
    )

    return object_points


# ============================================================
# 3. 查找图片
# ============================================================

def get_image_paths(image_dir):
    """读取目录中的 JPG、PNG、BMP 图片。"""

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ]

    image_paths = []

    for extension in extensions:
        image_paths.extend(
            glob.glob(os.path.join(image_dir, extension))
        )

    # 按文件名排序
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(
            f"目录中没有找到图片：{image_dir}"
        )

    return image_paths


# ============================================================
# 4. 检测棋盘格角点
# ============================================================

def detect_chessboard(image, pattern_size):
    """
    检测棋盘格内角点，并进行亚像素优化。

    Returns:
        success: 是否检测成功
        corners: 棋盘格角点
        gray: 灰度图
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    success, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags
    )

    if not success:
        return False, None, gray

    # 亚像素级角点优化
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        SUBPIX_CRITERIA
    )

    return True, corners, gray


# ============================================================
# 5. 计算单张图片的重投影误差
# ============================================================

def calculate_reprojection_error(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs
):
    """
    将三维棋盘格点重新投影到图像，
    比较预测像素坐标与实际检测角点。

    Returns:
        当前图片的 RMS 重投影误差，单位 pixel
    """

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)
    actual_points = image_points.reshape(-1, 2)

    errors = projected_points - actual_points

    # 每个角点的二维欧氏距离平方
    squared_errors = np.sum(errors ** 2, axis=1)

    # 所有角点的 RMS 误差
    rms = np.sqrt(np.mean(squared_errors))

    return float(rms)


# ============================================================
# 6. 执行相机内参标定
# ============================================================

def calibrate_camera(
    image_dir,
    output_path,
    pattern_size=PATTERN_SIZE,
    square_size=SQUARE_SIZE,
    min_views=MIN_VIEWS,
    debug_dir=None
):
    """
    执行相机内参标定并保存结果。

    Args:
        image_dir: 棋盘格图片目录
        output_path: NPZ 输出路径
        pattern_size: 棋盘格内角点数量 (列, 行)
        square_size: 方格实际边长
        min_views: 最少有效图片数量
        debug_dir: 可选，角点检测预览图输出目录

    Returns:
        camera_matrix, dist_coeffs, rms
    """

    print("=" * 60)
    print("开始相机内参标定")
    print("=" * 60)

    # --------------------------------------------------------
    # 6.1 检查图片目录
    # --------------------------------------------------------

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"图片目录不存在：{image_dir}"
        )

    image_paths = get_image_paths(image_dir)

    print(f"找到图片数量：{len(image_paths)}")
    print(f"棋盘格内角点：{pattern_size}")
    print(f"方格边长：{square_size} mm")

    # --------------------------------------------------------
    # 6.2 创建棋盘格三维坐标
    # --------------------------------------------------------

    object_template = create_object_points(
        pattern_size,
        square_size
    )

    object_points_list = []
    image_points_list = []

    image_size = None

    # --------------------------------------------------------
    # 6.3 逐张读取图片并检测角点
    # --------------------------------------------------------

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    for index, image_path in enumerate(image_paths):

        image = cv2.imread(image_path)

        if image is None:
            print(f"[跳过] 图片无法读取：{image_path}")
            continue

        height, width = image.shape[:2]
        current_size = (width, height)

        if image_size is None:
            image_size = current_size

        elif current_size != image_size:
            raise ValueError(
                "标定图片分辨率不一致："
                f"{image_path} 为 {current_size}，"
                f"预期为 {image_size}"
            )

        success, corners, gray = detect_chessboard(
            image,
            pattern_size
        )

        if success:

            object_points_list.append(
                object_template.copy()
            )

            image_points_list.append(corners)

            print(
                f"[成功] {os.path.basename(image_path)}"
            )

            if debug_dir is not None:

                preview = image.copy()

                cv2.drawChessboardCorners(
                    preview,
                    pattern_size,
                    corners,
                    True
                )

                preview_path = os.path.join(
                    debug_dir,
                    f"{index:03d}_corners.jpg"
                )

                cv2.imwrite(preview_path, preview)

        else:
            print(
                f"[失败] 未检测到棋盘格："
                f"{os.path.basename(image_path)}"
            )

    valid_views = len(object_points_list)

    print("-" * 60)
    print(f"有效图片数量：{valid_views}")

    if valid_views < min_views:
        raise RuntimeError(
            f"有效图片不足：{valid_views} 张，"
            f"至少需要 {min_views} 张。"
            "请增加不同角度的棋盘格图片。"
        )

    # --------------------------------------------------------
    # 6.4 计算相机内参和畸变参数
    # --------------------------------------------------------

    print("正在计算相机内参...")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points_list,
            image_points_list,
            image_size,
            None,
            None
        )
    )

    # --------------------------------------------------------
    # 6.5 计算逐张图片的重投影误差
    # --------------------------------------------------------

    per_view_errors = []

    for i in range(valid_views):

        error = calculate_reprojection_error(
            object_points_list[i],
            image_points_list[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        per_view_errors.append(error)

    per_view_errors = np.asarray(
        per_view_errors,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # 6.6 保存标定结果
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        output_path,

        K=camera_matrix,
        D=dist_coeffs,

        image_width=image_size[0],
        image_height=image_size[1],

        rms=rms,
        per_view_errors=per_view_errors,

        pattern_cols=pattern_size[0],
        pattern_rows=pattern_size[1],
        square_size=square_size,

        valid_views=valid_views
    )

    # --------------------------------------------------------
    # 6.7 输出结果
    # --------------------------------------------------------

    print("=" * 60)
    print("相机内参标定完成")
    print("=" * 60)

    print("\n相机内参矩阵 K：")
    print(camera_matrix)

    print("\n畸变参数 D：")
    print(dist_coeffs)

    print(f"\n整体 RMS 重投影误差：{rms:.6f} pixel")

    print(
        f"逐张图片平均 RMS："
        f"{np.mean(per_view_errors):.6f} pixel"
    )

    print(
        f"逐张图片最大 RMS："
        f"{np.max(per_view_errors):.6f} pixel"
    )

    print(f"\n标定结果已保存：{output_path}")

    print("\n逐张图片误差：")

    for i, error in enumerate(per_view_errors):
        print(f"View {i:02d}: {error:.6f} pixel")

    return camera_matrix, dist_coeffs, rms


# ============================================================
# 7. 命令行入口
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Nova 2 项目相机内参标定工具"
    )

    parser.add_argument(
        "--image-dir",
        default="data/camera_calibration",
        help="棋盘格图片目录"
    )

    parser.add_argument(
        "--output",
        default="config/camera_intrinsics.npz",
        help="标定结果输出路径"
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="棋盘格内角点列数"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="棋盘格内角点行数"
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="棋盘格方格边长，单位 mm"
    )

    parser.add_argument(
        "--min-views",
        type=int,
        default=10,
        help="最少有效标定图片数量"
    )

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="角点检测预览图输出目录"
    )

    args = parser.parse_args()

    calibrate_camera(
        image_dir=args.image_dir,
        output_path=args.output,
        pattern_size=(args.cols, args.rows),
        square_size=args.square_size,
        min_views=args.min_views,
        debug_dir=args.debug_dir
    )


if __name__ == "__main__":
    main()
