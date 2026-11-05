import importlib.util
import os
import tempfile

import cv2
import numpy as np

SRC = "calibration/camera_calibration.py"
spec = importlib.util.spec_from_file_location("camcal", SRC)
camcal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camcal)

# ============================================================
# 1. 虚拟真实相机（真值）
# ============================================================
W, H = 1280, 960
K_true = np.array([[1000.0, 0.0, W / 2],
                   [0.0, 1000.0, H / 2],
                   [0.0, 0.0, 1.0]])
D_true = np.zeros(5)              # 先用无畸变，判据最清晰

PATTERN = (9, 6)
SQUARE = 25.0

# ============================================================
# 2. 造一张“标准棋盘格模板”（10x7 方格 -> 9x6 内角点）
# ============================================================
cell, margin = 60, 120
bw = (PATTERN[0] + 1) * cell + 2 * margin
bh = (PATTERN[1] + 1) * cell + 2 * margin
board = np.full((bh, bw), 255, np.uint8)
for r in range(PATTERN[1] + 1):
    for c in range(PATTERN[0] + 1):
        if (r + c) % 2 == 0:
            board[margin + r * cell: margin + (r + 1) * cell,
                  margin + c * cell: margin + (c + 1) * cell] = 0

# 模板棋盘格外四角（像素）
src_pts = np.array([
    [margin, margin],
    [margin + (PATTERN[0] + 1) * cell, margin],
    [margin + (PATTERN[0] + 1) * cell, margin + (PATTERN[1] + 1) * cell],
    [margin, margin + (PATTERN[1] + 1) * cell],
], np.float32)

# 棋盘格外四角（世界坐标 mm），与 create_object_points 的原点约定一致
outer_obj = np.array([
    [-SQUARE,                          -SQUARE,                           0],
    [ PATTERN[0] * SQUARE,             -SQUARE,                           0],
    [ PATTERN[0] * SQUARE,              PATTERN[1] * SQUARE,              0],
    [-SQUARE,                           PATTERN[1] * SQUARE,              0],
], np.float32)

# ============================================================
# 3. 合成 12 张不同位姿的棋盘格图片
# ============================================================
rng = np.random.default_rng(0)
tmp_dir = tempfile.mkdtemp(prefix="calib_test_")
n, tries = 0, 0
while n < 12 and tries < 500:
    tries += 1
    rvec = np.deg2rad(rng.uniform(-20, 20, 3)).astype(np.float64)
    tvec = np.array([0, 0, 420.0]) + rng.uniform(-60, 60, 3) * np.array([1, 1, 0])

    img_pts, _ = cv2.projectPoints(outer_obj, rvec, tvec, K_true, D_true)
    img_pts = img_pts.reshape(4, 2).astype(np.float32)
    if np.any(img_pts < 40) or np.any(img_pts[:, 0] > W - 40) \
            or np.any(img_pts[:, 1] > H - 40):
        continue

    Hm = cv2.getPerspectiveTransform(src_pts, img_pts)
    img = cv2.warpPerspective(board, Hm, (W, H), borderValue=255)
    cv2.imwrite(os.path.join(tmp_dir, f"view_{n:02d}.png"), img)  # PNG 避免 JPG 压缩伪影
    n += 1

print(f"合成图片目录: {tmp_dir}  共 {n} 张")

# ============================================================
# 4. 调用被测试的函数
# ============================================================
out_npz = os.path.join(tmp_dir, "camera_intrinsics.npz")
K_est, D_est, rms = camcal.calibrate_camera(
    image_dir=tmp_dir,
    output_path=out_npz,
    pattern_size=PATTERN,
    square_size=SQUARE,
    min_views=10,
)

# ============================================================
# 5. 断言
# ============================================================
Ke = K_est.copy()
Kt = K_true.copy()

# 去掉分母为 0 的 nan
mask = np.abs(Kt) > 1e-9
rel_err = np.divide(np.abs(Ke - Kt), np.abs(Kt),
                    out=np.zeros_like(Kt), where=mask)

print("\n===== 测试结果 =====")
print("K_est:\n", Ke)
print("K 相对误差:\n", rel_err)
print("D_est:", D_est.ravel())
print("整体 RMS:", rms)

assert os.path.exists(out_npz), "npz 未生成"
assert rms < 1.0, f"RMS 过大: {rms}"
assert np.all(rel_err[mask] < 0.01), \
    f"K 相对误差超 1%:\n{rel_err}"
assert np.linalg.norm(Ke[:2, 2] - Kt[:2, 2]) < 2.0, "主点偏差 > 2px"
assert np.allclose(D_est.ravel(), 0, atol=0.05), "畸变系数应接近 0"
print("\n[PASS] 内参标定流程正确")