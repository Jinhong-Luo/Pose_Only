import numpy as np
import cv2
import argparse
import pathlib
import tqdm
from scipy.spatial.transform import Rotation as R

'''
w = [0.2, -0.1, 0.05]      # axis-angle (rad)
R_mat = R.from_rotvec(w).as_matrix()
w = R.from_matrix(R_mat).as_rotvec()
旋转矩阵与旋转向量相互转换


R_gt = R.from_matrix(R_gt_mat)
R_pred = R.from_matrix(R_pred_mat)

R_rel = R_gt.inv() * R_pred
e = R_rel.as_rotvec()
rot_err_rad = np.linalg.norm(e)
位姿误差（你毕设最常用）
'''

K=[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
   0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
   0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]

def pixel_to_normalized_homogeneous(
    uv: np.ndarray,
    K: np.ndarray,
    distCoeffs: np.ndarray | None = None
) -> np.ndarray:
    """
    将像素坐标 (u,v) 转换为归一化齐次坐标 (x,y,1)

    参数
    ----
    uv : (N,2) 或 (2,) ndarray
        像素坐标
    K : (3,3) ndarray
        相机内参矩阵
    distCoeffs : ndarray or None
        畸变参数；KITTI 可设为 None 或 zeros

    返回
    ----
    x_h : (N,3) ndarray
        归一化齐次坐标 (x,y,1)
    """
    uv = np.asarray(uv, dtype=np.float64)

    # 允许输入 (2,) 或 (N,2)
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)

    # OpenCV 要求 (N,1,2)
    uv = uv.reshape(-1, 1, 2)

    if distCoeffs is None:
        # KITTI: 已去畸变，等价于无畸变
        distCoeffs = np.zeros((5, 1), dtype=np.float64)

    # 去畸变 + 乘 K^{-1}
    x_norm = cv2.undistortPoints(
        uv,
        cameraMatrix=K,
        distCoeffs=distCoeffs
    )  # (N,1,2)

    x_norm = x_norm.reshape(-1, 2)  # (N,2)

    # 齐次化 -> (x,y,1)
    ones = np.ones((x_norm.shape[0], 1), dtype=np.float64)
    x_h = np.hstack([x_norm, ones])  # (N,3)

    return x_h

'''
# 输入：像素级 tracks
tracks_px = [
    {0: (u0,v0), 3: (u3,v3), 5: (u5,v5)},
    ...
]

# 转换为：归一化齐次坐标 tracks
tracks_norm = []

for track in tracks_px:
    tr = {}
    for i, (u,v) in track.items():
        x_h = pixel_to_normalized_homogeneous((u,v), K)[0]
        tr[i] = x_h   # (x,y,1)
    if len(tr) >= 3:
        tracks_norm.append(tr)
'''


