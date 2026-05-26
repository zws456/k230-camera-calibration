#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 端相机标定脚本 - OpenCV
使用从 K230 导出的棋盘格照片，计算相机内参和畸变系数。

使用方法:
    python3 calibrate_camera.py --image_dir ./calib_images --square_size 20.0
    python3 calibrate_camera.py --image_dir ./calib_images --checkerboard 9 6

输出:
    camera_intrinsic.yaml  - 内参矩阵、畸变系数、图像尺寸等
    calib_result.jpg       - 可视化重投影误差
"""

import cv2
import numpy as np
import os
import glob
import argparse
import yaml

# ==================== 默认配置（可通过命令行覆盖） ====================
DEFAULT_CHECKERBOARD = (9, 6)     # 内角点: (横向-1, 纵向-1)
DEFAULT_SQUARE_SIZE = 25.0        # 方格边长 mm
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="K230 Camera Calibration with OpenCV")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="标定图片所在目录 (支持 .jpg .png .bmp)",
    )
    parser.add_argument(
        "--checkerboard",
        type=int,
        nargs=2,
        default=DEFAULT_CHECKERBOARD,
        metavar=("COLS", "ROWS"),
        help=f"棋盘格内角点数量，默认 {DEFAULT_CHECKERBOARD}",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=DEFAULT_SQUARE_SIZE,
        help=f"方格边长 (mm)，默认 {DEFAULT_SQUARE_SIZE}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_intrinsic.yaml",
        help="输出标定结果文件名",
    )
    return parser.parse_args()


def calibrate(image_dir, checkerboard, square_size, output_file):
    # 准备理论 3D 角点坐标
    objp = np.zeros((checkerboard[1] * checkerboard[0], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
        * square_size
    )

    objpoints = []   # 3D 点集
    imgpoints = []   # 2D 像素点集
    image_size = None

    # 搜索所有图片
    patterns = [
        os.path.join(image_dir, "*.jpg"),
        os.path.join(image_dir, "*.jpeg"),
        os.path.join(image_dir, "*.png"),
        os.path.join(image_dir, "*.bmp"),
    ]
    images = []
    for p in patterns:
        images.extend(glob.glob(p))
    images = sorted(images)

    if len(images) == 0:
        print(f"[错误] 目录 {image_dir} 中没有找到图片")
        return

    print(f"[信息] 找到 {len(images)} 张图片，开始处理...")
    print("-" * 60)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    valid_images = []

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"[{idx+1}/{len(images)}] 读取失败: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images.append(fname)

            # 可视化
            vis = img.copy()
            cv2.drawChessboardCorners(vis, checkerboard, corners2, ret)
            h, w = vis.shape[:2]
            scale = 800 / max(h, w)
            vis_small = cv2.resize(vis, None, fx=scale, fy=scale)
            cv2.imshow("Corners", vis_small)
            cv2.waitKey(100)
            print(f"[{idx+1}/{len(images)}] ✓ 检测到角点: {os.path.basename(fname)}")
        else:
            print(f"[{idx+1}/{len(images)}] ✗ 未检测到角点: {os.path.basename(fname)}")

    cv2.destroyAllWindows()
    print("-" * 60)

    if len(objpoints) < 5:
        print(f"[错误] 有效图片仅 {len(objpoints)} 张，至少需要 5 张才能标定")
        return

    print(f"[信息] 有效图片: {len(objpoints)}/{len(images)} 张")
    print("[信息] 开始计算相机参数...")

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    print("\n" + "=" * 60)
    print("标定结果")
    print("=" * 60)
    print(f"重投影误差 (RMSE): {ret:.4f} pixels")
    print(f"图像尺寸: {image_size[0]} x {image_size[1]}")
    print(f"\n相机内参矩阵 K ({mtx.shape[0]}x{mtx.shape[1]}):")
    print(np.array2string(mtx, precision=4, suppress_small=True))
    print(f"\n畸变系数 D (k1, k2, p1, p2, k3):")
    print(dist.flatten())

    # 计算每个图片的重投影误差详情
    print("\n各图片重投影误差:")
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"  {os.path.basename(valid_images[i]):30s}  error={error:.4f} px")
    print(f"  平均误差: {total_error / len(objpoints):.4f} px")

    # 保存结果
    calib_data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": mtx.flatten().tolist(),
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": dist.flatten().tolist(),
        },
        "reprojection_error": float(ret),
        "checkerboard": {
            "inner_corners": list(checkerboard),
            "square_size_mm": float(square_size),
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(calib_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[保存] 标定结果已保存到: {output_file}")

    # 生成去畸变映射并展示效果
    print("\n[信息] 生成去畸变示例...")
    sample_img = cv2.imread(valid_images[0])
    h, w = sample_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(sample_img, mapx, mapy, cv2.INTER_LINEAR)

    # 拼接对比图
    comparison = np.hstack((sample_img, dst))
    scale = 1600 / comparison.shape[1]
    comparison_small = cv2.resize(comparison, None, fx=scale, fy=scale)
    cv2.imwrite("calib_undistort_comparison.jpg", comparison)
    print("[保存] 去畸变对比图: calib_undistort_comparison.jpg")

    print("\n" + "=" * 60)
    print("下一步:")
    print("  1. 确认重投影误差 < 0.5 px 为良好，< 1.0 px 为可接受")
    print("  2. 将 camera_intrinsic.yaml 放到 PC 端项目目录")
    print("  3. 继续执行相机-雷达外参标定")
    print("=" * 60)

    return mtx, dist, image_size


if __name__ == "__main__":
    args = parse_args()
    calibrate(args.image_dir, tuple(args.checkerboard), args.square_size, args.output)
