#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机-雷达外参标定脚本 (简化版)

本脚本提供两种方案:
1. 手动标定: 直接输入相机相对雷达的位移和旋转，生成 4x4 变换矩阵
2. 基于棋盘格的联合标定: (TODO，需要雷达能精确看到棋盘格)

使用方法 (手动标定):
    python3 camera_lidar_extrinsic.py --mode manual \
        --dx 0.05 --dy 0.0 --dz 0.02 \
        --roll 0 --pitch -10 --yaw 0

输出:
    camera_lidar_extrinsic.yaml

坐标系说明:
    - 输入的 dx,dy,dz 是相机光心相对于雷达中心的位移
    - 旋转角度单位: 度 (degree)
    - 旋转顺序: Z-Y-X (Yaw-Pitch-Roll)，即先绕Z轴，再绕Y轴，最后绕X轴
"""

import numpy as np
import argparse
import yaml
import os


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    Z-Y-X 欧拉角 (Yaw-Pitch-Roll) 转旋转矩阵
    roll: 绕 X 轴 (deg)
    pitch: 绕 Y 轴 (deg)
    yaw: 绕 Z 轴 (deg)
    """
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
    )
    Ry = np.array(
        [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
    )
    Rz = np.array(
        [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
    )

    # Z-Y-X 顺序: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def manual_calibration(args):
    """手动标定：直接构造变换矩阵"""
    print("=" * 60)
    print("相机-雷达外参标定 (手动模式)")
    print("=" * 60)

    R = euler_to_rotation_matrix(args.roll, args.pitch, args.yaw)
    t = np.array([[args.dx], [args.dy], [args.dz]])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    print(f"\n位移 (相机相对雷达, 单位 m):")
    print(f"  dx={args.dx:.4f}, dy={args.dy:.4f}, dz={args.dz:.4f}")
    print(f"\n旋转 (Z-Y-X 欧拉角, 单位 deg):")
    print(f"  roll={args.roll:.2f}, pitch={args.pitch:.2f}, yaw={args.yaw:.2f}")
    print(f"\n变换矩阵 T_lidar^camera (4x4):")
    print(np.array2string(T, precision=6, suppress_small=True))

    # 保存
    result = {
        "calibration_mode": "manual",
        "translation": {
            "dx_m": float(args.dx),
            "dy_m": float(args.dy),
            "dz_m": float(args.dz),
        },
        "rotation": {
            "roll_deg": float(args.roll),
            "pitch_deg": float(args.pitch),
            "yaw_deg": float(args.yaw),
            "order": "ZYX (Yaw-Pitch-Roll)",
        },
        "transform_matrix": {
            "rows": 4,
            "cols": 4,
            "data": T.flatten().tolist(),
        },
        "description": "T_lidar^camera: 将雷达坐标系下的点变换到相机坐标系",
    }

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[保存] 外参已保存到: {args.output}")
    print("\n提示:")
    print("  - 此矩阵为 T_lidar^camera，即 雷达点 → 相机坐标系")
    print("  - 若要将相机射线转换到雷达坐标系，请使用逆矩阵: T_camera^lidar = inv(T)")
    print("  - 运行 hand_tracker_node.py 时会自动读取此文件")

    return T


def main():
    parser = argparse.ArgumentParser(
        description="Camera-LiDAR Extrinsic Calibration"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="manual",
        choices=["manual"],
        help="标定模式，目前仅支持 manual",
    )
    parser.add_argument(
        "--dx", type=float, default=0.0, help="相机相对雷达 X 方向位移 (m)"
    )
    parser.add_argument(
        "--dy", type=float, default=0.0, help="相机相对雷达 Y 方向位移 (m)"
    )
    parser.add_argument(
        "--dz", type=float, default=0.0, help="相机相对雷达 Z 方向位移 (m)"
    )
    parser.add_argument(
        "--roll", type=float, default=0.0, help="绕 X 轴旋转 (deg)"
    )
    parser.add_argument(
        "--pitch", type=float, default=0.0, help="绕 Y 轴旋转 (deg)"
    )
    parser.add_argument(
        "--yaw", type=float, default=0.0, help="绕 Z 轴旋转 (deg)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_lidar_extrinsic.yaml",
        help="输出文件路径",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        manual_calibration(args)


if __name__ == "__main__":
    main()
