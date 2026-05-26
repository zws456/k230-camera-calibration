#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K230 庐山派 - 棋盘格标定图片采集脚本
在 K230 大核 Linux 环境下运行，通过摄像头采集标定图片。

使用方法:
    python3 capture_calib_images.py

按 's' 保存当前帧（仅当检测到棋盘格角点时）
按 'q' 退出
"""

import cv2
import os
import sys

# ==================== 配置参数（请根据你的棋盘格修改） ====================
CHECKERBOARD = (9, 6)       # 内角点数量: (横向格点数-1, 纵向格点数-1)
SQUARE_SIZE_MM = 25.0       # 每个方格的边长，单位毫米 (mm)
CAPTURE_DELAY_FRAMES = 10   # 连续检测到角点后，等待多少帧才允许再次保存（防抖）
# ========================================================================

SAVE_DIR = "/root/calib_images"   # K230 上保存路径，可改为 /mnt/SD 或 U盘路径


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # K230 摄像头设备号通常是 0，如有多个可尝试 1, 2...
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头，请检查设备号")
        sys.exit(1)

    # 尝试设置分辨率 (K230 摄像头常见分辨率)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 准备角点搜索的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备理论角点坐标 (以方格尺寸为单位)
    objp = []
    for i in range(CHECKERBOARD[1]):
        for j in range(CHECKERBOARD[0]):
            objp.append([j * SQUARE_SIZE_MM, i * SQUARE_SIZE_MM, 0.0])
    objp = np.array(objp, dtype=np.float32)

    saved_count = 0
    cooldown = 0

    print("=" * 60)
    print("K230 棋盘格标定图片采集")
    print("=" * 60)
    print(f"棋盘格规格: {CHECKERBOARD[0]} x {CHECKERBOARD[1]} 内角点")
    print(f"方格尺寸: {SQUARE_SIZE_MM} mm")
    print(f"保存路径: {SAVE_DIR}")
    print("操作说明:")
    print("  [s] 保存当前帧（仅检测到角点时有效）")
    print("  [q] 退出程序")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 帧读取失败")
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 尝试检测棋盘格角点
        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            # 亚像素精确化
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, found)

            status_text = "Corners Found - Press 's' to save"
            color = (0, 255, 0)

            if cooldown > 0:
                cooldown -= 1
                status_text = f"Cooldown... ({cooldown})"
                color = (0, 165, 255)
        else:
            status_text = "No Corners"
            color = (0, 0, 255)
            cooldown = 0

        # 绘制信息
        cv2.putText(
            display,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            display,
            f"Saved: {saved_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("K230 Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if found and cooldown == 0:
                filepath = os.path.join(SAVE_DIR, f"calib_{saved_count:03d}.jpg")
                cv2.imwrite(filepath, frame)
                print(f"[保存] {filepath}")
                saved_count += 1
                cooldown = CAPTURE_DELAY_FRAMES
            elif not found:
                print("[提示] 当前帧未检测到棋盘格角点，无法保存")
            else:
                print("[提示] 请等待防抖冷却结束")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[完成] 共保存 {saved_count} 张标定图片到 {SAVE_DIR}")
    print("[下一步] 请将照片导出到 PC，运行 calibrate_camera.py")


if __name__ == "__main__":
    main()
