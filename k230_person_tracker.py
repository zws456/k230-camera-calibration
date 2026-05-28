"""
K230 人体跟踪 + 底盘控制程序

功能：
    1. 实时检测画面中所有人体
    2. 选择距离画面中心最近的人体作为跟踪目标
    3. 计算目标到相机的距离（单目测距）
    4. 计算底盘控制量（旋转+前进），使人物保持在画面中心且距离合适
    5. 通过 UART 发送控制指令给底盘（参考 cmd_vel_to_uart.py 协议）

硬件连接：
    - K230 UART2 TX (排针 Pin11, GPIO5) → 底盘 UART RX
    - K230 UART2 RX (排针 Pin13, GPIO6) → 底盘 UART TX
    - 共地

运行方式：
    用 CanMV IDE 打开本文件，连接 K230，点击运行。
"""

import os, sys, time, math, struct, gc
from machine import UART, FPIOA, Pin
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aicube

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *

# ============================================================
# 标定参数（来自 camera_intrinsic.yaml，1920x1080 标定）
# ============================================================
CALIB_FX = 1623.7143
CALIB_FY = 1622.9086
CALIB_CX = 915.0068
CALIB_CY = 597.0911
CALIB_WIDTH = 1920
CALIB_HEIGHT = 1080

# ============================================================
# 运行分辨率（可根据需要修改）
# ============================================================
RGB888P_SIZE = [1280, 720]   # sensor 输出给 AI 的分辨率
DISPLAY_MODE = "hdmi"         # hdmi/lcd/virt

# ============================================================
# 测距参数
# ============================================================
PERSON_REAL_HEIGHT = 1.70     # 人体平均身高（米）

# ============================================================
# 底盘控制参数
# ============================================================
TARGET_DISTANCE = 2.0         # 目标跟踪距离（米）
KP_YAW = 2.0                  # 旋转控制比例系数
KP_DIST = 0.8                 # 距离控制比例系数
MAX_OMEGA = 1.0               # 最大旋转角速度（rad/s）
MAX_VX = 0.5                  # 最大前进速度（m/s）
VY = 0.0                      # 左右速度（全向轮，暂不使用）
DEADZONE_X = 20               # 水平偏差死区（像素）
DEADZONE_DIST = 0.2           # 距离偏差死区（米）

# ============================================================
# UART 配置（连接底盘）
# ============================================================
UART_PORT = UART.UART2
UART_BAUD = 115200
UART_TX_PIN = 11              # 排针 Pin11 = GPIO5 = UART2_TXD
UART_RX_PIN = 13              # 排针 Pin13 = GPIO6 = UART2_RXD

# ============================================================
# 人体检测模型配置
# ============================================================
KMODEL_PATH = "/sdcard/examples/kmodel/person_detect_yolov5n.kmodel"
MODEL_INPUT_SIZE = [640, 640]
LABELS = ["person"]
ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.6
STRIDES = [8, 16, 32]

# ============================================================
# 工具函数
# ============================================================

def pack_uart_frame(vx, vy, omega):
    """
    打包底盘控制帧（16字节定长，小端，参考 cmd_vel_to_uart.py）
    [0xAA][0x55][vx:float32][vy:float32][omega:float32][0x0D][0x0A]
    """
    data = bytearray(16)
    data[0] = 0xAA
    data[1] = 0x55
    struct.pack_into('<f', data, 2, float(vx))
    struct.pack_into('<f', data, 6, float(vy))
    struct.pack_into('<f', data, 10, float(omega))
    data[14] = 0x0D
    data[15] = 0x0A
    return bytes(data)


def init_uart():
    """初始化 UART2 用于连接底盘"""
    fpioa = FPIOA()
    fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
    fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)
    uart = UART(UART_PORT, baudrate=UART_BAUD, bits=UART.EIGHTBITS,
                parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
    return uart


def clamp(val, min_val, max_val):
    """限幅"""
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


# ============================================================
# 人体跟踪检测类
# ============================================================

class PersonTrackerApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, labels, anchors,
                 confidence_threshold=0.2, nms_threshold=0.6,
                 nms_option=False, strides=[8, 16, 32],
                 rgb888p_size=[1280, 720], display_size=[1920, 1080],
                 debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.labels = labels
        self.anchors = anchors
        self.strides = strides
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.nms_option = nms_option
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                 nn.ai2d_format.NCHW_FMT,
                                 np.uint8, np.uint8)

        # 根据运行分辨率缩放标定参数
        scale_x = self.rgb888p_size[0] / CALIB_WIDTH
        scale_y = self.rgb888p_size[1] / CALIB_HEIGHT
        self.fx = CALIB_FX * scale_x
        self.fy = CALIB_FY * scale_y
        self.cx = CALIB_CX * scale_x
        self.cy = CALIB_CY * scale_y

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right, _ = center_pad_param(self.rgb888p_size, self.model_input_size)
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            dets = aicube.anchorbasedet_post_process(
                results[0], results[1], results[2],
                self.model_input_size, self.rgb888p_size,
                self.strides, len(self.labels),
                self.confidence_threshold, self.nms_threshold,
                self.anchors, self.nms_option)
            return dets

    def find_nearest_person(self, dets):
        """
        从所有检测框中找到距离画面中心最近的人体。
        返回: (det_box, distance, center_x, center_y, pixel_height)
               如果没有检测到人体，返回 None
        """
        if not dets:
            return None

        best_det = None
        best_dist = float('inf')

        for det in dets:
            label_idx, confidence, x1, y1, x2, y2 = det
            # 只关注 person 类别（label_idx == 0）
            if int(label_idx) != 0:
                continue

            # 计算检测框中心
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 计算到画面中心的欧氏距离（像素）
            dist = math.sqrt((cx - self.cx) ** 2 + (cy - self.cy) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_det = det
                best_cx = cx
                best_cy = cy
                best_h = y2 - y1

        if best_det is None:
            return None

        # 单目测距：distance = (real_height * focal_length) / pixel_height
        pixel_height = best_h
        if pixel_height > 0:
            distance = (PERSON_REAL_HEIGHT * self.fy) / pixel_height
        else:
            distance = 999.0

        return {
            'det': best_det,
            'distance': distance,
            'cx': best_cx,
            'cy': best_cy,
            'pixel_height': pixel_height,
            'center_dist': best_dist
        }

    def compute_control(self, person_info):
        """
        根据目标位置计算底盘控制量。
        返回: (vx, vy, omega)
        """
        if person_info is None:
            # 丢失目标，停止
            return 0.0, 0.0, 0.0

        cx = person_info['cx']
        cy = person_info['cy']
        distance = person_info['distance']

        # ---- 1. 旋转控制（让人物在画面水平中间）----
        error_x = cx - self.cx
        if abs(error_x) < DEADZONE_X:
            omega = 0.0
        else:
            # 归一化误差到 [-1, 1]，然后乘比例和最大速度
            normalized_x = error_x / (self.rgb888p_size[0] / 2.0)
            omega = -KP_YAW * normalized_x * MAX_OMEGA
            omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)

        # ---- 2. 前进/后退控制（保持目标距离）----
        error_dist = distance - TARGET_DISTANCE
        if abs(error_dist) < DEADZONE_DIST:
            vx = 0.0
        else:
            vx = KP_DIST * error_dist
            vx = clamp(vx, -MAX_VX, MAX_VX)

        # ---- 3. 左右平移（暂不使用）----
        vy = VY

        return vx, vy, omega

    def draw_result(self, pl, person_info, dets):
        """绘制检测结果和跟踪信息到 OSD"""
        with ScopedTiming("display_draw", self.debug_mode > 0):
            pl.osd_img.clear()

            if dets:
                for det in dets:
                    x1, y1, x2, y2 = det[2], det[3], det[4], det[5]
                    sx = int(x1 * self.display_size[0] // self.rgb888p_size[0])
                    sy = int(y1 * self.display_size[1] // self.rgb888p_size[1])
                    w = int((x2 - x1) * self.display_size[0] // self.rgb888p_size[0])
                    h = int((y2 - y1) * self.display_size[1] // self.rgb888p_size[1])

                    # 被跟踪的目标用红色框，其他用绿色框
                    if person_info and det is person_info['det']:
                        color = (255, 255, 0, 0)  # 红色 (A,R,G,B)
                        label = "TARGET %.2fm" % person_info['distance']
                    else:
                        color = (255, 0, 255, 0)  # 绿色
                        label = self.labels[int(det[0])] + " %.2f" % det[1]

                    pl.osd_img.draw_rectangle(sx, sy, w, h, color=color, thickness=2)
                    pl.osd_img.draw_string_advanced(sx, sy - 40, 28, label, color=color)

            # 绘制画面中心十字
            center_x = int(self.cx * self.display_size[0] // self.rgb888p_size[0])
            center_y = int(self.cy * self.display_size[1] // self.rgb888p_size[1])
            pl.osd_img.draw_cross(center_x, center_y, color=(255, 255, 255, 0), size=15, thickness=2)

            # 绘制控制信息
            if person_info:
                info_str = "Dist:%.2fm Omega:%.2f Vx:%.2f" % (
                    person_info['distance'],
                    person_info.get('omega', 0),
                    person_info.get('vx', 0)
                )
                pl.osd_img.draw_string_advanced(10, 10, 28, info_str, color=(255, 255, 255, 0))
            else:
                pl.osd_img.draw_string_advanced(10, 10, 28, "No person detected", color=(255, 255, 0, 0))


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 50)
    print("K230 Person Tracker + Chassis Control")
    print("=" * 50)

    # ---- 初始化显示和摄像头 ----
    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
    pl.create()
    display_size = pl.get_display_size()
    print("Display size:", display_size)

    # ---- 初始化人体检测 ----
    tracker = PersonTrackerApp(
        kmodel_path=KMODEL_PATH,
        model_input_size=MODEL_INPUT_SIZE,
        labels=LABELS,
        anchors=ANCHORS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        nms_option=False,
        strides=STRIDES,
        rgb888p_size=RGB888P_SIZE,
        display_size=display_size,
        debug_mode=0
    )
    tracker.config_preprocess()
    print("Person detection model loaded.")

    # ---- 初始化 UART ----
    try:
        uart = init_uart()
        print("UART2 initialized @ %d baud" % UART_BAUD)
    except Exception as e:
        print("UART init failed:", e)
        uart = None

    # ---- 板载 RGB 灯（状态指示）----
    fpioa_led = FPIOA()
    fpioa_led.set_function(62, FPIOA.GPIO62)
    led_r = Pin(62, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
    led_r.high()  # 初始灭（共阳极，低电平亮）

    # ---- 主循环 ----
    frame_count = 0
    last_send_time = 0
    send_interval_ms = 50  # 20Hz 发送频率，和 cmd_vel_to_uart.py 一致

    print("Main loop started...")

    while True:
        with ScopedTiming("total", 1):
            # 1. 获取图像
            img = pl.get_frame()

            # 2. 人体检测
            dets = tracker.run(img)

            # 3. 找到最近的人体
            person_info = tracker.find_nearest_person(dets)

            # 4. 计算底盘控制量
            vx, vy, omega = tracker.compute_control(person_info)

            # 更新 person_info 用于显示
            if person_info:
                person_info['vx'] = vx
                person_info['vy'] = vy
                person_info['omega'] = omega

            # 5. UART 发送（固定频率）
            now = time.ticks_ms()
            if uart and time.ticks_diff(now, last_send_time) >= send_interval_ms:
                frame = pack_uart_frame(vx, vy, omega)
                uart.write(frame)
                last_send_time = now

                # LED 闪烁指示（跟踪到目标时绿灯亮）
                if person_info:
                    led_r.low()
                else:
                    led_r.high()

            # 6. 绘制结果
            tracker.draw_result(pl, person_info, dets)
            pl.show_image()

            # 7. 垃圾回收
            gc.collect()
            frame_count += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        import traceback
        traceback.print_exc()
    finally:
        print("Program terminated.")
