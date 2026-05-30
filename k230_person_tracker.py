"""
K230 人体跟踪 + 测距（关键点版）

功能：
    1. 用 YOLOv8-Pose 检测人体关键点（17个关节点）
    2. 选择距离画面中心最近的人体
    3. 用关键点推算全身像素高度，解决"只有上半身时测距不准"的问题
       - 能看到髋部 → 全身 = (头到髋) × 2
       - 能看到肩膀 → 全身 = (头到肩) × 3.3
       - 只有头部 → 全身 = 头高 × 8
    4. 计算旋转控制量（omega），让人物保持在画面中心
    5. 通过 UART 发送控制指令

模型：/sdcard/examples/kmodel/yolov8n-pose.kmodel
"""

import os, sys, time, math, struct, gc
from machine import UART, FPIOA, Pin
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.Utils import *

# 导入标定参数
try:
    from camera_config import CALIB_WIDTH, CALIB_HEIGHT, CALIB_FX, CALIB_FY, CALIB_CX, CALIB_CY
except ImportError:
    CALIB_FX = 1623.7143
    CALIB_FY = 1622.9086
    CALIB_CX = 915.0068
    CALIB_CY = 597.0911
    CALIB_WIDTH = 1920
    CALIB_HEIGHT = 1080

# ============================================================
# 配置
# ============================================================
RGB888P_SIZE = [1280, 720]
DISPLAY_MODE = "virt"

PERSON_REAL_HEIGHT = 1.70     # 米

KP_YAW = 2.0
MAX_OMEGA = 1.0
DEADZONE_X = 20

UART_PORT = UART.UART2
UART_BAUD = 115200
UART_TX_PIN = 11
UART_RX_PIN = 13

# 关键点模型
KMODEL_PATH = "/sdcard/examples/kmodel/yolov8n-pose.kmodel"
MODEL_INPUT_SIZE = [320, 320]
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5

# 关键点索引（COCO，0-based）
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

HEAD_KPS = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
SHOULDER_KPS = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER]
HIP_KPS = [KP_LEFT_HIP, KP_RIGHT_HIP]

# ============================================================
# 工具函数
# ============================================================

def pack_uart_frame(vx, vy, omega):
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
    fpioa = FPIOA()
    fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
    fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)
    return UART(UART_PORT, baudrate=UART_BAUD, bits=UART.EIGHTBITS,
                parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)


def clamp(val, min_val, max_val):
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


# ============================================================
# 人体跟踪关键点类
# ============================================================

class PersonTrackerKeypointApp(AIBase):
    def __init__(self, kmodel_path, model_input_size,
                 confidence_threshold=0.2, nms_threshold=0.5,
                 rgb888p_size=[1280, 720], display_size=[1920, 1080],
                 debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode

        # 缩放标定参数
        scale_x = self.rgb888p_size[0] / CALIB_WIDTH
        scale_y = self.rgb888p_size[1] / CALIB_HEIGHT
        self.fx = CALIB_FX * scale_x
        self.fy = CALIB_FY * scale_y
        self.cx = CALIB_CX * scale_x
        self.cy = CALIB_CY * scale_y

        # 骨骼绘制配置（SKELETON 是 1-based 索引）
        self.SKELETON = [(16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
                         (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
                         (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
                         (2, 4), (3, 5), (4, 6), (5, 7)]
        self.KPS_COLORS = [(255, 0, 255, 0)] * 17
        self.LIMB_COLORS = [(255, 51, 153, 255)] * 5 + [(255, 255, 51, 255)] * 3 + \
                           [(255, 255, 128, 0)] * 5 + [(255, 0, 255, 0)] * 6

        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                 nn.ai2d_format.NCHW_FMT,
                                 np.uint8, np.uint8)

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
            return aidemo.person_kp_postprocess(
                results[0],
                [self.rgb888p_size[1], self.rgb888p_size[0]],
                self.model_input_size,
                self.confidence_threshold,
                self.nms_threshold
            )

    def estimate_full_height(self, kps):
        """
        根据关键点推算全身像素高度。
        返回 (estimated_height, method_string)

        注意：COCO 关键点没有"头顶"点，head_y 取的是鼻子/眼/耳的最上方，
        不是真正头顶。因此所有基于 head_y 的比例系数都要修正：
        - 鼻尖到肩 ≈ 全身 × 0.18  → 系数 1/0.18 ≈ 5.5
        - 鼻尖到髋 ≈ 全身 × 0.46  → 系数 1/0.46 ≈ 2.2
        - 头高(鼻尖到下巴) ≈ 全身 × 0.09 → 系数 1/0.09 ≈ 11
        - 肩宽 ≈ 全身 / 4.2
        """
        # 找头部最上点（鼻子/眼/耳的最上方，不是头顶）
        head_y = None
        for idx in HEAD_KPS:
            if kps[idx][2] > 0.15:
                y = kps[idx][1]
                if head_y is None or y < head_y:
                    head_y = y

        if head_y is None:
            return None, "no_head"

        # 能看到髋部？
        hip_y = None
        for idx in HIP_KPS:
            if kps[idx][2] > 0.15:
                y = kps[idx][1]
                if hip_y is None or y > hip_y:
                    hip_y = y

        if hip_y is not None:
            # 鼻尖到髋 ≈ 全身 × 0.46
            return (hip_y - head_y) * 2.2, "hip"

        # 能看到双肩？优先用肩宽（不受头部位置影响）
        left_shoulder = kps[KP_LEFT_SHOULDER]
        right_shoulder = kps[KP_RIGHT_SHOULDER]
        if left_shoulder[2] > 0.15 and right_shoulder[2] > 0.15:
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            # 肩宽 ≈ 全身 / 4.2
            return shoulder_width * 4.2, "shoulder_w"

        # 能看到单侧肩膀？
        shoulder_y = None
        for idx in SHOULDER_KPS:
            if kps[idx][2] > 0.15:
                y = kps[idx][1]
                if shoulder_y is None or y > shoulder_y:
                    shoulder_y = y

        if shoulder_y is not None:
            # 鼻尖到肩 ≈ 全身 × 0.18
            return (shoulder_y - head_y) * 5.5, "shoulder"

        # 只有头部，用头高(鼻尖到下巴) × 11
        head_bottom_y = None
        for idx in HEAD_KPS:
            if kps[idx][2] > 0.15:
                y = kps[idx][1]
                if head_bottom_y is None or y > head_bottom_y:
                    head_bottom_y = y

        if head_bottom_y is not None:
            return (head_bottom_y - head_y) * 11.0, "head_only"

        return None, "fail"

    def find_nearest_person(self, res):
        """
        从 pose 结果中找到距离画面中心最近的人体。
        res: person_kp_postprocess 返回的结果
        注意：res[0] 仅用于获取人数，不读取其具体格式（官方 API 中 det 格式不确定）
        """
        if not res or not res[0]:
            return None, -1

        kpses = res[1]
        num = len(res[0])  # 人数

        # 从关键点计算每个人的边界框和中心，不依赖 res[0] 的检测框格式
        persons = []
        for i in range(num):
            kps = kpses[i]

            # 收集所有可见关键点
            visible = [kp for kp in kps if kp[2] > 0.15]
            if not visible:
                continue

            xs = [kp[0] for kp in visible]
            ys = [kp[1] for kp in visible]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            # 优先用 nose 作为人物中心
            if kps[KP_NOSE][2] > 0.15:
                person_cx = kps[KP_NOSE][0]
            else:
                person_cx = (x1 + x2) / 2.0

            dist = abs(person_cx - self.cx)
            persons.append({
                'idx': i,
                'kps': kps,
                'bbox': [x1, y1, x2, y2],
                'person_cx': person_cx,
                'dist': dist
            })

        if not persons:
            return None, -1

        # 找离画面中心最近的
        best = min(persons, key=lambda p: p['dist'])
        best_idx = best['idx']
        kps = best['kps']
        x1, y1, x2, y2 = best['bbox']

        # 用关键点估算全身高度
        estimated_height, est_method = self.estimate_full_height(kps)

        # fallback 到检测框高度
        if estimated_height is None or estimated_height <= 0:
            estimated_height = y2 - y1
            est_method = "bbox_fallback"

        # 测距
        distance = (PERSON_REAL_HEIGHT * self.fy) / estimated_height

        return {
            'idx': best_idx,
            'kps': kps,
            'bbox': best['bbox'],
            'distance': distance,
            'cx': best['person_cx'],
            'estimated_height': estimated_height,
            'bbox_height': y2 - y1,
            'est_method': est_method
        }, best_idx

    def compute_control(self, person_info):
        """只计算旋转控制量"""
        if person_info is None:
            return 0.0, 0.0

        error_x = person_info['cx'] - self.cx
        if abs(error_x) < DEADZONE_X:
            omega = 0.0
        else:
            normalized_x = error_x / (self.rgb888p_size[0] / 2.0)
            omega = -KP_YAW * normalized_x * MAX_OMEGA
            omega = clamp(omega, -MAX_OMEGA, MAX_OMEGA)

        return omega, person_info['distance']

    def draw_result(self, pl, person_info, res, target_idx=-1):
        """绘制关键点、骨骼和跟踪信息"""
        with ScopedTiming("display_draw", self.debug_mode > 0):
            pl.osd_img.clear()

            if res and res[0]:
                dets = res[0]
                kpses = res[1]

                for i in range(len(dets)):
                    det = dets[i]
                    kps = kpses[i]
                    is_target = (i == target_idx)

                    # 绘制关键点
                    for k in range(17):
                        if kps[k][2] > 0.15:
                            kx = int(kps[k][0] * self.display_size[0] // self.rgb888p_size[0])
                            ky = int(kps[k][1] * self.display_size[1] // self.rgb888p_size[1])
                            color = (255, 255, 0, 0) if is_target else (255, 0, 255, 0)
                            pl.osd_img.draw_circle(kx, ky, 4, color, 2)

                    # 绘制骨骼
                    for k in range(len(self.SKELETON)):
                        ske = self.SKELETON[k]
                        idx1, idx2 = ske[0] - 1, ske[1] - 1
                        if kps[idx1][2] > 0.15 and kps[idx2][2] > 0.15:
                            x1_ = int(kps[idx1][0] * self.display_size[0] // self.rgb888p_size[0])
                            y1_ = int(kps[idx1][1] * self.display_size[1] // self.rgb888p_size[1])
                            x2_ = int(kps[idx2][0] * self.display_size[0] // self.rgb888p_size[0])
                            y2_ = int(kps[idx2][1] * self.display_size[1] // self.rgb888p_size[1])
                            color = (255, 255, 0, 0) if is_target else (255, 0, 255, 0)
                            pl.osd_img.draw_line(x1_, y1_, x2_, y2_, color, 2)

                    # 目标框和标签
                    if is_target and person_info:
                        x1, y1, x2, y2 = person_info['bbox']
                        sx = int(x1 * self.display_size[0] // self.rgb888p_size[0])
                        sy = int(y1 * self.display_size[1] // self.rgb888p_size[1])
                        w = int((x2 - x1) * self.display_size[0] // self.rgb888p_size[0])
                        h = int((y2 - y1) * self.display_size[1] // self.rgb888p_size[1])
                        pl.osd_img.draw_rectangle(sx, sy, w, h, color=(255, 255, 0, 0), thickness=2)

                        label = "Dist:%.2fm" % person_info['distance']
                        pl.osd_img.draw_string_advanced(sx, sy - 40, 28, label, color=(255, 255, 0, 0))

                            # 显示推算信息（已移到左上角，避免目标框在顶部时出界）
                        pass

            # 画面中心十字
            center_x = int(self.cx * self.display_size[0] // self.rgb888p_size[0])
            center_y = int(self.cy * self.display_size[1] // self.rgb888p_size[1])
            pl.osd_img.draw_cross(center_x, center_y, color=(255, 255, 255, 0), size=15, thickness=2)

            # 控制信息（左上角固定位置，不会出界）
            if person_info:
                info_str = "Dist:%.2fm Omega:%.2f" % (
                    person_info['distance'],
                    person_info.get('omega', 0)
                )
                pl.osd_img.draw_string_advanced(10, 10, 28, info_str, color=(255, 255, 255, 0))

                # 调试信息：估算高度 + 方法
                est_h = person_info['estimated_height']
                bbox_h = person_info['bbox_height']
                method = person_info.get('est_method', '?')
                debug_str = "%s estH:%d bboxH:%d" % (method, int(est_h), int(bbox_h))
                pl.osd_img.draw_string_advanced(10, 45, 24, debug_str, color=(255, 255, 255, 0))
            else:
                pl.osd_img.draw_string_advanced(10, 10, 28, "No person detected", color=(255, 255, 0, 0))


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 50)
    print("K230 Person Tracker (Keypoint Version)")
    print("=" * 50)

    pl = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
    pl.create()
    display_size = pl.get_display_size()
    print("Display size:", display_size)

    tracker = PersonTrackerKeypointApp(
        kmodel_path=KMODEL_PATH,
        model_input_size=MODEL_INPUT_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        rgb888p_size=RGB888P_SIZE,
        display_size=display_size,
        debug_mode=0
    )
    tracker.config_preprocess()
    print("Pose model loaded.")

    try:
        uart = init_uart()
        print("UART2 initialized @ %d baud" % UART_BAUD)
    except Exception as e:
        print("UART init failed:", e)
        uart = None

    fpioa_led = FPIOA()
    fpioa_led.set_function(62, FPIOA.GPIO62)
    led_r = Pin(62, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
    led_r.high()

    last_send_time = 0
    send_interval_ms = 50

    print("Main loop started...")

    while True:
        with ScopedTiming("total", 1):
            img = pl.get_frame()
            res = tracker.run(img)
            person_info, target_idx = tracker.find_nearest_person(res)
            omega, distance = tracker.compute_control(person_info)

            if person_info:
                person_info['omega'] = omega

            now = time.ticks_ms()
            if uart and time.ticks_diff(now, last_send_time) >= send_interval_ms:
                frame = pack_uart_frame(0.0, 0.0, omega)
                uart.write(frame)
                last_send_time = now

                if person_info:
                    led_r.low()
                else:
                    led_r.high()

            tracker.draw_result(pl, person_info, res, target_idx)
            pl.show_image()
            gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        import sys
        sys.print_exception(e)
    finally:
        print("Program terminated.")
