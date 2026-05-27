/*
 * K230 相机去畸变查找表接口
 *
 * 使用方式:
 *   1. 把 PC 端生成的 undistort_mapx.bin / undistort_mapy.bin 放到 K230 文件系统
 *   2. 运行时调用 undistort_map_load() 加载查找表
 *   3. 每帧调用 undistort_frame() 做去畸变
 *
 * 原理:
 *   mapx/mapy 是 PC 端通过 OpenCV initUndistortRectifyMap() 预计算的坐标映射表。
 *   K230 端只需查表 + 双线性插值，无需实时解算畸变模型。
 */

#ifndef UNDISTORT_K230_H
#define UNDISTORT_K230_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- 标定参数 ---------- */
#define UNDISTORT_IMAGE_W   1920
#define UNDISTORT_IMAGE_H   1080
#define UNDISTORT_MAP_SIZE  (UNDISTORT_IMAGE_W * UNDISTORT_IMAGE_H)

/* 去畸变后的新相机内参 (fx, fy, cx, cy) */
#define UNDISTORT_NEW_FX    1618.58291f
#define UNDISTORT_NEW_FY    1639.77368f
#define UNDISTORT_NEW_CX    898.91580f
#define UNDISTORT_NEW_CY    600.99650f

/* ---------- 查找表结构 ---------- */
typedef struct {
    float *mapx;        /* [H x W] 原图 x 坐标查找表 */
    float *mapy;        /* [H x W] 原图 y 坐标查找表 */
    int    width;       /* 图像宽度 */
    int    height;      /* 图像高度 */
    int    loaded;      /* 0=未加载, 1=已加载 */
} undistort_map_t;

/* ---------- API ---------- */

/**
 * @brief 加载去畸变查找表 (从二进制文件)
 * @param map       查找表结构体指针
 * @param mapx_path mapx.bin 文件路径
 * @param mapy_path mapy.bin 文件路径
 * @return 0 成功, -1 失败
 *
 * 说明: 只需在程序启动时调用一次。同一台相机 + 同一分辨率，查找表永不变。
 */
int undistort_map_load(undistort_map_t *map,
                       const char *mapx_path,
                       const char *mapy_path);

/**
 * @brief 释放查找表内存
 */
void undistort_map_free(undistort_map_t *map);

/**
 * @brief 对单帧图像做去畸变 (RGB/BGR 三通道)
 * @param map   已加载的查找表
 * @param src   输入图像 (畸变图), 大小 = W*H*3 bytes
 * @param dst   输出图像 (去畸变图), 大小 = W*H*3 bytes
 * @return 0 成功, -1 失败
 *
 * 说明: 逐像素查表 + 双线性插值。无浮点数学运算，适合嵌入式实时处理。
 */
int undistort_frame(const undistort_map_t *map,
                    const uint8_t *src,
                    uint8_t *dst);

/**
 * @brief 对单帧图像做去畸变 (单通道灰度)
 */
int undistort_frame_gray(const undistort_map_t *map,
                         const uint8_t *src,
                         uint8_t *dst);

#ifdef __cplusplus
}
#endif

#endif /* UNDISTORT_K230_H */
