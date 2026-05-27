/*
 * K230 相机去畸变实现 (纯 C, 无 OpenCV 依赖)
 *
 * 编译示例:
 *   riscv64-linux-musl-gcc -O2 -o undistort_demo undistort_k230.c
 *
 * 运行示例:
 *   ./undistort_demo raw_frame.yuv undistorted_frame.yuv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "undistort_k230.h"

/* ---------- 内部函数: 双线性插值 ---------- */

static inline uint8_t clip_u8(int x)
{
    if (x < 0) return 0;
    if (x > 255) return 255;
    return (uint8_t)x;
}

/**
 * @brief 对单通道做双线性插值
 * @param src   源图像, W*H bytes
 * @param u     浮点 x 坐标
 * @param v     浮点 y 坐标
 * @param w     图像宽
 * @param h     图像高
 */
static uint8_t bilinear_gray(const uint8_t *src, float u, float v, int w, int h)
{
    int u0 = (int)floorf(u);
    int v0 = (int)floorf(v);
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    /* 边界裁剪 */
    if (u0 < 0) u0 = 0;
    if (v0 < 0) v0 = 0;
    if (u1 >= w) u1 = w - 1;
    if (v1 >= h) v1 = h - 1;

    float du = u - (float)u0;
    float dv = v - (float)v0;

    const uint8_t *p00 = src + v0 * w + u0;
    const uint8_t *p01 = src + v1 * w + u0;

    float w00 = (1.0f - du) * (1.0f - dv);
    float w10 = du * (1.0f - dv);
    float w01 = (1.0f - du) * dv;
    float w11 = du * dv;

    float val = p00[0] * w00 + p00[1] * w10 + p01[0] * w01 + p01[1] * w11;
    return clip_u8((int)(val + 0.5f));
}

/**
 * @brief 对三通道 (RGB/BGR) 做双线性插值
 */
static void bilinear_rgb(const uint8_t *src, float u, float v,
                         int w, int h, uint8_t *out_r, uint8_t *out_g, uint8_t *out_b)
{
    int u0 = (int)floorf(u);
    int v0 = (int)floorf(v);
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    if (u0 < 0) u0 = 0;
    if (v0 < 0) v0 = 0;
    if (u1 >= w) u1 = w - 1;
    if (v1 >= h) v1 = h - 1;

    float du = u - (float)u0;
    float dv = v - (float)v0;

    float w00 = (1.0f - du) * (1.0f - dv);
    float w10 = du * (1.0f - dv);
    float w01 = (1.0f - du) * dv;
    float w11 = du * dv;

    int idx00 = (v0 * w + u0) * 3;
    int idx10 = (v0 * w + u1) * 3;
    int idx01 = (v1 * w + u0) * 3;
    int idx11 = (v1 * w + u1) * 3;

    float r = src[idx00 + 0] * w00 + src[idx10 + 0] * w10 +
              src[idx01 + 0] * w01 + src[idx11 + 0] * w11;
    float g = src[idx00 + 1] * w00 + src[idx10 + 1] * w10 +
              src[idx01 + 1] * w01 + src[idx11 + 1] * w11;
    float b = src[idx00 + 2] * w00 + src[idx10 + 2] * w10 +
              src[idx01 + 2] * w01 + src[idx11 + 2] * w11;

    *out_r = clip_u8((int)(r + 0.5f));
    *out_g = clip_u8((int)(g + 0.5f));
    *out_b = clip_u8((int)(b + 0.5f));
}

/* ---------- API 实现 ---------- */

int undistort_map_load(undistort_map_t *map,
                       const char *mapx_path,
                       const char *mapy_path)
{
    if (!map || !mapx_path || !mapy_path) {
        fprintf(stderr, "[undistort] invalid arguments\n");
        return -1;
    }

    memset(map, 0, sizeof(undistort_map_t));
    map->width  = UNDISTORT_IMAGE_W;
    map->height = UNDISTORT_IMAGE_H;

    size_t nbytes = (size_t)map->width * map->height * sizeof(float);

    map->mapx = (float *)malloc(nbytes);
    map->mapy = (float *)malloc(nbytes);
    if (!map->mapx || !map->mapy) {
        fprintf(stderr, "[undistort] malloc failed\n");
        undistort_map_free(map);
        return -1;
    }

    /* 加载 mapx */
    FILE *fp = fopen(mapx_path, "rb");
    if (!fp) {
        fprintf(stderr, "[undistort] failed to open %s\n", mapx_path);
        undistort_map_free(map);
        return -1;
    }
    size_t nread = fread(map->mapx, 1, nbytes, fp);
    fclose(fp);
    if (nread != nbytes) {
        fprintf(stderr, "[undistort] %s read error: %zu/%zu\n", mapx_path, nread, nbytes);
        undistort_map_free(map);
        return -1;
    }

    /* 加载 mapy */
    fp = fopen(mapy_path, "rb");
    if (!fp) {
        fprintf(stderr, "[undistort] failed to open %s\n", mapy_path);
        undistort_map_free(map);
        return -1;
    }
    nread = fread(map->mapy, 1, nbytes, fp);
    fclose(fp);
    if (nread != nbytes) {
        fprintf(stderr, "[undistort] %s read error: %zu/%zu\n", mapy_path, nread, nbytes);
        undistort_map_free(map);
        return -1;
    }

    map->loaded = 1;
    printf("[undistort] map loaded: %dx%d, %.1f MB each\n",
           map->width, map->height, nbytes / (1024.0f * 1024.0f));
    return 0;
}

void undistort_map_free(undistort_map_t *map)
{
    if (!map) return;
    if (map->mapx) { free(map->mapx); map->mapx = NULL; }
    if (map->mapy) { free(map->mapy); map->mapy = NULL; }
    map->loaded = 0;
}

int undistort_frame(const undistort_map_t *map, const uint8_t *src, uint8_t *dst)
{
    if (!map || !map->loaded || !src || !dst) return -1;

    int w = map->width;
    int h = map->height;
    int total = w * h;

    for (int i = 0; i < total; i++) {
        float u = map->mapx[i];
        float v = map->mapy[i];

        uint8_t *out = dst + i * 3;
        bilinear_rgb(src, u, v, w, h, &out[0], &out[1], &out[2]);
    }
    return 0;
}

int undistort_frame_gray(const undistort_map_t *map, const uint8_t *src, uint8_t *dst)
{
    if (!map || !map->loaded || !src || !dst) return -1;

    int w = map->width;
    int h = map->height;
    int total = w * h;

    for (int i = 0; i < total; i++) {
        float u = map->mapx[i];
        float v = map->mapy[i];
        dst[i] = bilinear_gray(src, u, v, w, h);
    }
    return 0;
}

/* ---------- 可选: 命令行测试入口 ---------- */

#ifdef UNDISTORT_STANDALONE_TEST

int main(int argc, char *argv[])
{
    if (argc < 4) {
        printf("Usage: %s <input.yuv> <output.yuv> <mapx.bin> <mapy.bin>\n", argv[0]);
        printf("  input/output: raw RGB24 format, %dx%d\n", UNDISTORT_IMAGE_W, UNDISTORT_IMAGE_H);
        return 1;
    }

    const char *in_path  = argv[1];
    const char *out_path = argv[2];
    const char *mx_path  = argv[3];
    const char *my_path  = argv[4];

    size_t frame_bytes = (size_t)UNDISTORT_IMAGE_W * UNDISTORT_IMAGE_H * 3;

    uint8_t *src = (uint8_t *)malloc(frame_bytes);
    uint8_t *dst = (uint8_t *)malloc(frame_bytes);
    if (!src || !dst) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* 加载查找表 */
    undistort_map_t map;
    if (undistort_map_load(&map, mx_path, my_path) != 0) {
        return 1;
    }

    /* 读一帧 */
    FILE *fp_in = fopen(in_path, "rb");
    if (!fp_in) {
        fprintf(stderr, "failed to open %s\n", in_path);
        return 1;
    }
    fread(src, 1, frame_bytes, fp_in);
    fclose(fp_in);

    /* 去畸变 */
    undistort_frame(&map, src, dst);

    /* 写输出 */
    FILE *fp_out = fopen(out_path, "wb");
    if (!fp_out) {
        fprintf(stderr, "failed to open %s\n", out_path);
        return 1;
    }
    fwrite(dst, 1, frame_bytes, fp_out);
    fclose(fp_out);

    printf("[done] undistorted frame saved to %s\n", out_path);

    undistort_map_free(&map);
    free(src);
    free(dst);
    return 0;
}

#endif /* UNDISTORT_STANDALONE_TEST */
