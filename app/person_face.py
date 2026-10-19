# -*- coding: utf-8 -*-
"""人物平均脸：同模板对齐裁剪的像素均值（纯函数，供 UI 线程外计算）。"""
import cv2, numpy as np


def imread_crop(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None


def average_face(paths, max_n=16):
    """按给定优先级（通常为质量降序）取最多 max_n 张可用裁剪求均值。

    缺失/损坏文件不占采样名额；一张可用都没有返回 None。
    所有裁剪来自同一对齐模板，均值即"平均脸"，模板几何对齐保证不糊。
    """
    acc, n = None, 0
    for p in paths:
        if n >= max_n:
            break
        img = imread_crop(p)
        if img is None:
            continue
        f = img.astype(np.float32)
        acc = f if acc is None else acc + f
        n += 1
    if acc is None:
        return None
    return (acc / n).astype(np.uint8)
