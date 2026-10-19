import cv2, numpy as np, pytest
from datetime import datetime
from app.features import extract, hellinger, parse_time, TARGET

def _texture(seed):
    # 结构化条纹 + 轻噪声：iid 随机噪声的两张图直方图统计相同，分不出同源/异源
    x = np.arange(300)
    base = np.sin(x * (2 + seed * 3) * np.pi / 300.0) * 120 + 128
    img = np.tile(base, (300, 1))
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.integers(-5, 6, img.shape), 0, 255).astype(np.uint8)

def test_extract_shape_and_normalized():
    f = extract(_texture(0))
    assert f.shape == (5540,) and f.dtype == np.float32
    assert abs(np.linalg.norm(f) - 1.0) < 1e-4

def test_same_texture_closer_than_different():
    a = extract(_texture(1))
    a2 = extract(cv2.resize(_texture(1), (200, 200)))          # 同源缩放
    b = extract(_texture(2))                                    # 异源
    assert hellinger(a, a2) < hellinger(a, b)

def test_smooth_vs_textured_separation():
    rng = np.random.default_rng(3)
    smooth = extract((np.ones((200, 200)) * 128).astype(np.uint8))
    smooth2 = extract((np.ones((200, 200)) * 130).astype(np.uint8))
    tex = extract((rng.random((200, 200)) * 255).astype(np.uint8))
    assert hellinger(smooth, smooth2) < hellinger(smooth, tex)

def test_parse_time():
    assert parse_time("2026:01:02 10:00:00") == datetime(2026, 1, 2, 10, 0)
    assert parse_time(None) is None and parse_time("垃圾") is None
