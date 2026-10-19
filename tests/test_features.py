import cv2, numpy as np, pytest
from datetime import datetime
from app.features import extract, hellinger, parse_time, TARGET, cosine_distance, distance

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


def test_cosine_distance_basics():
    a = np.array([1.0, 0, 0, 0], dtype=np.float32)
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert cosine_distance(a, np.array([0, 1.0, 0, 0], dtype=np.float32)) == pytest.approx(1.0)
    # 反向：余弦距离 1-(-1)=2（未归一化输入也要先归一化）
    assert cosine_distance(a, np.array([-5.0, 0, 0, 0], dtype=np.float32)) == pytest.approx(2.0)


def test_distance_dispatch_by_length():
    # SFace/ArcFace 嵌入长度走余弦；旧 LBP 长度走 hellinger —— 同向量自身距离都为 0
    v = _norm_len(128)
    assert distance(v, v) == pytest.approx(0.0, abs=1e-5)
    lbp = extract(_texture(0))
    assert distance(lbp, lbp) == pytest.approx(0.0, abs=1e-5)
    # 长度混用（异常防御）：归为最大距离 1.0 量级，不崩溃
    assert distance(v, lbp) >= 1.0


def _norm_len(n):
    v = np.zeros(n, dtype=np.float32); v[0] = 1.0
    return v
