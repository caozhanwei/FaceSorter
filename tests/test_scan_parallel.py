# -*- coding: utf-8 -*-
"""扫描多线程：计算并行 + 提交按序。
- 并行结果必须与串行完全一致（照片/脸/人物分配/缩略图/裁剪）
- thread_safe 引擎（onnxruntime）真并发；cv2 系引擎加锁串行推理
- 取消/坏文件语义与串行路径一致
"""
import threading, time
from pathlib import Path

import cv2, numpy as np
import pytest

from app.cluster import Clusterer
from app.detector import FaceBox, FaceDetector
from app.pipeline import run_scan
from app.store import Store


def _write_img(path, h=120, w=160, seed=0, bright=False):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * (255 if bright else 64)).astype(np.uint8)
    cv2.imencode(".jpg", img)[1].tofile(str(path))


class _BucketEngine:
    """按图像均值分桶出特征：同桶同人（余弦距离0），异桶远（距离1）。"""

    def __init__(self, n_buckets=3):
        self.n_buckets = n_buckets
        self.detect_lock = threading.Lock()
        self.peak = 0
        self.cur = 0

    def detect(self, img):
        with self.detect_lock:
            self.cur += 1
            self.peak = max(self.peak, self.cur)
        try:
            b = int(img.mean()) % self.n_buckets
            box = FaceBox(10, 10, 80, 80, 100.0, True,
                          row=np.arange(15, dtype=np.float32), score=0.9)
            box.bucket = b
            return [box]
        finally:
            with self.detect_lock:
                self.cur -= 1

    def embed(self, img, box):
        v = np.zeros(128, np.float32)
        v[box.bucket % 128] = 1.0
        return v


def _snapshot(store, root_dirs):
    photos = [dict(r) for r in store.con.execute(
        "SELECT * FROM photos ORDER BY path").fetchall()]
    for p in photos:  # thumb_path 含各 run 自己的目录前缀，比较只看文件名
        p["thumb_path"] = p["thumb_path"] and Path(p["thumb_path"]).name
    faces = [dict(r) for r in store.con.execute(
        "SELECT id, photo_id, x, y, w, h, quality, person_id, source"
        " FROM faces ORDER BY id").fetchall()]
    persons = [dict(r) for r in store.con.execute(
        "SELECT id, name FROM persons ORDER BY id").fetchall()]
    files = sorted(
        p.name for d in root_dirs for p in d.glob("*") if p.is_file())
    return photos, faces, persons, files


def test_parallel_matches_sequential_results(tmp_path):
    # p0 亮图=慢路径，强制乱序完成；提交必须按序，聚类结果才与串行一致
    src = tmp_path / "src"
    src.mkdir()
    _write_img(src / "p0.jpg", seed=0, bright=True)
    for i in range(1, 5):
        _write_img(src / f"p{i}.jpg", seed=i)

    class _SlowFirst(_BucketEngine):
        def detect(self, img):
            if img.mean() > 200:
                time.sleep(0.3)  # 首张最慢：其余先算完，提交顺序不能乱
            return super().detect(img)

    results = []
    for workers in (1, 3):
        d = tmp_path / f"run{workers}"
        d.mkdir()
        store = Store(d / "t.db")
        eng = _SlowFirst()
        run_scan([str(src)], store, eng, Clusterer(store), d / "th",
                 d / "faces", workers=workers)
        results.append(_snapshot(store, [d / "th", d / "faces"]))
    assert results[0] == results[1]


def test_parallel_detects_concurrently(tmp_path):
    # thread_safe 引擎：Barrier(2) 只有真并发才能双双通过（串行必超时破裂）
    src = tmp_path / "src"
    src.mkdir()
    _write_img(src / "a.jpg", seed=1)
    _write_img(src / "b.jpg", seed=2)
    barrier = threading.Barrier(2)

    class _BarrierEngine(_BucketEngine):
        thread_safe = True

        def detect(self, img):
            barrier.wait(timeout=10)
            return super().detect(img)

    store = Store(tmp_path / "t.db")
    run_scan([str(src)], store, _BarrierEngine(), None, tmp_path / "th",
             workers=2)
    assert store.count_photos() == 2
    assert store.count_photos("bad") == 0


def test_non_thread_safe_engine_inference_serialized(tmp_path):
    # cv2 系引擎未声明 thread_safe：detect/embed 必须加锁，峰值并发=1
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        _write_img(src / f"p{i}.jpg", seed=i)

    class _Slow(_BucketEngine):
        def detect(self, img):
            time.sleep(0.03)
            return super().detect(img)

    store = Store(tmp_path / "t.db")
    eng = _Slow()
    run_scan([str(src)], store, eng, None, tmp_path / "th", workers=4)
    assert eng.peak == 1, "非线程安全引擎的推理必须串行"


def test_parallel_cancel_keeps_processed_and_cleans_fresh(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(30):
        _write_img(src / f"p{i}.jpg", seed=i)

    class _Slow(_BucketEngine):
        def detect(self, img):
            time.sleep(0.02)
            return super().detect(img)

    store = Store(tmp_path / "t.db")
    cancel = threading.Event()
    seen = []
    run_scan([str(src)], store, _Slow(), None, tmp_path / "th",
             workers=2,
             progress=lambda i, n: (seen.append(i), cancel.set() if i >= 3 else None),
             cancel=cancel)
    assert 0 < store.count_photos() < 30
    kept = {r["path"] for r in store.photos_for_view("all")}
    assert len(kept) == store.count_photos()  # 未处理的 fresh 行已清


def test_parallel_bad_photo_marked_bad_good_kept(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.jpg").write_bytes(b"not an image")
    _write_img(src / "good.jpg", seed=1)
    store = Store(tmp_path / "t.db")
    run_scan([str(src)], store, _BucketEngine(), None, tmp_path / "th",
             workers=2)
    assert [r["path"] for r in store.photos_for_view("bad")] == [
        (src / "bad.jpg").as_posix()]
    assert store.count_photos() == 1  # count 仅统计 ok：好图 1 张


def test_parallel_disk_full_raises(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _write_img(src / "p.jpg", seed=1)
    store = Store(tmp_path / "t.db")
    import app.pipeline as pl

    def _boom(path, img, quality=88):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pl, "imwrite_unicode", _boom)
    with pytest.raises(OSError):
        run_scan([str(src)], store, _BucketEngine(), None,
                 tmp_path / "th", tmp_path / "faces", workers=2)
