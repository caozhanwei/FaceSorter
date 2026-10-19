import os, threading, cv2, numpy as np, pytest
from pathlib import Path
from app.store import Store
from app.detector import FaceDetector
from app.cluster import Clusterer
from app.pipeline import imread_unicode, imwrite_unicode, read_exif_taken_at, \
    make_thumb, process_one, run_scan

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

def _write_img(path: Path, h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    cv2.imencode(".jpg", (rng.random((h, w)) * 255).astype(np.uint8))[1].tofile(str(path))

def test_unicode_roundtrip(tmp_path):
    p = tmp_path / "中文 目录" / "照片 1.jpg"
    p.parent.mkdir(parents=True)
    _write_img(p)
    img = imread_unicode(str(p))
    assert img is not None and img.shape[:2] == (120, 160)
    assert imwrite_unicode(str(tmp_path / "缩 略.jpg"), img)
    assert (tmp_path / "缩 略.jpg").exists()

def test_thumb_size(tmp_path):
    img = np.zeros((1000, 500), np.uint8)
    make_thumb(img, tmp_path / "t.jpg")
    t = cv2.imdecode(np.fromfile(str(tmp_path / "t.jpg"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert max(t.shape) <= 256

def test_exif_missing_ok(tmp_path):
    p = tmp_path / "x.jpg"; _write_img(p)
    assert read_exif_taken_at(str(p)) is None

def test_process_one_bad_file(tmp_path):
    store = Store(tmp_path / "t.db")
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"not an image")
    process_one(str(bad), store, FaceDetector(), None, tmp_path / "thumbs")
    assert [r["path"] for r in store.photos_for_view("bad")] == [str(bad).replace("\\", "/")]

@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "face1.jpg")),
                    reason="需要 tests/assets/face1.jpg")
def test_process_one_real_face(tmp_path):
    store = Store(tmp_path / "t.db")
    det = FaceDetector(); cl = Clusterer(store)
    process_one(os.path.join(ASSETS, "face1.jpg").replace("\\", "/"),
                store, det, cl, tmp_path / "thumbs")
    assert store.count_photos() == 1
    rows = store.con.execute("SELECT * FROM faces").fetchall()
    assert len(rows) >= 1 and rows[0]["person_id"] is not None
    photo = store.photos_for_view("with_faces")[0]
    assert photo["thumb_path"] and os.path.exists(photo["thumb_path"])
    assert photo["has_face"] == 1

def test_run_scan_progress_and_cancel(tmp_path):
    store = Store(tmp_path / "t.db")
    for i in range(3):
        _write_img(tmp_path / f"p{i}.jpg", seed=i)
    seen = []
    cancel = threading.Event()
    run_scan([str(tmp_path)], store, FaceDetector(), None, tmp_path / "th",
             progress=lambda i, n: seen.append((i, n)), cancel=cancel)
    assert seen[-1][0] == 3 and store.count_photos() == 3
    # 再扫一遍：无新文件,progress 不再推进
    seen.clear()
    run_scan([str(tmp_path)], store, FaceDetector(), None, tmp_path / "th",
             progress=lambda i, n: seen.append((i, n)), cancel=None)
    assert seen == []

def test_run_scan_disk_full_raises(tmp_path, monkeypatch):
    store = Store(tmp_path / "t.db")
    _write_img(tmp_path / "p.jpg")
    import app.pipeline as pl
    def _boom(img, thumb_path):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(pl, "make_thumb", _boom)
    with pytest.raises(OSError):
        run_scan([str(tmp_path)], store, FaceDetector(), None, tmp_path / "th")


class _FakeDeepEngine:
    """有 embed 的引擎：process_one 必须走深度嵌入而不是 LBP。"""
    def __init__(self):
        self.embed_calls = []
    def detect(self, img):
        from app.detector import FaceBox
        return [FaceBox(10, 10, 80, 80, 100.0, True,
                        row=np.arange(15, dtype=np.float32), score=0.9)]
    def embed(self, img, box):
        self.embed_calls.append(tuple(box.row.shape))
        v = np.zeros(128, np.float32); v[0] = 1.0
        return v

def test_process_one_uses_engine_embed(tmp_path):
    store = Store(tmp_path / "t.db")
    eng = _FakeDeepEngine()
    p = tmp_path / "p.jpg"; _write_img(p)
    process_one(p.as_posix(), store, eng, None, tmp_path / "thumbs")
    rows = store.con.execute("SELECT feature FROM faces").fetchall()
    assert eng.embed_calls == [(15,)]
    assert len(rows) == 1
    assert np.frombuffer(rows[0]["feature"], np.float32).shape == (128,)

def test_process_one_embed_failure_falls_back_to_lbp(tmp_path, monkeypatch):
    # embed 抛异常（如模型损坏）：降级 LBP，照片仍被索引，扫描不中断
    store = Store(tmp_path / "t.db")
    class _Broken(_FakeDeepEngine):
        def embed(self, img, box):
            raise RuntimeError("onnx broken")
    p = tmp_path / "p.jpg"; _write_img(p)
    process_one(p.as_posix(), store, _Broken(), None, tmp_path / "thumbs")
    rows = store.con.execute("SELECT feature FROM faces").fetchall()
    assert len(rows) == 1
    assert np.frombuffer(rows[0]["feature"], np.float32).shape == (5540,)


# ---- 扫描期人脸裁剪落盘（清晰头像数据源，替代缩略图二次裁剪） ----

class _FakeRowEngine:
    """带五点行的引擎（YuNet/SCRFD 同布局），无 embed。"""
    def detect(self, img):
        from app.detector import FaceBox
        return [FaceBox(10, 10, 80, 80, 100.0, True,
                        row=np.arange(15, dtype=np.float32), score=0.9)]

def test_process_one_saves_aligned_face_crop(tmp_path):
    from app.facecrop import FACE_SIZE
    store = Store(tmp_path / "t.db")
    p = tmp_path / "p.jpg"; _write_img(p)
    process_one(p.as_posix(), store, _FakeRowEngine(), None,
                tmp_path / "thumbs", tmp_path / "faces")
    fid = store.con.execute("SELECT id FROM faces").fetchone()["id"]
    fp = tmp_path / "faces" / f"{fid}.jpg"
    assert fp.exists()
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (FACE_SIZE, FACE_SIZE)

def test_process_one_saves_rect_crop_without_row(tmp_path):
    # Haar 回退无关键点：也要落盘矩形裁剪
    class _NoRow(_FakeRowEngine):
        def detect(self, img):
            from app.detector import FaceBox
            return [FaceBox(10, 10, 80, 60, 100.0, True)]
    store = Store(tmp_path / "t.db")
    p = tmp_path / "p.jpg"; _write_img(p)
    process_one(p.as_posix(), store, _NoRow(), None,
                tmp_path / "thumbs", tmp_path / "faces")
    fid = store.con.execute("SELECT id FROM faces").fetchone()["id"]
    assert (tmp_path / "faces" / f"{fid}.jpg").exists()

def test_run_scan_excludes_face_dir(tmp_path):
    # faces 与 thumbs 同级：若扫描根覆盖数据目录，裁剪图不得被当照片索引
    store = Store(tmp_path / "t.db")
    for i in range(2):
        _write_img(tmp_path / f"p{i}.jpg", seed=i)
    (tmp_path / "faces").mkdir()
    cv2.imencode(".jpg", np.zeros((8, 8), np.uint8))[1].tofile(
        str(tmp_path / "faces" / "999.jpg"))
    run_scan([str(tmp_path)], store, FaceDetector(), None,
             tmp_path / "thumbs", tmp_path / "faces")
    paths = [r["path"] for r in store.photos_for_view("all")]
    assert not any("/faces/" in p for p in paths), paths
