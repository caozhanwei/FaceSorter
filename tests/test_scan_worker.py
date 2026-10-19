import numpy as np, pytest
from PyQt6.QtCore import QTimer
from app.store import Store
from app.detector import FaceDetector
from app.ui.scan_progress import ScanWorker

def _write_img(path, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", (rng.random((60, 80)) * 255).astype(np.uint8))[1].tofile(str(path))

def test_worker_runs_to_done(tmp_path, qapp):
    for i in range(2):
        _write_img(tmp_path / f"p{i}.jpg", seed=i)
    store = Store(tmp_path / "t.db")
    w = ScanWorker([str(tmp_path)], store, FaceDetector(), None,
                   tmp_path / "th")
    got = []
    w.progress.connect(lambda i, n: got.append((i, n)))
    w.done.connect(qapp.quit)
    w.start(); qapp.exec()
    assert got[-1] == (2, 2) and store.count_photos() == 2

def test_worker_cancel(tmp_path, qapp):
    for i in range(50):
        _write_img(tmp_path / f"p{i}.jpg", seed=i)
    store = Store(tmp_path / "t.db")
    w = ScanWorker([str(tmp_path)], store, FaceDetector(), None,
                   tmp_path / "th")
    w.progress.connect(lambda i, n: w.cancel() if i >= 2 else None)
    w.done.connect(qapp.quit)
    w.start(); qapp.exec()
    assert 0 < store.count_photos() < 50

def test_worker_pause_then_resume_completes(tmp_path, qapp):
    for i in range(3):
        _write_img(tmp_path / f"p{i}.jpg", seed=i)
    store = Store(tmp_path / "t.db")
    w = ScanWorker([str(tmp_path)], store, FaceDetector(), None, tmp_path / "th")
    w.progress.connect(lambda i, n: w.pause() if i == 1 else None)
    QTimer.singleShot(300, w.resume)  # 暂停后 300ms 自动恢复
    w.done.connect(qapp.quit)
    w.start(); qapp.exec()
    assert store.count_photos() == 3
