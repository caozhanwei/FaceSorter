import threading
from PyQt6.QtCore import QThread, pyqtSignal
from app.pipeline import run_scan

class ScanWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, roots, store, detector, clusterer, thumb_dir, face_dir=None):
        super().__init__()
        self.roots, self.store, self.detector = roots, store, detector
        self.clusterer, self.thumb_dir, self.face_dir = clusterer, thumb_dir, face_dir
        self._cancel = threading.Event()
        self._resume = threading.Event(); self._resume.set()

    def cancel(self):
        self._cancel.set(); self._resume.set()

    def pause(self):
        self._resume.clear()

    def resume(self):
        self._resume.set()

    def run(self):
        def _progress(i, n):
            self.progress.emit(i, n)
            self.msleep(1)  # 让接收线程有机会处理信号（暂停/取消）再继续
            while not self._resume.is_set() and not self._cancel.is_set():
                self._resume.wait(0.2)
        try:
            run_scan(self.roots, self.store, self.detector, self.clusterer,
                     self.thumb_dir, progress=_progress, cancel=self._cancel)
        except Exception as e:
            self.failed.emit(str(e))
        self.done.emit()
