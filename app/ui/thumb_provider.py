from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap

class _Loader(QRunnable):
    def __init__(self, path, cb):
        super().__init__(); self.path, self.cb = path, cb
    def run(self):
        img = QImage(self.path)  # QImage 可在非 GUI 线程用；QPixmap 只能主线程
        if not img.isNull() and max(img.width(), img.height()) > 256:
            img = img.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self.cb(self.path, img)

class ThumbProvider(QObject):
    _loaded = pyqtSignal(str, QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache = {}    # path -> QIcon，上限 500
        self._pending = {}  # path -> callback(QIcon)
        self._loaded.connect(self._on_loaded)
        self.pool = QThreadPool.globalInstance()

    def _on_loaded(self, path, img):
        icon = QIcon(QPixmap.fromImage(img)) if img is not None else QIcon()
        self._cache[path] = icon
        if len(self._cache) > 500:
            for k in list(self._cache)[: len(self._cache) - 500]:
                self._cache.pop(k)
        cb = self._pending.pop(path, None)
        if cb:
            cb(icon)

    def thumb_for(self, path, cb=None):
        if path in self._cache:
            if cb: cb(self._cache[path])
            return self._cache[path]
        if cb and path not in self._pending:
            self._pending[path] = cb
            self.pool.start(_Loader(path, self._loaded.emit))
        return None

    def pixmap_for(self, path, cb):
        # 脸部裁剪用：与 thumb_for 同机制，回调给 QPixmap
        self.thumb_for(path, lambda icon: cb(icon.pixmap(256, 256)))

    def preload(self, paths):
        for p in paths[:60]:
            self.thumb_for(p, None)
