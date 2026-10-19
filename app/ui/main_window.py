from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QListView, QLabel, QStatusBar,
    QPushButton, QProgressBar, QFileDialog)
from app.settings import load_settings
from app.paths import data_dir
from app.ui.thumb_provider import ThumbProvider
from app.ui.scan_progress import ScanWorker

VIEW_ALL, VIEW_WITH, VIEW_NO, VIEW_BAD = "全部照片", "有人的", "无人的", "问题文件"

class MainWindow(QMainWindow):
    def __init__(self, store, detector_factory, settings_path, clusterer_factory=None):
        super().__init__()
        self.store = store
        self.settings_path = settings_path
        self._detector_factory = detector_factory
        self._cfg = load_settings(settings_path)
        self.clusterer = clusterer_factory() if clusterer_factory else None
        self.setWindowTitle("FaceSorter 照片人物分类")
        self.thumbs = ThumbProvider(self)
        top = QHBoxLayout()
        self.view_selector = QComboBox()
        self.view_selector.addItems([VIEW_ALL, VIEW_WITH, VIEW_NO, VIEW_BAD])
        self.view_selector.currentTextChanged.connect(lambda _: self.refresh_view())
        self.search_box = QLineEdit(placeholderText="按文件名过滤")
        self.search_box.textChanged.connect(lambda _: self.refresh_view())
        top.addWidget(self.view_selector); top.addWidget(self.search_box, 1)
        self.grid_model = QStandardItemModel(self)
        self.grid = QListView()
        self.grid.setModel(self.grid_model)
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setIconSize(self.grid.iconSize() * 4)
        self.grid.setSpacing(6)
        body = QVBoxLayout()
        wrap = QWidget(); wrap.setLayout(body)
        body.addLayout(top); body.addWidget(self.grid, 1)
        self.setCentralWidget(wrap)
        self.statusBar().showMessage("就绪")
        self._detector = None
        self.roots = []
        self._worker = None
        self._paused = False
        self.btn_dir = QPushButton("选择目录"); self.btn_scan = QPushButton("扫描")
        self.btn_pause = QPushButton("暂停"); self.btn_cancel = QPushButton("取消")
        self.prog = QProgressBar(); self.prog.setMaximum(0)
        self.btn_dir.clicked.connect(self._pick_roots)
        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cancel.clicked.connect(lambda: self._worker and self._worker.cancel())
        top.insertWidget(0, self.btn_dir); top.insertWidget(1, self.btn_scan)
        top.insertWidget(2, self.btn_pause); top.insertWidget(3, self.btn_cancel)
        top.addWidget(self.prog)

    def _thumb_dir(self):
        return self._cfg.get("thumb_dir") or (data_dir() / "thumbs")

    def _pick_roots(self):
        d = QFileDialog.getExistingDirectory(self, "选择照片目录")
        if d:
            self.roots = [d]; self.statusBar().showMessage(f"目录: {d}")

    def _start_scan(self):
        if not self.roots or getattr(self, "_worker", None) and self._worker.isRunning():
            return
        det = self._detector_factory()
        self._worker = ScanWorker(self.roots, self.store, det, self.clusterer,
                                  self._thumb_dir())
        self._worker.progress.connect(lambda i, n: (self.prog.setMaximum(n),
                                                    self.prog.setValue(i)))
        self._worker.done.connect(lambda: (self.refresh_view(), self.refresh_persons()))
        self._worker.failed.connect(lambda m: self.statusBar().showMessage(f"扫描失败: {m}"))
        self._worker.start()

    def _toggle_pause(self):
        if not self._worker:
            return
        if self._paused:
            self._worker.resume(); self._paused = False; self.btn_pause.setText("暂停")
        else:
            self._worker.pause(); self._paused = True; self.btn_pause.setText("继续")

    def refresh_persons(self):
        self.refresh_view()  # Task 10 替换为人物列表刷新

    def refresh_view(self):
        self.grid_model.clear()
        view = {"全部照片": "all", "有人的": "with_faces",
                "无人的": "no_faces", "问题文件": "bad"}[self.view_selector.currentText()]
        flt = self.search_box.text().strip() or None
        rows = self.store.photos_for_view(view, flt)
        for r in rows:
            item = QStandardItem(r["path"].rsplit("/", 1)[-1])
            item.setData(r["path"], Qt.ItemDataRole.UserRole)
            item.setEditable(False)
            self.grid_model.appendRow(item)
            if r["thumb_path"]:
                self.thumbs.thumb_for(r["thumb_path"],
                    lambda icon, it=item: it.setIcon(icon))
        self.statusBar().showMessage(f"{len(rows)} 张 · 共 {self.store.count_photos()} 张")
