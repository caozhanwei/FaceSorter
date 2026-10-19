from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QListView, QLabel, QStatusBar)
from app.settings import load_settings
from app.ui.thumb_provider import ThumbProvider

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
