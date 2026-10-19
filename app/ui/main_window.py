from PyQt6.QtCore import (Qt, pyqtSignal, QUrl, QMimeData, QByteArray,
                          QObject, QRunnable, QThreadPool)
from PyQt6.QtGui import (QStandardItemModel, QStandardItem, QPixmap, QIcon,
                         QDesktopServices, QDrag)
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QListView, QLabel, QStatusBar, QListWidget,
    QListWidgetItem, QMenu, QInputDialog, QDialog, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialogButtonBox, QMessageBox, QPushButton, QProgressBar,
    QFileDialog, QAbstractItemView)
from app.settings import load_settings, save_settings
from app.paths import data_dir
from app.ui.thumb_provider import ThumbProvider
from app.ui.scan_progress import ScanWorker
from app.ui.persons_model import (MIME_FACE_IDS, persons_display,
                                  encode_face_ids, decode_face_ids)

VIEW_ALL, VIEW_WITH, VIEW_NO, VIEW_BAD = "全部照片", "有人的", "无人的", "问题文件"
SPECIAL_VIEWS = ["未归类脸", "全部照片", "有人的", "无人的", "问题文件"]

PID_ROLE = Qt.ItemDataRole.UserRole + 1


class PersonsList(QListWidget):
    person_activated = pyqtSignal(int)
    person_drop = pyqtSignal(int, object)  # list 参数用 object 传，(int,list) 声明在 PyQt6 下 marshal 崩溃

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._pid_role = PID_ROLE

    def mimeTypes(self):
        return [MIME_FACE_IDS]

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_FACE_IDS):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME_FACE_IDS):
            e.acceptProposedAction()

    def dropEvent(self, e):
        item = self.itemAt(e.position().toPoint())
        if item is not None and item.data(self._pid_role):
            ids = decode_face_ids(bytes(e.mimeData().data(MIME_FACE_IDS)))
            e.acceptProposedAction()
            self.person_drop.emit(item.data(self._pid_role), ids)

    def contextMenuEvent(self, e):
        item = self.itemAt(e.pos())
        if item is None or not item.data(self._pid_role):
            return
        pid = item.data(self._pid_role)
        menu = QMenu(self)
        menu.addAction("重命名", lambda: self.window()._rename_person(pid))
        menu.addAction("合并到…", lambda: self.window()._merge_person(pid))
        menu.addAction("置顶/取消置顶", lambda: self.window()._toggle_pin(pid))
        menu.exec(e.globalPos())


class FaceView(QListView):
    """缩略图墙；拖拽时把选中脸部 id 以自定义 mime 发出（spec §6）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def startDrag(self, actions):
        face_ids = []
        for idx in self.selectedIndexes():
            data = idx.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and data[0] == "face":
                face_ids.append(data[1])
        if not face_ids:
            return
        md = QMimeData()
        md.setData(MIME_FACE_IDS, QByteArray(encode_face_ids(face_ids)))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.exec(Qt.DropAction.CopyAction)


class _ExportSignals(QObject):
    done = pyqtSignal(int, str)    # (数量, 人物标签)
    failed = pyqtSignal(str)


class _ExportRunner(QRunnable):
    def __init__(self, store, person_id, dest):
        super().__init__()
        self.store, self.person_id, self.dest = store, person_id, dest
        self.signals = _ExportSignals()

    def run(self):
        from app.export import export_person
        try:
            label = ""
            for r in self.store.persons_with_counts():
                if r["id"] == self.person_id:
                    label = r["name"] or f"人物 {r['id']}"
                    break
            copied = export_person(self.store, self.person_id, self.dest)
            self.signals.done.emit(len(copied), label)
        except Exception as e:
            self.signals.failed.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent); self.setWindowTitle("设置")
        form = QFormLayout(self)
        self.thr = QDoubleSpinBox(); self.thr.setRange(0.05, 0.9); self.thr.setSingleStep(0.01)
        self.thr.setDecimals(2)
        self.thr.setValue(cfg["threshold"])
        self.boost = QDoubleSpinBox(); self.boost.setRange(1.0, 1.5); self.boost.setSingleStep(0.05)
        self.boost.setValue(cfg["time_boost"])
        self.win_min = QSpinBox(); self.win_min.setRange(0, 240); self.win_min.setValue(cfg["time_window"])
        self.thumb_dir = QLineEdit(cfg.get("thumb_dir") or "")
        form.addRow("聚类阈值(越小越严，不易并堆)", self.thr)
        form.addRow("时间加成系数", self.boost)
        form.addRow("时间窗口(分钟)", self.win_min)
        form.addRow("缩略图缓存目录(留空=exe旁data/thumbs)", self.thumb_dir)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def result_cfg(self, cfg):
        cfg = dict(cfg)
        cfg.update(threshold=self.thr.value(), time_boost=self.boost.value(),
                   time_window=self.win_min.value(),
                   thumb_dir=self.thumb_dir.text().strip() or None)
        return cfg


class MainWindow(QMainWindow):
    def __init__(self, store, detector_factory, settings_path, clusterer_factory=None,
                 engine_label=""):
        super().__init__()
        self.store = store
        self.settings_path = settings_path
        self._detector_factory = detector_factory
        self._clusterer_factory = clusterer_factory
        self._cfg = load_settings(settings_path)
        self.clusterer = clusterer_factory() if clusterer_factory else None
        self.setWindowTitle(f"FaceSorter 照片人物分类{engine_label}")
        self.thumbs = ThumbProvider(self)
        top = QHBoxLayout()
        self.view_selector = QComboBox()
        self.view_selector.addItems([VIEW_ALL, VIEW_WITH, VIEW_NO, VIEW_BAD])
        self.view_selector.currentTextChanged.connect(lambda _: self.refresh_view())
        self.search_box = QLineEdit(placeholderText="按文件名/人名过滤")
        self.search_box.textChanged.connect(self._search_changed)
        top.addWidget(self.view_selector); top.addWidget(self.search_box, 1)
        self.grid_model = QStandardItemModel(self)
        self.grid = FaceView()
        self.grid.setModel(self.grid_model)
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setIconSize(self.grid.iconSize() * 4)
        self.grid.setSpacing(6)
        self.grid.doubleClicked.connect(self._open_item)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

        self.persons_list = PersonsList()
        self.persons_list.itemActivated.connect(self._person_clicked)
        self.persons_list.itemClicked.connect(self._person_clicked)
        # 直接 connect 绑定方法在本机 PyQt6 上 emit 即原生崩溃，包一层 lambda 规避
        self.persons_list.person_drop.connect(
            lambda pid, ids: self._move_faces(ids, pid))

        body = QVBoxLayout()
        wrap = QWidget(); wrap.setLayout(body)
        body.addLayout(top)
        split = QHBoxLayout()
        split.addWidget(self.persons_list)
        split.addWidget(self.grid, 1)
        self.persons_list.setMaximumWidth(220)
        self.persons_list.setMinimumWidth(160)
        body.addLayout(split, 1)
        self.setCentralWidget(wrap)
        self.statusBar().showMessage("就绪")
        self._detector = None
        self.roots = []
        self._worker = None
        self._paused = False
        self.btn_dir = QPushButton("选择目录"); self.btn_scan = QPushButton("扫描")
        self.btn_pause = QPushButton("暂停"); self.btn_cancel = QPushButton("取消")
        self.btn_settings = QPushButton("设置")
        self.btn_export = QPushButton("导出")
        self.prog = QProgressBar()
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setFormat("扫描中 %v/%m")
        self.prog.hide()  # 空闲不显示；勿用 setMaximum(0)——那是永久忙碌动画
        self.btn_dir.clicked.connect(self._pick_roots)
        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cancel.clicked.connect(lambda: self._worker and self._worker.cancel())
        self.btn_settings.clicked.connect(self._open_settings)
        self.btn_export.clicked.connect(self._export_person)
        top.insertWidget(0, self.btn_dir); top.insertWidget(1, self.btn_scan)
        top.insertWidget(2, self.btn_pause); top.insertWidget(3, self.btn_cancel)
        top.addWidget(self.prog)
        top.insertWidget(4, self.btn_settings)
        top.insertWidget(5, self.btn_export)
        self.refresh_persons()

    # ---- 搜索：同时刷中区视图与左栏人物 ----
    def _search_changed(self, _):
        self.refresh_view()
        self.refresh_persons()

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
        self._worker.progress.connect(self.on_scan_progress)
        self._worker.done.connect(self.on_scan_done)
        self._worker.failed.connect(self.on_scan_failed)
        self.btn_scan.setEnabled(False)
        self.prog.setRange(0, 0)  # 清点目录阶段总数未知，短暂忙碌
        self.prog.show()
        self.statusBar().showMessage("开始扫描：正在清点目录…")
        self._worker.start()

    def on_scan_progress(self, i, n):
        self.prog.show()
        self.prog.setMaximum(n)
        self.prog.setValue(i)
        self.statusBar().showMessage(f"扫描中 {i}/{n}")

    def on_scan_done(self):
        self.prog.hide()
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.btn_scan.setEnabled(True)
        self._paused = False; self.btn_pause.setText("暂停")
        self.refresh_view(); self.refresh_persons()
        if getattr(self, "_worker", None) and self._worker._cancel.is_set():
            self.statusBar().showMessage("已取消（已处理部分保留，下次续扫）")
        else:
            self.statusBar().showMessage(
                f"扫描完成 · 共 {self.store.count_photos()} 张")

    def on_scan_failed(self, msg):
        self.prog.hide()
        self.btn_scan.setEnabled(True)
        self.statusBar().showMessage(f"扫描失败: {msg}")

    def _toggle_pause(self):
        if not self._worker:
            return
        if self._paused:
            self._worker.resume(); self._paused = False; self.btn_pause.setText("暂停")
            self.statusBar().showMessage("继续扫描")
        else:
            self._worker.pause(); self._paused = True; self.btn_pause.setText("继续")
            self.statusBar().showMessage("已暂停")

    def _open_settings(self):
        dlg = SettingsDialog(self._cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._cfg = dlg.result_cfg(self._cfg)
            save_settings(self.settings_path, self._cfg)
            if self._clusterer_factory:
                self.clusterer = self._clusterer_factory()
            self.refresh_view(); self.refresh_persons()

    # ---- 导出 ----
    def _export_person(self):
        rows = persons_display(self.store.persons_with_counts())
        if not rows:
            QMessageBox.information(self, "导出", "还没有任何人物可导出。")
            return
        names = [d["label"] for d in rows]
        choice, ok = QInputDialog.getItem(self, "导出人物", "选择人物:", names, 0, False)
        if not ok:
            return
        pid = rows[names.index(choice)]["id"]
        dest = QFileDialog.getExistingDirectory(self, "选择导出目标目录")
        if not dest:
            return
        self.btn_export.setEnabled(False)
        self.statusBar().showMessage("正在导出…")
        self._export_runner = _ExportRunner(self.store, pid, dest)
        self._export_runner.signals.done.connect(self._export_done)
        self._export_runner.signals.failed.connect(self._export_failed)
        QThreadPool.globalInstance().start(self._export_runner)

    def _export_done(self, n, label):
        self.btn_export.setEnabled(True)
        self.statusBar().showMessage("就绪")
        QMessageBox.information(self, "导出完成",
                                f"已为「{label}」导出 {n} 张照片副本（原文件未动）。")

    def _export_failed(self, msg):
        self.btn_export.setEnabled(True)
        self.statusBar().showMessage("就绪")
        QMessageBox.warning(self, "导出失败", msg)

    # ---- 人物列表 ----
    def refresh_persons(self):
        self.persons_list.clear()
        flt = self.search_box.text().strip().lower()  # 按人名过滤左栏（spec §6）
        for d in persons_display(self.store.persons_with_counts()):
            if flt and flt not in d["label"].lower():
                continue
            it = QListWidgetItem(f"{'📌' if d['pinned'] else ''}{d['label']}({d['cnt']})")
            it.setData(PID_ROLE, d["id"])
            self.persons_list.addItem(it)
        for label in SPECIAL_VIEWS:
            it = QListWidgetItem(label)
            self.persons_list.addItem(it)

    def _person_clicked(self, item):
        self.refresh_view()

    def _current_person_view(self):
        """返回 (mode, person_id)：mode ∈ face_person/face_unassigned"""
        item = self.persons_list.currentItem()
        if item is None:
            return None, None
        if item.text() == "未归类脸":
            return "face_unassigned", None
        pid = item.data(PID_ROLE)
        return ("face_person", pid) if pid else (None, None)

    def _face_rows_for_current(self):
        mode, pid = self._current_person_view()
        if mode == "face_person":
            return self.store.faces_of_person(pid)
        if mode == "face_unassigned":
            return self.store.faces_without_person()
        return []

    # ---- 中区 ----
    def refresh_view(self):
        mode, _ = self._current_person_view()
        if mode:
            self.grid_model.clear()
            rows = self._face_rows_for_current()
            flt = self.search_box.text().strip().lower()
            for r in rows:
                if flt and flt not in r["path"].lower():
                    continue
                item = QStandardItem(r["path"].rsplit("/", 1)[-1])
                item.setData(("face", r["id"]), Qt.ItemDataRole.UserRole)
                item.setEditable(False)
                self.grid_model.appendRow(item)
                self._face_icon_async(item, r)
            self.statusBar().showMessage(f"{len(rows)} 张脸")
            return
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

    def _face_icon_async(self, item, r):
        if not r["thumb_path"]:
            return
        def _make(pm_thumb, r=r, item=item):
            if pm_thumb.isNull() or not r["pw"]:
                return
            sx = pm_thumb.width() / r["pw"]; sy = pm_thumb.height() / r["ph"]
            x, y = int(r["x"]*sx), int(r["y"]*sy)
            w, h = int(r["w"]*sx), int(r["h"]*sy)
            crop = pm_thumb.copy(max(0, x), max(0, y), max(8, w), max(8, h)).scaled(128, 128)
            # setIcon 必须传 QIcon：传 QPixmap 在 PyQt6 下原生段错误（点击人物闪退根因）
            item.setIcon(QIcon(crop))
        self.thumbs.pixmap_for(r["thumb_path"], _make)

    # ---- 人物操作 ----
    def _rename_person(self, pid):
        name, ok = QInputDialog.getText(self, "重命名", "人物姓名:")
        if ok and name.strip():
            self.store.rename_person(pid, name.strip())
            if self.clusterer: self.clusterer.invalidate(pid)
            self.refresh_persons()

    def _merge_person(self, pid):
        rows = [r for r in self.store.persons_with_counts() if r["id"] != pid]
        names = [r["name"] or f"人物 {r['id']}" for r in rows]
        if not names:
            return
        choice, ok = QInputDialog.getItem(self, "合并到…", "目标人物:", names, 0, False)
        if ok:
            dst = rows[names.index(choice)]["id"]
            self.store.merge_persons(pid, dst)
            if self.clusterer:
                self.clusterer.invalidate(dst)
            self.refresh_persons(); self.refresh_view()

    def _toggle_pin(self, pid):
        row = [r for r in self.store.persons_with_counts() if r["id"] == pid][0]
        self.store.set_pinned(pid, not row["pinned"])
        self.refresh_persons()

    def _move_faces(self, face_ids, dst_pid):
        for fid in face_ids:
            self.store.set_face_person(fid, dst_pid, "manual")
        if self.clusterer: self.clusterer.invalidate(dst_pid)
        self.refresh_persons(); self.refresh_view()

    def _new_person_from_faces(self, face_ids):
        pid = self.store.create_person()
        self._move_faces(face_ids, pid)

    def _remove_from_person(self, face_ids):
        for fid in face_ids:
            self.store.set_face_person(fid, None, "manual")
        self.refresh_persons(); self.refresh_view()

    def _open_item(self, index):
        data = index.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple) and data[0] == "face":
            data = self.store.photo_of_face(data[1])["path"]
        if data:
            QDesktopServices.openUrl(QUrl.fromLocalFile(data))  # 双击看整照片

    # ---- 脸部右键菜单 ----
    def contextMenuEvent(self, e):
        idx = self.grid.indexAt(self.grid.mapFrom(self, e.pos()))
        if not idx.isValid():
            return
        data = idx.data(Qt.ItemDataRole.UserRole)
        if not (isinstance(data, tuple) and data[0] == "face"):
            return
        face_ids = [d.data(Qt.ItemDataRole.UserRole)[1]
                    for d in self.grid.selectedIndexes()
                    if isinstance(d.data(Qt.ItemDataRole.UserRole), tuple)]
        menu = QMenu(self)
        move = menu.addMenu("移动到…")
        for d in persons_display(self.store.persons_with_counts()):
            move.addAction(d["label"], lambda pid=d["id"]: self._move_faces(face_ids, pid))
        menu.addAction("新建人物", lambda: self._new_person_from_faces(face_ids))
        menu.addAction("从这人移除", lambda: self._remove_from_person(face_ids))
        menu.exec(e.globalPos())
