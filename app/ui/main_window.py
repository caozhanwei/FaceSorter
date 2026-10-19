from PyQt6.QtCore import (Qt, pyqtSignal, QMimeData, QByteArray,
                          QObject, QRunnable, QThreadPool)
from PyQt6.QtGui import (QStandardItemModel, QStandardItem, QPixmap, QImage, QIcon,
                         QDrag, QAction, QKeySequence, QShortcut)
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QListView, QLabel, QStatusBar, QListWidget,
    QListWidgetItem, QMenu, QInputDialog, QDialog, QFormLayout, QDoubleSpinBox,
    QSpinBox, QDialogButtonBox, QMessageBox, QPushButton, QProgressBar,
    QFileDialog, QAbstractItemView, QToolBar, QStyle)
from app.settings import load_settings, save_settings, need_recluster
from app.paths import data_dir
from app.person_face import average_face
from app.ui.thumb_provider import ThumbProvider
from app.ui.scan_progress import ScanWorker
from app.ui.photo_viewer import PhotoViewer
from app.ui.persons_model import (MIME_FACE_IDS, persons_display,
                                  encode_face_ids, decode_face_ids)

VIEW_ALL, VIEW_WITH, VIEW_NO, VIEW_BAD = "全部照片", "有人的", "无人的", "问题文件"
SPECIAL_VIEWS = ["全部脸", "未归类脸", "全部照片", "有人的", "无人的", "问题文件"]

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
        menu.addAction("管理此人的脸", lambda: self.window()._manage_faces(pid))
        menu.addSeparator()
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


class _AvgFaceSignals(QObject):
    ready = pyqtSignal(int, object)  # (person_id, QImage|None)


class _AvgFaceRunner(QRunnable):
    """后台算人物平均脸（≤16 张对齐裁剪均值），完成后回主线程设图标。"""

    def __init__(self, store, pid, face_dir, signals):
        super().__init__()
        self.store, self.pid, self.face_dir = store, pid, face_dir
        self.signals = signals

    def run(self):
        try:
            rows = self.store.face_crops_of_person(self.pid)
            paths = [self.face_dir / f"{r['id']}.jpg" for r in rows]
            img = average_face(paths)
        except Exception:
            img = None
        qimg = None
        if img is not None:
            h, w = img.shape[:2]
            # numpy 缓冲跨线程须 copy 固化；BGR 三通道对应 Format_BGR888
            qimg = QImage(img.data, w, h, 3 * w,
                          QImage.Format.Format_BGR888).copy()
        self.signals.ready.emit(self.pid, qimg)


class _ReclusterSignals(QObject):
    done = pyqtSignal()
    failed = pyqtSignal(str)


class _ReclusterRunner(QRunnable):
    """后台按（新）阈值重排全部自动分配的脸。"""

    def __init__(self, clusterer, signals):
        super().__init__()
        self.clusterer, self.signals = clusterer, signals

    def run(self):
        try:
            self.clusterer.recluster_auto()
            self.signals.done.emit()
        except Exception as e:
            self.signals.failed.emit(str(e))


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
        self.threads = QSpinBox(); self.threads.setRange(1, 32)
        self.threads.setValue(cfg.get("scan_threads", 4))
        self.threads.setToolTip("同时处理的照片数：解码/检测并行，不改变聚类结果")
        self.thumb_dir = QLineEdit(cfg.get("thumb_dir") or "")
        form.addRow("聚类阈值(越小越严，不易并堆)", self.thr)
        form.addRow("时间加成系数", self.boost)
        form.addRow("时间窗口(分钟)", self.win_min)
        form.addRow("扫描线程数", self.threads)
        form.addRow("缩略图缓存目录(留空=exe旁data/thumbs)", self.thumb_dir)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def result_cfg(self, cfg):
        cfg = dict(cfg)
        cfg.update(threshold=self.thr.value(), time_boost=self.boost.value(),
                   time_window=self.win_min.value(),
                   scan_threads=self.threads.value(),
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
        self.roots = list(self._cfg.get("last_roots") or [])
        self._worker = None
        self._paused = False
        self._face_mgr_pid = None            # 「管理此人的脸」模式
        self._avg_cache = {}                 # pid -> 平均脸 QIcon
        self._avg_sig = _AvgFaceSignals()
        self._avg_sig.ready.connect(self._on_avg_face)
        self._reclustering = False
        self._recluster_sig = _ReclusterSignals()
        self._recluster_sig.done.connect(lambda: self._recluster_done())
        self._recluster_sig.failed.connect(self._recluster_failed)

        # ---- 工具栏（桌面工具标准：图标+文字+快捷键） ----
        st = self.style()
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize() * 0.9)
        self.addToolBar(tb)
        self.act_dir = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                               "选择目录(&O)")
        self.act_dir.setShortcut(QKeySequence("Ctrl+O"))
        self.act_scan = QAction(st.standardIcon(QStyle.StandardPixmap.SP_BrowserReload),
                                "扫描(&S)")
        self.act_scan.setShortcut(QKeySequence("F5"))
        self.act_pause = QAction(st.standardIcon(QStyle.StandardPixmap.SP_MediaPause),
                                 "暂停")
        self.act_cancel = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton),
                                  "取消")
        self.act_settings = QAction(st.standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView),
                                    "设置(&P)")
        self.act_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.act_export = QAction(st.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon),
                                  "导出(&E)")
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_about = QAction(st.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
                                 "关于")
        self.act_dir.triggered.connect(self._pick_roots)
        self.act_scan.triggered.connect(self._start_scan)
        self.act_pause.triggered.connect(self._toggle_pause)
        self.act_cancel.triggered.connect(lambda: self._worker and self._worker.cancel())
        self.act_settings.triggered.connect(self._open_settings)
        self.act_export.triggered.connect(self._export_person)
        self.act_about.triggered.connect(self._about)
        for a in (self.act_dir, self.act_scan, self.act_pause, self.act_cancel):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_settings, self.act_export):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_about)

        # 状态栏右侧常驻：进度条 + 计数
        self.prog = QProgressBar()
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setFormat("扫描中 %v/%m")
        self.prog.hide()  # 空闲不显示；勿用 setMaximum(0)——那是永久忙碌动画
        self._counts_label = QLabel("")
        sb = self.statusBar()
        sb.addPermanentWidget(self.prog)
        sb.addPermanentWidget(self._counts_label)

        # 空态引导（无照片时盖在缩略图区中央）
        self._empty_hint = QLabel(
            "<div style='color:#888; font-size:14px;'>"
            "还没有照片索引<br/><br/>"
            "1. 点工具栏「选择目录」（Ctrl+O）或把照片文件夹拖进窗口<br/>"
            "2. 点「扫描」（F5）开始索引 —— 只读原照片，绝不修改<br/>"
            "3. 扫描完成后再按人整理、导出</div>")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setParent(self.grid)
        self._empty_hint.setGeometry(self.grid.rect())
        self._empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 记忆窗口状态
        geo = self._cfg.get("geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromHex(geo.encode("ascii")))
        vi = self._cfg.get("view_index")
        if isinstance(vi, int) and 0 <= vi < self.view_selector.count():
            self.view_selector.setCurrentIndex(vi)
        self.setAcceptDrops(True)
        QShortcut(QKeySequence("F2"), self).activated.connect(
            lambda: self._rename_current_person())
        QShortcut(QKeySequence("Delete"), self).activated.connect(
            lambda: self._remove_selected_faces())
        self.refresh_persons()

    # ---- 搜索：同时刷中区视图与左栏人物 ----
    def _search_changed(self, _):
        self.refresh_view()
        self.refresh_persons()

    def _thumb_dir(self):
        return self._cfg.get("thumb_dir") or (data_dir() / "thumbs")

    def _face_dir(self):
        return self._thumb_dir().parent / "faces"

    def _pick_roots(self):
        d = QFileDialog.getExistingDirectory(self, "选择照片目录")
        if d:
            self.roots = [d]; self.statusBar().showMessage(f"目录: {d}")

    def _start_scan(self):
        if not self.roots or getattr(self, "_worker", None) and self._worker.isRunning():
            return
        det = self._detector_factory()
        self._worker = ScanWorker(self.roots, self.store, det, self.clusterer,
                                  self._thumb_dir(), self._face_dir(),
                                  workers=int(self._cfg.get("scan_threads", 4)))
        self._worker.progress.connect(self.on_scan_progress)
        self._worker.done.connect(self.on_scan_done)
        self._worker.failed.connect(self.on_scan_failed)
        self.act_scan.setEnabled(False)
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
        self.act_scan.setEnabled(True)
        self._paused = False; self.act_pause.setText("暂停")
        self._avg_cache.clear()  # 新扫描可能产出新裁剪
        self.refresh_view(); self.refresh_persons()
        if getattr(self, "_worker", None) and self._worker._cancel.is_set():
            self.statusBar().showMessage("已取消（已处理部分保留，下次续扫）")
        else:
            self.statusBar().showMessage(
                f"扫描完成 · 共 {self.store.count_photos()} 张")

    def on_scan_failed(self, msg):
        self.prog.hide()
        self.act_scan.setEnabled(True)
        self.statusBar().showMessage(f"扫描失败: {msg}")

    def _toggle_pause(self):
        if not self._worker:
            return
        if self._paused:
            self._worker.resume(); self._paused = False; self.act_pause.setText("暂停")
            self.statusBar().showMessage("继续扫描")
        else:
            self._worker.pause(); self._paused = True; self.act_pause.setText("继续")
            self.statusBar().showMessage("已暂停")

    def _open_settings(self):
        dlg = SettingsDialog(self._cfg, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._apply_settings(self._cfg, dlg.result_cfg(self._cfg))

    def _apply_settings(self, old, new):
        self._cfg = new
        save_settings(self.settings_path, self._cfg)
        if self._clusterer_factory:
            self.clusterer = self._clusterer_factory()  # 工厂应读最新 settings
        if need_recluster(old, new) and self.clusterer is not None:
            self._start_recluster()
            return
        self.refresh_view(); self.refresh_persons()

    def _start_recluster(self):
        if self._reclustering:
            return
        self._reclustering = True
        self.act_scan.setEnabled(False)
        self.act_export.setEnabled(False)
        self.statusBar().showMessage("正在按新阈值重排人物…")
        QThreadPool.globalInstance().start(
            _ReclusterRunner(self.clusterer, self._recluster_sig))

    def _recluster_done(self):
        self._reclustering = False
        self.act_scan.setEnabled(True)
        self.act_export.setEnabled(True)
        self._avg_cache.clear()
        self.refresh_view(); self.refresh_persons()
        self.statusBar().showMessage("已按新阈值重排完成")

    def _recluster_failed(self, msg):
        self._reclustering = False
        self.act_scan.setEnabled(True)
        self.act_export.setEnabled(True)
        self.statusBar().showMessage(f"重排失败: {msg}")

    def _about(self):
        from app.version import APP_VERSION
        QMessageBox.about(
            self, "关于 FaceSorter",
            f"<b>FaceSorter {APP_VERSION}</b><br/>"
            "完全离线的照片人物分类工具：只索引，绝不修改原照片。<br/><br/>"
            "双击看原图 · 拖拽脸部到左侧人物 · 右键更多操作<br/>"
            "Ctrl+O 选目录 · F5 扫描 · Ctrl+E 导出 · Ctrl+, 设置")

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
        self.act_export.setEnabled(False)
        self.statusBar().showMessage("正在导出…")
        self._export_runner = _ExportRunner(self.store, pid, dest)
        self._export_runner.signals.done.connect(self._export_done)
        self._export_runner.signals.failed.connect(self._export_failed)
        QThreadPool.globalInstance().start(self._export_runner)

    def _export_done(self, n, label):
        self.act_export.setEnabled(True)
        self.statusBar().showMessage("就绪")
        QMessageBox.information(self, "导出完成",
                                f"已为「{label}」导出 {n} 张照片副本（原文件未动）。")

    def _export_failed(self, msg):
        self.act_export.setEnabled(True)
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
            icon = self._avg_cache.get(d["id"])
            if icon is not None:
                it.setIcon(icon)
            self.persons_list.addItem(it)
            if icon is None:
                QThreadPool.globalInstance().start(_AvgFaceRunner(
                    self.store, d["id"], self._face_dir(), self._avg_sig))
        for label in SPECIAL_VIEWS:
            it = QListWidgetItem(label)
            self.persons_list.addItem(it)

    def _on_avg_face(self, pid, qimg):
        if qimg is None:
            return
        icon = QIcon(QPixmap.fromImage(qimg))
        self._avg_cache[pid] = icon
        for i in range(self.persons_list.count()):
            it = self.persons_list.item(i)
            if it.data(PID_ROLE) == pid:
                it.setIcon(icon)
                break

    def _person_clicked(self, item):
        self._face_mgr_pid = None  # 点击=默认整照片堆视图
        self.refresh_view()

    def _manage_faces(self, pid):
        self._face_mgr_pid = pid
        for i in range(self.persons_list.count()):
            it = self.persons_list.item(i)
            if it.data(PID_ROLE) == pid:
                self.persons_list.setCurrentItem(it)
                break
        self.refresh_view()
        self.statusBar().showMessage("脸部管理：拖拽到左侧人物或右键调整")

    def _current_person_view(self):
        """返回 (mode, person_id)：mode ∈ photo_person/face_person/face_unassigned/face_all"""
        item = self.persons_list.currentItem()
        if item is None:
            return None, None
        if item.text() == "全部脸":
            return "face_all", None
        if item.text() == "未归类脸":
            return "face_unassigned", None
        pid = item.data(PID_ROLE)
        if not pid:
            return None, None
        if self._face_mgr_pid == pid:
            return "face_person", pid
        return "photo_person", pid

    def _face_rows_for_current(self):
        mode, pid = self._current_person_view()
        if mode == "face_person":
            return self.store.faces_of_person(pid)
        if mode == "face_unassigned":
            return self.store.faces_without_person()
        if mode == "face_all":
            return self.store.all_faces()
        return []

    # ---- 中区 ----
    def refresh_view(self):
        self._refresh_rows()
        self._update_counts()
        self._update_empty_hint()

    def _update_counts(self):
        n_photos = self.store.count_photos()
        n_persons = len(self.store.persons_with_counts())
        self._counts_label.setText(f"{n_photos} 照片 · {n_persons} 人")

    def _update_empty_hint(self):
        self._empty_hint.setGeometry(self.grid.rect())
        self._empty_hint.setVisible(self.grid_model.rowCount() == 0)

    def _refresh_rows(self):
        mode, pid = self._current_person_view()
        if mode == "photo_person":
            # 人物照片堆：整张照片（合照会出现在每个人的堆里）
            self.grid_model.clear()
            rows = self.store.photos_of_person(pid)
            flt = self.search_box.text().strip().lower()
            shown = 0
            for r in rows:
                if flt and flt not in r["path"].lower():
                    continue
                shown += 1
                item = QStandardItem(r["path"].rsplit("/", 1)[-1])
                item.setData(r["path"], Qt.ItemDataRole.UserRole)
                item.setEditable(False)
                self.grid_model.appendRow(item)
                if r["thumb_path"]:
                    self.thumbs.thumb_for(r["thumb_path"],
                        lambda icon, it=item: it.setIcon(icon))
            self.statusBar().showMessage(f"{shown} 张照片")
            return
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
        # 优先扫描期对齐裁剪（原图生成，清晰）；旧库无裁剪时退回缩略图裁剪
        crop_path = self._face_dir() / f"{r['id']}.jpg"
        if crop_path.exists():
            self.thumbs.thumb_for(str(crop_path),
                                  lambda icon, it=item: it.setIcon(icon))
            return
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
            self._avg_cache.clear()
            self.refresh_persons(); self.refresh_view()

    def _toggle_pin(self, pid):
        row = [r for r in self.store.persons_with_counts() if r["id"] == pid][0]
        self.store.set_pinned(pid, not row["pinned"])
        self.refresh_persons()

    def _move_faces(self, face_ids, dst_pid):
        for fid in face_ids:
            self.store.set_face_person(fid, dst_pid, "manual")
        if self.clusterer: self.clusterer.invalidate(dst_pid)
        self._avg_cache.clear()
        self.refresh_persons(); self.refresh_view()

    def _new_person_from_faces(self, face_ids):
        pid = self.store.create_person()
        self._move_faces(face_ids, pid)

    def _remove_from_person(self, face_ids):
        for fid in face_ids:
            self.store.set_face_person(fid, None, "manual")
        self._avg_cache.clear()
        self.refresh_persons(); self.refresh_view()

    def _open_item(self, index):
        """双击：全部脸→进该人物照片堆；其余→内置浏览器（当前人物绿框）。"""
        data = index.data(Qt.ItemDataRole.UserRole)
        mode, pid = self._current_person_view()
        if mode == "face_all" and isinstance(data, tuple) and data[0] == "face":
            row = self.store.face_by_id(data[1])
            if row is not None and row["person_id"]:
                for i in range(self.persons_list.count()):
                    if self.persons_list.item(i).data(PID_ROLE) == row["person_id"]:
                        self.persons_list.setCurrentRow(i)
                        self._person_clicked(self.persons_list.item(i))
                        return
        cur = data
        if isinstance(cur, tuple) and cur[0] == "face":
            photo = self.store.photo_of_face(cur[1])
            cur = photo["path"] if photo else None
        paths = self._grid_paths()
        if cur and cur in paths:
            PhotoViewer(paths, paths.index(cur), self.store,
                        highlight_pid=pid, parent=self).exec()

    def _grid_paths(self):
        """当前格子的照片路径序列：脸格子解析成所属照片，按序去重。"""
        paths = []
        for r in range(self.grid_model.rowCount()):
            d = self.grid_model.item(r).data(Qt.ItemDataRole.UserRole)
            if isinstance(d, tuple) and d[0] == "face":
                photo = self.store.photo_of_face(d[1])
                d = photo["path"] if photo else None
            if d and d not in paths:
                paths.append(d)
        return paths

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
    # ---- 窗口级：记忆状态 / 拖放目录 / 快捷键 ----
    def closeEvent(self, e):
        self._cfg["geometry"] = bytes(self.saveGeometry()).hex()
        self._cfg["last_roots"] = list(self.roots)
        self._cfg["view_index"] = self.view_selector.currentIndex()
        save_settings(self.settings_path, self._cfg)
        super().closeEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._empty_hint.setGeometry(self.grid.rect())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        from pathlib import Path
        dirs = [u.toLocalFile() for u in e.mimeData().urls()
                if u.isLocalFile() and Path(u.toLocalFile()).is_dir()]
        if dirs:
            self.roots = dirs
            self.statusBar().showMessage(f"目录: {'、'.join(dirs)}（按 F5 开始扫描）")
            e.acceptProposedAction()

    def _rename_current_person(self):
        mode, pid = self._current_person_view()
        if mode in ("photo_person", "face_person"):
            self._rename_person(pid)

    def _remove_selected_faces(self):
        face_ids = [d.data(Qt.ItemDataRole.UserRole)[1]
                    for d in self.grid.selectedIndexes()
                    if isinstance(d.data(Qt.ItemDataRole.UserRole), tuple)]
        if face_ids:
            self._remove_from_person(face_ids)
