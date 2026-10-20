# -*- coding: utf-8 -*-
"""内置图片浏览器：看大图 + 自动框出人脸。
当前人物的脸绿框、其他人灰框；←/→ 翻页、滚轮/按钮缩放、拖拽平移、F 适应窗口。
（双击照片不再调系统程序，保持在应用内浏览。）
"""
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QAction, QColor, QImage, QImageReader, QPainter,
                         QPen)
from PyQt6.QtWidgets import (QDialog, QLabel, QStyle, QToolBar,
                             QVBoxLayout, QWidget)

GREEN = QColor(0, 200, 0)
GRAY = QColor(170, 170, 170)


def box_specs(faces, highlight_pid):
    """[(x, y, w, h, is_current)]。highlight_pid=None → 全部脸视为当前（全绿）。"""
    out = []
    for f in faces:
        cur = highlight_pid is None or f["person_id"] == highlight_pid
        out.append((f["x"], f["y"], f["w"], f["h"], cur))
    return out


class _Canvas(QWidget):
    """缩放/平移画布：图片 + 恒定线宽的人脸框（cosmetic pen 不随缩放变粗）。"""

    def __init__(self):
        super().__init__()
        self._img = None
        self._boxes = []
        self.scale = 1.0
        self._pan = QPointF(0, 0)
        self._drag_from = None
        self._drag_pan0 = QPointF(0, 0)
        self.setMinimumSize(200, 150)
        self.setMouseTracking(True)

    def set_image(self, img, boxes):
        self._img = img if img is not None else QImage()
        self._boxes = boxes
        self.fit()

    def fit(self):
        if self._img is None or self._img.isNull():
            self.update()
            return
        self.scale = min(self.width() / self._img.width(),
                         self.height() / self._img.height())
        self._center()

    def _center(self):
        if self._img is None or self._img.isNull():
            return
        iw = self._img.width() * self.scale
        ih = self._img.height() * self.scale
        self._pan = QPointF((self.width() - iw) / 2, (self.height() - ih) / 2)
        self.update()

    def zoom_at(self, factor, cx, cy):
        new = max(0.02, min(24.0, self.scale * factor))
        f = new / self.scale
        pos = QPointF(cx, cy)
        self._pan = pos - (pos - self._pan) * f
        self.scale = new
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._img is None or self._img.isNull():
            return
        p.translate(self._pan)
        p.scale(self.scale, self.scale)
        p.drawImage(QPointF(0, 0), self._img)
        for (x, y, w, h, cur) in self._boxes:
            pen = QPen(GREEN if cur else GRAY)
            pen.setCosmetic(True)  # 线宽用设备像素：放大缩小都保持清晰可见
            pen.setWidthF(3.0 if cur else 2.0)
            p.setPen(pen)
            p.drawRect(QRectF(x, y, w, h))

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        self.zoom_at(1.25 if d > 0 else 0.8,
                     e.position().x(), e.position().y())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_from = e.position()
            self._drag_pan0 = QPointF(self._pan)

    def mouseMoveEvent(self, e):
        if self._drag_from is not None:
            self._pan = self._drag_pan0 + (e.position() - self._drag_from)
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag_from = None

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._center()


class PhotoViewer(QDialog):
    """paths 顺序翻页；highlight_pid 决定绿框归属（None=全部绿）。"""

    def __init__(self, paths, index, store, highlight_pid=None, parent=None):
        super().__init__(parent)
        self.paths = [p for p in (paths or []) if p]
        self._i = max(0, min(index, len(self.paths) - 1)) if self.paths else -1
        self.store = store
        self.highlight_pid = highlight_pid
        self.setWindowTitle("图片浏览器")
        self.resize(1000, 700)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        st = self.style()
        self.canvas = _Canvas()
        self._info = QLabel()
        bar = QToolBar()
        bar.setMovable(False)
        self.act_prev = QAction(st.standardIcon(
            QStyle.StandardPixmap.SP_ArrowBack), "上一张(←)")
        self.act_next = QAction(st.standardIcon(
            QStyle.StandardPixmap.SP_ArrowForward), "下一张(→)")
        self.act_zoom_in = QAction("放大(+)")
        self.act_zoom_out = QAction("缩小(-)")
        self.act_fit = QAction("适应窗口(F)")
        self.act_prev.triggered.connect(self.prev)
        self.act_next.triggered.connect(self.next)
        self.act_zoom_in.triggered.connect(
            lambda: self.canvas.zoom_at(1.25, self.canvas.width() / 2,
                                        self.canvas.height() / 2))
        self.act_zoom_out.triggered.connect(
            lambda: self.canvas.zoom_at(0.8, self.canvas.width() / 2,
                                        self.canvas.height() / 2))
        self.act_fit.triggered.connect(self.canvas.fit)
        for a in (self.act_prev, self.act_next, self.act_zoom_in,
                  self.act_zoom_out, self.act_fit):
            bar.addAction(a)
        lay.addWidget(bar)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self._info)
        self._show_current()

    # ---- 翻页 ----
    def next(self):
        if self._i < len(self.paths) - 1:
            self._i += 1
            self._show_current()

    def prev(self):
        if self._i > 0:
            self._i -= 1
            self._show_current()

    def current_path(self):
        return self.paths[self._i] if 0 <= self._i < len(self.paths) else None

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Left:
            self.prev()
        elif k == Qt.Key.Key_Right:
            self.next()
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.canvas.zoom_at(1.25, self.canvas.width() / 2,
                                self.canvas.height() / 2)
        elif k == Qt.Key.Key_Minus:
            self.canvas.zoom_at(0.8, self.canvas.width() / 2,
                                self.canvas.height() / 2)
        elif k == Qt.Key.Key_F:
            self.canvas.fit()
        else:
            super().keyPressEvent(e)  # Esc 等默认行为（关闭）

    def _show_current(self):
        path = self.current_path()
        if path is None:
            self._info.setText("没有照片")
            return
        # QImage 不读 EXIF；扫描端 cv2.imdecode 已按 EXIF 转正记坐标，
        # 这里必须同样转正，否则图倒框歪
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        img = reader.read()
        boxes = []
        photo = self.store.photo_by_path(path)
        if photo is not None:
            faces = self.store.faces_of_photo(photo["id"])
            boxes = box_specs(faces, self.highlight_pid)
        self.canvas.set_image(img, boxes)
        n_cur = sum(1 for b in boxes if b[4])
        if self.highlight_pid is not None:
            tip = f"当前人物 {n_cur}/{len(boxes)} 张脸（绿）"
        else:
            tip = f"共 {len(boxes)} 张脸"
        self._info.setText(f"{self._i + 1}/{len(self.paths)} · {tip}"
                           f" · 绿=当前人物 灰=其他人")
        self.setWindowTitle(
            f"{path.rsplit('/', 1)[-1]} ({self._i + 1}/{len(self.paths)})"
            " - FaceSorter")
