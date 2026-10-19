import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QFont
from app.paths import data_dir
from app.settings import load_settings
from app.store import Store, StoreCorrupted
from app.cluster import Clusterer
from app.variant import resolve_variant, build_detector_factory, ENGINE_LABELS
from app.ui.main_window import MainWindow

def _open_store(db):
    try:
        return Store(db)
    except StoreCorrupted:
        ans = QMessageBox.question(
            None, "数据库损坏",
            "索引数据库损坏，是否重建？（只丢索引与缩略图记录，原照片不受影响）")
        if ans != QMessageBox.StandardButton.Yes:
            raise
        db.rename(db.with_suffix(".corrupt.db"))
        return Store(db)

def main():
    d = data_dir()
    eng = resolve_variant()
    load_settings(d / "settings.json", engine=eng if eng != "none" else None)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 9))
    store = _open_store(d / "facesorter.db")

    def _clusterer_factory():
        # 每次重建都读最新设置：改完阈值立即生效（含重排）
        cfg = load_settings(d / "settings.json")
        return Clusterer(store, threshold=cfg["threshold"],
                         time_boost=cfg["time_boost"],
                         time_window_min=cfg["time_window"])

    win = MainWindow(store, build_detector_factory(eng), d / "settings.json",
                     _clusterer_factory, engine_label=ENGINE_LABELS[eng])
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
