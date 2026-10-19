import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from app.paths import data_dir
from app.settings import load_settings
from app.store import Store, StoreCorrupted
from app.detector import FaceDetector
from app.cluster import Clusterer
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
    cfg = load_settings(d / "settings.json")
    app = QApplication(sys.argv)
    store = _open_store(d / "facesorter.db")
    win = MainWindow(store, FaceDetector, d / "settings.json",
                     lambda: Clusterer(store, threshold=cfg["threshold"],
                                       time_boost=cfg["time_boost"],
                                       time_window_min=cfg["time_window"]))
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
