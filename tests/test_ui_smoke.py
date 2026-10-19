import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QMainWindow
from app.store import Store
from app.ui.main_window import MainWindow
from app.ui.thumb_provider import ThumbProvider

@pytest.fixture
def win(tmp_path, qapp):
    store = Store(tmp_path / "t.db")
    pid = store.upsert_photo("a.jpg", 1.0, 1)
    store.update_photo_meta(pid, thumb_path=None, has_face=0)
    return MainWindow(store, lambda: None, tmp_path / "s.json")

def test_window_constructs_and_lists_all(win):
    win.refresh_view()
    assert win.view_selector.currentText() == "全部照片"
    assert win.grid_model.rowCount() == 1

def test_view_switch_no_faces(win):
    win.view_selector.setCurrentText("无人的")
    win.refresh_view()
    assert win.grid_model.rowCount() == 1
    win.view_selector.setCurrentText("有人的")
    win.refresh_view()
    assert win.grid_model.rowCount() == 0

def test_thumb_provider_async(tmp_path, qapp, qtbot):
    img = QImage(300, 200, QImage.Format.Format_RGB32)
    img.fill(0)
    p = tmp_path / "t.png"
    img.save(str(p))
    tp = ThumbProvider()
    got = []
    tp.thumb_for(str(p), lambda icon: got.append(icon))
    qtbot.waitUntil(lambda: len(got) == 1, timeout=5000)
    assert not got[0].isNull()
