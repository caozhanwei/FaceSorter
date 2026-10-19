import pytest
from app.store import Store
from app.ui.main_window import MainWindow

@pytest.fixture
def env(tmp_path, qapp):
    store = Store(tmp_path / "t.db")
    pid = store.upsert_photo("a.jpg", 1.0, 1)
    store.update_photo_meta(pid, has_face=1, thumb_path=None)
    f1 = store.add_face(pid, 0, 0, 80, 80, 100.0, b"\x01\x02")
    p = store.create_person("张三")
    store.set_face_person(f1, p, "manual")
    win = MainWindow(store, lambda: None, tmp_path / "s.json")
    return win, store, p, f1

def test_person_list_shows_named(env):
    win, store, p, f1 = env
    win.refresh_persons()
    labels = [win.persons_list.item(i).text() for i in range(win.persons_list.count())]
    assert any("张三" in l for l in labels)

def test_rename_via_api_locks(env):
    win, store, p, f1 = env
    assert hasattr(win, "_rename_person")  # 对话框交互走手动验收,这里只测 store 路径
    store.rename_person(p, "李四")
    assert [r for r in store.persons_with_counts() if r["id"] == p][0]["locked"] == 1

def test_merge_via_menu_logic(env):
    win, store, p, f1 = env
    p2 = store.create_person()
    store.merge_persons(p2, p)
    assert store.persons_with_counts()[0]["cnt"] == 1


# ---- 人物视图 = 整照片堆（合照多人共享）；脸部管理为独立模式 ----

def test_person_view_shows_whole_photos(env):
    from PyQt6.QtCore import Qt
    win, store, p, f1 = env
    win.persons_list.setCurrentRow(0)  # 张三
    win.refresh_view()
    assert win.grid_model.rowCount() == 1
    assert win.grid_model.item(0).data(Qt.ItemDataRole.UserRole) == "a.jpg"

def test_person_manage_faces_mode(env):
    from PyQt6.QtCore import Qt
    win, store, p, f1 = env
    win.persons_list.setCurrentRow(0)  # 张三
    win._face_mgr_pid = p
    win.refresh_view()
    assert win.grid_model.item(0).data(Qt.ItemDataRole.UserRole) == ("face", f1)


# ---- 内置图片浏览器接线：双击按当前视图传人物高亮 ----

def test_open_item_passes_person_highlight(env, monkeypatch):
    from PyQt6.QtCore import Qt
    import app.ui.main_window as mw
    captured = {}
    class _FakeViewer:
        def __init__(self, paths, i, store, highlight_pid=None, parent=None):
            captured.update(paths=paths, i=i, pid=highlight_pid)
        def exec(self):
            return 0
    monkeypatch.setattr(mw, "PhotoViewer", _FakeViewer)
    win, store, p, f1 = env
    pid2 = store.upsert_photo("b.jpg", 1.0, 1)
    f2 = store.add_face(pid2, 5, 5, 30, 30, 100.0, b"\x01")
    store.set_face_person(f2, p, "manual")
    win.persons_list.setCurrentRow(0)  # 张三照片堆
    win.refresh_view()
    win._open_item(win.grid_model.index(0, 0))
    assert captured["pid"] == p, "人物堆里打开须高亮该人物"
    assert captured["paths"] == ["a.jpg", "b.jpg"]
    assert captured["i"] == 0


def test_open_item_from_face_view_resolves_photo(env, monkeypatch):
    from PyQt6.QtCore import Qt
    import app.ui.main_window as mw
    captured = {}
    class _FakeViewer:
        def __init__(self, paths, i, store, highlight_pid=None, parent=None):
            captured.update(paths=paths, i=i, pid=highlight_pid)
        def exec(self):
            return 0
    monkeypatch.setattr(mw, "PhotoViewer", _FakeViewer)
    win, store, p, f1 = env
    win.persons_list.setCurrentRow(0)
    win._face_mgr_pid = p  # 脸部管理模式：格子是脸
    win.refresh_view()
    win._open_item(win.grid_model.index(0, 0))
    assert captured["paths"] == ["a.jpg"]
    assert captured["pid"] == p


# ---- 全部脸视图：清晰脸排前，双击进人物照片堆 ----

def _select_special(win, label):
    for i in range(win.persons_list.count()):
        if win.persons_list.item(i).text() == label:
            win.persons_list.setCurrentRow(i)
            win.refresh_view()
            return True
    return False


def test_all_faces_view_quality_order(env):
    from PyQt6.QtCore import Qt
    win, store, p, f1 = env
    ph2 = store.upsert_photo("b.jpg", 1.0, 1)
    store.update_photo_meta(ph2, has_face=1)
    f2 = store.add_face(ph2, 5, 5, 30, 30, 200.0, b"\x03")
    assert _select_special(win, "全部脸")
    roles = [win.grid_model.item(r).data(Qt.ItemDataRole.UserRole)
             for r in range(win.grid_model.rowCount())]
    assert roles == [("face", f2), ("face", f1)], "清晰脸排前"


def test_all_faces_double_click_jumps_to_person_pile(env):
    from PyQt6.QtCore import Qt
    win, store, p, f1 = env
    assert _select_special(win, "全部脸")
    win._open_item(win.grid_model.index(0, 0))  # f1 属张三
    mode, pid = win._current_person_view()
    assert (mode, pid) == ("photo_person", p)
    assert win.grid_model.item(0).data(Qt.ItemDataRole.UserRole) == "a.jpg"


def test_all_faces_unassigned_opens_viewer(env, monkeypatch):
    import app.ui.main_window as mw
    captured = {}
    class _FakeViewer:
        def __init__(self, paths, i, store, highlight_pid=None, parent=None):
            captured.update(paths=paths, i=i, pid=highlight_pid)
        def exec(self):
            return 0
    monkeypatch.setattr(mw, "PhotoViewer", _FakeViewer)
    win, store, p, f1 = env
    store.set_face_person(f1, None, "manual")  # 变未归类
    win.refresh_persons()
    assert _select_special(win, "全部脸")
    win._open_item(win.grid_model.index(0, 0))
    assert captured["paths"] == ["a.jpg"]
    assert captured["pid"] is None, "未归类脸无高亮人物：全部绿框"
