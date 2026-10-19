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
