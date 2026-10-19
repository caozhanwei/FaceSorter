# -*- coding: utf-8 -*-
"""内置图片浏览器：双击看大图，当前人物的脸绿框、其他人灰框；
左右翻页、缩放平移为桌面浏览器基本盘。"""
import numpy as np
import cv2
import pytest

from app.store import Store
from app.ui.photo_viewer import PhotoViewer, box_specs


def _photo(path, h=150, w=200, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    img[:] = (rng.random((h, w, 1)) * 30 + 113).astype(np.uint8)
    cv2.imencode(".jpg", img)[1].tofile(str(path))
    return path


GREEN = (0, 200, 0)
GRAY = (170, 170, 170)


# ---- 纯逻辑：框选归属 ----

def _face(x, y, w, h, person_id, name=None):
    return {"x": x, "y": y, "w": w, "h": h,
            "person_id": person_id, "name": name}


def test_box_specs_highlights_current_person():
    faces = [_face(10, 10, 50, 50, 1), _face(100, 80, 40, 40, 2)]
    out = box_specs(faces, highlight_pid=1)
    assert out[0] == (10, 10, 50, 50, True)   # 当前人物 → 绿
    assert out[1] == (100, 80, 40, 40, False)  # 其他人 → 灰


def test_box_specs_without_person_all_green():
    # 非人物视图打开：所有人脸都框出（全绿）
    faces = [_face(0, 0, 30, 30, 7), _face(60, 60, 30, 30, 9)]
    assert all(spec[4] for spec in box_specs(faces, highlight_pid=None))


# ---- Store 取脸 ----

def test_faces_of_photo_with_names(tmp_path):
    store = Store(tmp_path / "t.db")
    pid = store.upsert_photo("a.jpg", 1.0, 10)
    pa = store.create_person("张三")
    f1 = store.add_face(pid, 10, 10, 50, 50, 100.0, None)
    f2 = store.add_face(pid, 100, 80, 40, 40, 100.0, None)
    store.set_face_person(f1, pa, "auto")
    rows = store.faces_of_photo(pid)
    assert {(r["x"], r["person_id"], r["name"]) for r in rows} == {
        (10, pa, "张三"), (100, None, None)}
    assert store.photo_by_path("a.jpg")["id"] == pid


# ---- 渲染 + 翻页（像素断言：无视觉也能验证框真的画出来） ----

@pytest.fixture
def viewer_env(tmp_path, qapp):
    store = Store(tmp_path / "t.db")
    pa = store.create_person("张三")
    p1 = _photo(tmp_path / "one.jpg", seed=1)
    pid1 = store.upsert_photo(p1.as_posix(), 1.0, 10)
    store.update_photo_meta(pid1, width=200, height=150)
    f_a = store.add_face(pid1, 10, 10, 50, 50, 100.0, None)
    f_b = store.add_face(pid1, 100, 80, 40, 40, 100.0, None)
    store.set_face_person(f_a, pa, "auto")
    p2 = _photo(tmp_path / "two.jpg", seed=2)
    pid2 = store.upsert_photo(p2.as_posix(), 1.0, 10)
    store.update_photo_meta(pid2, width=200, height=150)
    f_c = store.add_face(pid2, 20, 20, 60, 60, 100.0, None)
    store.set_face_person(f_c, pa, "auto")
    return store, pa, [p1.as_posix(), p2.as_posix()]


def test_viewer_draws_green_for_current_gray_for_others(viewer_env, qapp):
    store, pa, paths = viewer_env
    v = PhotoViewer(paths, 0, store, highlight_pid=pa)
    v.resize(400, 300)
    img = v.canvas.grab().toImage()
    greens = grays = 0
    for x in range(img.width()):
        for y in range(img.height()):
            c = img.pixel(x, y)
            rgb = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
            if rgb == GREEN:
                greens += 1
            elif rgb == GRAY:
                grays += 1
    assert greens > 0, "当前人物的脸必须有绿框"
    assert grays > 0, "其他人的脸必须有灰框"


def test_viewer_paging_and_recompute(viewer_env, qapp):
    store, pa, paths = viewer_env
    v = PhotoViewer(paths, 0, store, highlight_pid=pa)
    assert v.current_path() == paths[0]
    v.next()
    assert v.current_path() == paths[1]
    v.next()
    assert v.current_path() == paths[1], "末页不再前进"
    v.prev(); v.prev()
    assert v.current_path() == paths[0], "首页不再后退"


def test_viewer_zoom_fit(viewer_env, qapp):
    store, pa, paths = viewer_env
    v = PhotoViewer(paths, 0, store)
    v.resize(400, 300)
    s_fit = v.canvas.scale
    v.canvas.zoom_at(2.0, 10, 10)
    assert v.canvas.scale > s_fit
    v.canvas.fit()
    assert abs(v.canvas.scale - s_fit) < 1e-9
