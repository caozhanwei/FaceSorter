# -*- coding: utf-8 -*-
"""人物视图脸部图标异步加载端到端（回归：setIcon(QPixmap) 原生崩溃）。"""
import numpy as np
import cv2

import pytest
from PyQt6.QtGui import QImage

from app.store import Store
from app.ui.main_window import MainWindow, PID_ROLE


@pytest.fixture
def win_person(tmp_path, qapp):
    store = Store(tmp_path / "t.db")
    thumb = tmp_path / "thumb1.jpg"
    img = (np.random.rand(300, 400, 3) * 255).astype("uint8")
    cv2.imencode(".jpg", img)[1].tofile(str(thumb))
    pid = store.upsert_photo("X:/photos/a.jpg", 1.0, 1)
    store.update_photo_meta(pid, width=400, height=300,
                            thumb_path=thumb.as_posix(), has_face=1)
    fid = store.add_face(pid, 40, 30, 120, 120, 1.0, None)
    person = store.create_person("测试人")
    store.set_face_person(fid, person, "auto")
    w = MainWindow(store, lambda: None, tmp_path / "s.json")
    for i in range(w.persons_list.count()):
        it = w.persons_list.item(i)
        if it.data(PID_ROLE):
            w.persons_list.setCurrentItem(it)
            break
    w.refresh_view()
    return w


def test_person_face_icon_loaded(win_person, qtbot):
    item = win_person.grid_model.item(0)
    assert item is not None
    # 回调里 setIcon 必须传 QIcon；传 QPixmap 在本机 PyQt6 上原生段错误
    qtbot.waitUntil(lambda: not item.icon().isNull(), timeout=5000)
