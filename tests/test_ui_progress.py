# -*- coding: utf-8 -*-
"""扫描进度反馈：启动不忙碌、扫描中有数字、结束/取消/暂停有文字说明。

背景：进度条构造时 setMaximum(0)（不定长忙碌动画）导致一打开就永远动，
且扫描结束不复位、全程无文字 —— 用户"不知道在干啥"。
"""
import types

import pytest

from app.store import Store
from app.ui.main_window import MainWindow


@pytest.fixture
def win(tmp_path, qapp):
    store = Store(tmp_path / "t.db")
    pid = store.upsert_photo("a.jpg", 1.0, 1)
    store.update_photo_meta(pid, thumb_path=None, has_face=0)
    return MainWindow(store, lambda: None, tmp_path / "s.json")


def test_startup_bar_not_busy(win):
    # setMaximum(0)=不定长忙碌动画；空闲时禁止
    assert win.prog.maximum() != 0
    assert win.prog.isHidden()  # 空闲时不显示进度条


def test_progress_updates_bar_and_status(win):
    win.on_scan_progress(3, 10)
    assert win.prog.maximum() == 10
    assert win.prog.value() == 3
    assert not win.prog.isHidden()
    assert "3/10" in win.statusBar().currentMessage()


def test_done_resets_bar_and_reports(win):
    win.on_scan_progress(5, 10)
    win.on_scan_done()
    assert win.prog.isHidden()
    assert win.prog.maximum() != 0
    assert "完成" in win.statusBar().currentMessage()


def test_cancelled_scan_reports_cancelled(win):
    win.on_scan_progress(5, 10)
    win._worker = types.SimpleNamespace(
        _cancel=types.SimpleNamespace(is_set=lambda: True))
    win.on_scan_done()
    assert "取消" in win.statusBar().currentMessage()


def test_failed_hides_bar_and_reports(win):
    win.on_scan_progress(5, 10)
    win.on_scan_failed("磁盘满")
    assert win.prog.isHidden()
    assert "磁盘满" in win.statusBar().currentMessage()


def test_pause_resume_updates_status(win):
    win._worker = types.SimpleNamespace(pause=lambda: None, resume=lambda: None)
    win._toggle_pause()
    assert "暂停" in win.statusBar().currentMessage()
    assert win.act_pause.text() == "继续"
    win._toggle_pause()
    assert win.act_pause.text() == "暂停"
