# -*- coding: utf-8 -*-
"""使用帮助：help_text 纯函数内容覆盖 + 主窗口 F1 入口接线。"""
import pytest

from app.ui.help import help_text, HelpDialog


def test_help_text_covers_workflow():
    t = help_text()
    for kw in ("扫描", "阈值", "重排", "命名", "拖拽", "未归类脸", "全部脸",
               "导出", "绿框", "灰框", "多线程"):
        assert kw in t, f"帮助缺关键内容：{kw}"


def test_help_text_covers_shortcuts():
    t = help_text()
    for kw in ("Ctrl+O", "F5", "Ctrl+,", "Ctrl+E", "F1",
               "←/→", "+", "-", "F", "Esc", "双击"):
        assert kw in t, f"帮助缺快捷键：{kw}"


def test_help_dialog_renders_text(qapp):
    d = HelpDialog()
    assert help_text() in d._browser.toPlainText()


@pytest.fixture
def env(tmp_path, qapp):
    from app.store import Store
    from app.ui.main_window import MainWindow
    store = Store(tmp_path / "t.db")
    win = MainWindow(store, lambda: None, tmp_path / "s.json")
    return win, store


def test_main_window_help_action(env, monkeypatch):
    from PyQt6.QtGui import QKeySequence
    import app.ui.main_window as mw
    opened = []

    class _Fake:
        def __init__(self, parent=None):
            opened.append(1)
        def exec(self):
            return 0
    monkeypatch.setattr(mw, "HelpDialog", _Fake)
    win, store = env
    assert win.act_help is not None
    assert win.act_help.shortcut().toString() in (
        "F1", QKeySequence("F1").toString())
    win.act_help.trigger()
    assert opened, "触发帮助动作必须打开帮助对话框"
