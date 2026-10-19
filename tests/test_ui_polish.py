# -*- coding: utf-8 -*-
"""桌面工具级 UI：阈值变更触发重排、窗口状态记忆、快捷键、空态引导、状态栏计数。"""
import json
import pytest

from app.store import Store
from app.settings import need_recluster
from app.cluster import Clusterer
from app.ui.main_window import MainWindow


def _win(tmp_path, clusterer_factory=None):
    store = Store(tmp_path / "t.db")
    factory = clusterer_factory or (lambda: Clusterer(
        store, threshold=0.36, time_boost=1.1, time_window_min=30))
    return MainWindow(store, lambda: None, tmp_path / "s.json", factory)


def test_need_recluster_keys():
    old = {"threshold": 0.36, "time_boost": 1.1, "time_window": 30, "thumb_dir": None}
    assert not need_recluster(old, dict(old))
    assert not need_recluster(old, dict(old, thumb_dir="X:/t"))  # 缓存目录改动不重排
    assert need_recluster(old, dict(old, threshold=0.45))
    assert need_recluster(old, dict(old, time_boost=1.2))
    assert need_recluster(old, dict(old, time_window=60))


def test_apply_settings_reclusters_with_new_threshold(tmp_path, qapp, qtbot):
    # 用户诉求：改完阈值自动按新值重排并刷新（不是只影响以后扫描）
    store = Store(tmp_path / "t.db")
    calls = []

    def factory():
        s = tmp_path / "s.json"
        thr = 0.36
        if s.exists():
            thr = json.loads(s.read_text("utf-8")).get("threshold", 0.36)

        class _Cl:
            threshold = thr
            def recluster_auto(self):
                calls.append(self.threshold)
            def invalidate(self, pid):
                pass
        return _Cl()

    w = MainWindow(store, lambda: None, tmp_path / "s.json", factory)
    old = {"threshold": 0.36, "time_boost": 1.1, "time_window": 30}
    new = dict(old, threshold=0.45)
    w._apply_settings(old, new)
    qtbot.waitUntil(lambda: not w._reclustering, timeout=5000)
    assert calls == [0.45], "重排必须用新阈值执行"


def test_apply_settings_no_recluster_when_unchanged(tmp_path, qapp):
    store = Store(tmp_path / "t.db")
    calls = []

    def factory():
        class _Cl:
            threshold = 0.36
            def recluster_auto(self):
                calls.append(1)
            def invalidate(self, pid):
                pass
        return _Cl()

    w = MainWindow(store, lambda: None, tmp_path / "s.json", factory)
    old = {"threshold": 0.36, "time_boost": 1.1, "time_window": 30}
    w._apply_settings(old, dict(old))
    assert calls == []
    assert not w._reclustering


def test_window_state_restored(tmp_path, qapp):
    w1 = _win(tmp_path)
    w1.resize(777, 555)
    w1.roots = ["X:/pics"]
    w1.view_selector.setCurrentIndex(2)
    w1.close()  # closeEvent 持久化几何/目录/视图
    w2 = _win(tmp_path)
    assert (w2.size().width(), w2.size().height()) == (777, 555)
    assert w2.roots == ["X:/pics"]
    assert w2.view_selector.currentIndex() == 2


def test_shortcuts_bound(tmp_path, qapp):
    w = _win(tmp_path)
    assert w.act_dir.shortcut().toString().lower() == "ctrl+o"
    assert w.act_scan.shortcut().toString().lower() == "f5"
    assert w.act_export.shortcut().toString().lower() == "ctrl+e"
    assert w.act_settings.shortcut().toString().lower() == "ctrl+,"


def test_empty_state_hint(tmp_path, qapp):
    w = _win(tmp_path)
    w.refresh_view()
    assert w._empty_hint.isVisibleTo(w.grid), "空库应显示上手引导"
    w.store.upsert_photo("a.jpg", 1.0, 1)
    w.refresh_view()
    assert not w._empty_hint.isVisibleTo(w.grid), "有照片后引导应隐藏"


def test_status_counts_label(tmp_path, qapp):
    w = _win(tmp_path)
    w.store.upsert_photo("a.jpg", 1.0, 1)
    w.store.create_person("甲")
    w.refresh_view()
    txt = w._counts_label.text()
    assert "1" in txt and "照片" in txt and "1" in txt and "人" in txt
