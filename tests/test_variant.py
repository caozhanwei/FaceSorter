# -*- coding: utf-8 -*-
"""构建变体解析：A=YuNet+SFace（主线升级），B=SCRFD+ArcFace（并排对照）。"""
import pytest
from app import variant as V


def test_prefers_yunet_when_models_present(monkeypatch):
    monkeypatch.delenv("FACE_SORTER_ENGINE", raising=False)
    monkeypatch.setattr(V, "has_yunet_models", lambda: True)
    monkeypatch.setattr(V, "has_arcface_models", lambda: True)
    assert V.resolve_variant() == "yunet"  # 开发态两者都在 → 默认主线


def test_arcface_only_build(monkeypatch):
    monkeypatch.delenv("FACE_SORTER_ENGINE", raising=False)
    monkeypatch.setattr(V, "has_yunet_models", lambda: False)
    monkeypatch.setattr(V, "has_arcface_models", lambda: True)
    assert V.resolve_variant() == "arcface"  # B 版 MSI 只带 arcface 模型


def test_env_override_for_dev(monkeypatch):
    monkeypatch.setenv("FACE_SORTER_ENGINE", "arcface")
    monkeypatch.setattr(V, "has_yunet_models", lambda: True)
    assert V.resolve_variant() == "arcface"


def test_no_models_none(monkeypatch):
    monkeypatch.delenv("FACE_SORTER_ENGINE", raising=False)
    monkeypatch.setattr(V, "has_yunet_models", lambda: False)
    monkeypatch.setattr(V, "has_arcface_models", lambda: False)
    assert V.resolve_variant() == "none"  # 打包事故 → Haar 回退


def test_labels():
    assert V.ENGINE_LABELS["arcface"] and V.ENGINE_LABELS["yunet"]
    assert V.ENGINE_LABELS["none"] == ""


def test_threshold_per_engine():
    assert V.engine_threshold("yunet") == pytest.approx(0.36)
    assert V.engine_threshold("arcface") == pytest.approx(0.40)
    assert V.engine_threshold("none") == pytest.approx(0.5)  # Haar 沿用旧默认


def test_arcface_factory_tunes_ort_threads_from_settings(tmp_path, monkeypatch):
    # 多线程扫描防超订：workers×intra ≈ 核数。工厂每次现读 settings 的 scan_threads
    import os
    from app.settings import save_settings
    captured = {}
    import app.engine_arcface as ea
    monkeypatch.setattr(ea, "build_detector",
                        lambda intra_threads=None: captured.update(t=intra_threads) or 1)
    s = tmp_path / "s.json"
    save_settings(s, {"scan_threads": 4})
    fac = V.build_detector_factory("arcface", s)
    fac()
    cpu = os.cpu_count() or 4
    assert captured["t"] == max(1, cpu // 4)
    save_settings(s, {"scan_threads": 1})
    fac()
    assert captured["t"] is None  # 单线程：ORT 默认（吃满核）


def test_arcface_factory_without_settings_keeps_default():
    fac = V.build_detector_factory("arcface")
    assert callable(fac)
