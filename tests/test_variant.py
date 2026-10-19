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
