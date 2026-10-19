import json, pytest
from app.settings import load_settings, save_settings, DEFAULTS

def test_roundtrip_defaults(tmp_path):
    p = tmp_path / "s.json"
    cfg = load_settings(p)
    assert cfg == DEFAULTS
    cfg["threshold"] = 0.4
    save_settings(p, cfg)
    assert json.loads(p.read_text("utf-8"))["threshold"] == 0.4
    assert load_settings(p)["threshold"] == 0.4


def test_engine_migration_resets_stale_threshold(tmp_path):
    # 1.0.x 升上来：旧 LBP 阈值 0.5 对余弦距离过松（乱并堆根因之一），必须重置
    p = tmp_path / "s.json"
    save_settings(p, {"threshold": 0.5, "time_boost": 1.1, "time_window": 30})
    cfg = load_settings(p, engine="yunet")
    assert cfg["threshold"] == pytest.approx(0.36)
    assert cfg["engine"] == "yunet"

def test_same_engine_keeps_custom_threshold(tmp_path):
    p = tmp_path / "s.json"
    save_settings(p, {"threshold": 0.4, "engine": "yunet"})
    assert load_settings(p, engine="yunet")["threshold"] == pytest.approx(0.4)

def test_engine_switch_to_arcface_resets(tmp_path):
    p = tmp_path / "s.json"
    save_settings(p, {"threshold": 0.25, "engine": "yunet"})
    cfg = load_settings(p, engine="arcface")
    assert cfg["threshold"] == pytest.approx(0.40) and cfg["engine"] == "arcface"

def test_no_engine_arg_is_legacy_neutral(tmp_path):
    # load_settings() 无 engine 参数时行为不变（老调用方/测试兼容）
    p = tmp_path / "s.json"
    save_settings(p, {"threshold": 0.5})
    assert load_settings(p)["threshold"] == pytest.approx(0.5)
