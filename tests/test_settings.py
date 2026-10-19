import json
from app.settings import load_settings, save_settings, DEFAULTS

def test_roundtrip_defaults(tmp_path):
    p = tmp_path / "s.json"
    cfg = load_settings(p)
    assert cfg == DEFAULTS
    cfg["threshold"] = 0.4
    save_settings(p, cfg)
    assert json.loads(p.read_text("utf-8"))["threshold"] == 0.4
    assert load_settings(p)["threshold"] == 0.4
