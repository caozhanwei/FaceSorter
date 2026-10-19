import json
from pathlib import Path

DEFAULTS = {"threshold": 0.5, "time_boost": 1.1, "time_window": 30, "thumb_dir": None}

def load_settings(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(Path(path).read_text("utf-8")))
    except Exception:
        pass
    return cfg

def save_settings(path: Path, cfg: dict):
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
