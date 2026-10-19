import json
from pathlib import Path

# 深度嵌入引擎各自的余弦距离默认阈值（同人判定）：
# yunet=SFace，OpenCV 官方教程值 0.363；arcface=WebFace600K R50，按 sim≥0.6 判同人
ENGINE_THRESHOLDS = {"yunet": 0.36, "arcface": 0.40}

DEFAULTS = {"threshold": ENGINE_THRESHOLDS["yunet"], "time_boost": 1.1,
            "time_window": 30, "thumb_dir": None, "engine": None,
            "scan_threads": 4}

def load_settings(path: Path, engine: str = None) -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(Path(path).read_text("utf-8")))
    except Exception:
        pass
    if engine is not None and cfg.get("engine") != engine:
        # 引擎换了（或 1.0.x 升级无 engine 键）：旧阈值在新距离空间无意义，重置
        cfg["threshold"] = ENGINE_THRESHOLDS.get(engine, cfg["threshold"])
        cfg["engine"] = engine
    return cfg

def save_settings(path: Path, cfg: dict):
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


# 这些键影响已有分配结果：改了必须重排，而不是只影响以后的扫描
RECLUSTER_KEYS = ("threshold", "time_boost", "time_window")

def need_recluster(old: dict, new: dict) -> bool:
    return any(old.get(k) != new.get(k) for k in RECLUSTER_KEYS)
