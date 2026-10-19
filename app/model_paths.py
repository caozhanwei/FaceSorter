import sys
from pathlib import Path


def model_dir() -> Path:
    # 冻结（onedir）：PyInstaller 把 datas 放在 _MEIPASS(=_internal)下；
    # 开发态：仓库根 models/
    base = getattr(sys, "_MEIPASS", None)
    root = Path(base) if base else Path(__file__).resolve().parent.parent
    return root / "models"


YUNET_FILES = ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")
ARCFACE_FILES = ("det_10g.onnx", "w600k_r50.onnx")


def has_yunet_models() -> bool:
    return all((model_dir() / f).is_file() for f in YUNET_FILES)


def has_arcface_models() -> bool:
    return all((model_dir() / f).is_file() for f in ARCFACE_FILES)
