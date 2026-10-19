# -*- coding: utf-8 -*-
"""构建变体：同一份代码出两个 MSI。
A=FaceSorter（YuNet+SFace，主线升级），B=FaceSorter-ArcFace（SCRFD+ArcFace 并排对照）。
运行时按“打包进了哪组模型”识别变体；环境变量 FACE_SORTER_ENGINE 供开发态强制切换。
"""
import os

from app.model_paths import has_yunet_models, has_arcface_models
from app.settings import ENGINE_THRESHOLDS

ENGINE_LABELS = {"yunet": "（YuNet 引擎）", "arcface": "（ArcFace 引擎）", "none": ""}

_LBA_HAAR_THRESHOLD = 0.5  # 模型缺失回退 Haar+LBP 时的旧默认


def resolve_variant() -> str:
    env = os.environ.get("FACE_SORTER_ENGINE")
    if env in ("yunet", "arcface", "none"):
        return env
    if has_yunet_models():
        return "yunet"
    if has_arcface_models():
        return "arcface"
    return "none"


def engine_threshold(variant: str) -> float:
    if variant == "none":
        return _LBA_HAAR_THRESHOLD
    return ENGINE_THRESHOLDS.get(variant, ENGINE_THRESHOLDS["yunet"])


def build_detector_factory(variant: str):
    if variant == "arcface":
        from app.engine_arcface import build_detector
        return build_detector
    if variant == "yunet":
        from app.engine_yunet import build_detector
        return build_detector
    from app.detector import FaceDetector
    return FaceDetector
