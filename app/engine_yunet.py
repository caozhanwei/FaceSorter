# -*- coding: utf-8 -*-
"""引擎A：YuNet 检测（cv2.FaceDetectorYN）+ SFace 128维嵌入（cv2.FaceRecognizerSF）。

模型缺失（打包事故/用户删文件）时 build_detector 回退 Haar+LBP，应用不致不可用。
"""
import cv2, numpy as np

from app.detector import FaceBox, FaceDetector
from app.model_paths import model_dir, YUNET_FILES

# OpenCV 官方教程给 SFace 余弦距离的判定阈值 0.363（1:1 验证）
DEFAULT_COSINE_THRESHOLD = 0.36


class YuNetFaceDetector:
    def __init__(self, max_side=1024, score_threshold=0.6,
                 min_cluster=60, min_quality=60.0):
        d = model_dir()
        self._det = cv2.FaceDetectorYN.create(str(d / YUNET_FILES[0]), "",
                                              (320, 320),
                                              score_threshold=score_threshold,
                                              nms_threshold=0.3, top_k=5000)
        self._rec = cv2.FaceRecognizerSF.create(str(d / YUNET_FILES[1]), "")
        self.cfg = dict(max_side=max_side, min_cluster=min_cluster,
                        min_quality=min_quality)

    def _gate(self, w, h, quality):
        return min(w, h) >= self.cfg["min_cluster"] and quality >= self.cfg["min_quality"]

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = self.cfg["max_side"] / max(h, w)
        if scale < 1.0:
            small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small, scale = img_bgr, 1.0
        self._det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._det.detect(small)
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        out = []
        if faces is None:
            return out
        for fr in faces:
            row = np.asarray(fr, dtype=np.float32)
            row[:4] /= scale  # 框与关键点换回原图坐标
            row[4:14] /= scale
            x, y, fw, fh = (int(v) for v in row[:4])
            if fw <= 0 or fh <= 0:
                continue
            crop = gray_full[y:y + fh, x:x + fw]
            quality = (float(cv2.Laplacian(crop, cv2.CV_64F).var())
                       if crop.size else 0.0)
            out.append(FaceBox(x, y, fw, fh, quality,
                               self._gate(fw, fh, quality),
                               row=row, score=float(row[14])))
        return out

    def embed(self, img_bgr, box):
        # FaceRecognizerSF.alignCrop 需要 detect 原始 15 维行（含关键点）做对齐
        aligned = self._rec.alignCrop(img_bgr, box.row)
        v = self._rec.feature(aligned).ravel().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def build_detector():
    try:
        return YuNetFaceDetector()
    except cv2.error:
        return FaceDetector()
