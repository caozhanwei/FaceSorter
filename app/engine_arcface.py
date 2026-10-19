# -*- coding: utf-8 -*-
"""引擎B：SCRFD-10G 检测（det_10g.onnx）+ ArcFace R50 嵌入（w600k_r50.onnx）。

不依赖 insightface 包（Windows 编译难、PyInstaller 打包脆），用 onnxruntime
手写 SCRFD 解码与 5 点对齐，行为对齐 insightface 官方实现。
模型权重仅限非商业用途（见 models/LICENSES.txt），个人照片整理符合该限制。
"""
import cv2, numpy as np

from app.detector import FaceBox, FaceDetector
from app.model_paths import model_dir, ARCFACE_FILES
from app.facecrop import umeyama_similarity, template_at

_STRIDES = (8, 16, 32)


def _nms_indices(boxes, scores, thr=0.4):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0]); keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        iw = np.clip(xx2 - xx1, 0, None); ih = np.clip(yy2 - yy1, 0, None)
        inter = iw * ih
        union = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                 + (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
                 - inter)
        order = rest[inter / np.maximum(union, 1e-9) <= thr]
    return keep


class ArcFaceDetector:
    def __init__(self, det_len=640, score_threshold=0.5,
                 min_cluster=60, min_quality=60.0):
        import onnxruntime as ort  # 延迟导入：A 版打包不含 onnxruntime
        d = model_dir()
        self._det = ort.InferenceSession(
            str(d / ARCFACE_FILES[0]), providers=["CPUExecutionProvider"])
        self._rec = ort.InferenceSession(
            str(d / ARCFACE_FILES[1]), providers=["CPUExecutionProvider"])
        self._det_in = self._det.get_inputs()[0].name
        self._rec_in = self._rec.get_inputs()[0].name
        self.det_len = det_len
        self.cfg = dict(min_cluster=min_cluster, min_quality=min_quality,
                        score_threshold=score_threshold)

    def _gate(self, w, h, quality):
        return (min(w, h) >= self.cfg["min_cluster"]
                and quality >= self.cfg["min_quality"])

    def _decode(self, outputs, det_w, det_h):
        groups = {1: [], 4: [], 10: []}
        for out in outputs:
            arr = np.asarray(out).squeeze()
            key = 1 if arr.ndim == 1 else arr.shape[-1]
            groups[key].append(arr.reshape(-1, key))
        boxes, kps_all, scores = [], [], []
        for group in groups.values():
            group.sort(key=len, reverse=True)  # anchor 数降序 = stride 8/16/32
        for stride, sc, bb, kp in zip(_STRIDES, groups[1], groups[4], groups[10]):
            h, w = det_h // stride, det_w // stride
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1) \
                .astype(np.float32).reshape(-1, 2) * stride
            n_anchor = len(bb) // len(centers)  # det_10g 每格 2 anchor
            if n_anchor > 1:
                centers = np.stack([centers] * n_anchor, axis=1).reshape(-1, 2)
            dist = bb.reshape(-1, 4) * stride
            box = np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], 1)
            kps = centers[:, None, :] + kp.reshape(-1, 5, 2) * stride
            boxes.append(box); kps_all.append(kps.reshape(-1, 10))
            scores.append(sc.reshape(-1))
        return (np.concatenate(boxes), np.concatenate(kps_all), np.concatenate(scores))

    def detect(self, img_bgr):
        import onnxruntime as ort  # noqa: F401  仅为类型提示可读性
        h, w = img_bgr.shape[:2]
        L = self.det_len
        # 保比例缩到 L×L 内，再拉到正方形（与 insightface 行为一致）
        if h / w > 1.0:
            nh, nw = L, max(1, int(round(L * w / h)))
        else:
            nw, nh = L, max(1, int(round(L * h / w)))
        resized = cv2.resize(img_bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 128, (L, L),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self._det.run(None, {self._det_in: blob})
        boxes, kps_all, scores = self._decode(outputs, L, L)
        keep = _nms_indices(boxes, scores)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sx, sy = w / L, h / L  # 检测空间 → 原图（按轴，比例不完全一致）
        out = []
        for i in keep:
            if scores[i] < self.cfg["score_threshold"]:
                continue
            x1, y1, x2, y2 = boxes[i] * (sx, sy, sx, sy)
            kps = (kps_all[i].reshape(5, 2) * (sx, sy)).ravel()
            x, y = max(0, int(x1)), max(0, int(y1))
            fw, fh = int(x2) - x, int(y2) - y
            if fw <= 0 or fh <= 0:
                continue
            crop = gray[y:y + fh, x:x + fw]
            quality = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
            row = np.concatenate([[x, y, fw, fh], kps, [scores[i]]]).astype(np.float32)
            out.append(FaceBox(x, y, fw, fh, quality,
                               self._gate(fw, fh, quality),
                               row=row, score=float(scores[i])))
        return out

    def embed(self, img_bgr, box):
        kps = box.row[4:14].reshape(5, 2)
        m = umeyama_similarity(kps, template_at(112))
        aligned = cv2.warpAffine(img_bgr, m, (112, 112),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        v = self._rec.run(None, {self._rec_in: blob})[0].ravel().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


def build_detector():
    try:
        return ArcFaceDetector()
    except Exception:
        return FaceDetector()
