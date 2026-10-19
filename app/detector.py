from dataclasses import dataclass
import cv2, numpy as np

@dataclass
class FaceBox:
    x: int; y: int; w: int; h: int
    quality: float
    clusturable: bool

def expand_box(x, y, w, h, img_w, img_h, pad=0.2):
    dx, dy = int(w * pad), int(h * pad)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(img_w, x + w + dx), min(img_h, y + h + dy)
    return x1, y1, x2, y2

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

def nms(boxes, iou_thr=0.3):
    keep = []
    for b in sorted(boxes, key=lambda r: -(r[2]-r[0]) * (r[3]-r[1])):
        if all(_iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep

class FaceDetector:
    def __init__(self, max_side=1024, min_neighbors=5, min_size=48,
                 min_cluster=60, min_quality=60.0):
        base = cv2.data.haarcascades
        self._front = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self._prof = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        self.cfg = dict(max_side=max_side, min_neighbors=min_neighbors,
                        min_size=min_size, min_cluster=min_cluster,
                        min_quality=min_quality)

    def _gate(self, w, h, quality):
        return min(w, h) >= self.cfg["min_cluster"] and quality >= self.cfg["min_quality"]

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = self.cfg["max_side"] / max(h, w)
        if scale < 1.0:
            small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small, scale = img_bgr, 1.0
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        ms = self.cfg["min_size"]
        mn = self.cfg["min_neighbors"]
        raw = []
        flipped_gray = cv2.flip(gray, 1)
        for cascade, mat, flipped in ((self._front, gray, False),
                                      (self._prof, gray, False),
                                      (self._prof, flipped_gray, True)):
            for (x, y, fw, fh) in cascade.detectMultiScale(mat, scaleFactor=1.1,
                                                           minNeighbors=mn,
                                                           minSize=(ms, ms)):
                if flipped:  # 翻转图上的框要还原到原图坐标
                    x = mat.shape[1] - x - fw
                raw.append((int(x / scale), int(y / scale),
                            int((x + fw) / scale), int((y + fh) / scale)))
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for (x1, y1, x2, y2) in nms(raw, 0.3):
            crop = gray_full[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            quality = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            fw, fh = x2 - x1, y2 - y1
            if fw * 3 < fh or fh * 3 < fw:  # 长宽比过滤（Haar 误检）
                continue
            out.append(FaceBox(x1, y1, fw, fh, quality, self._gate(fw, fh, quality)))
        return out
