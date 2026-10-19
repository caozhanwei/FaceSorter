from datetime import datetime
import cv2, numpy as np

TARGET = 112
GRID = 8
LBP_BINS = 59

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

def _uniform_table():
    table = np.zeros(256, dtype=np.uint8)
    n = 0
    for code in range(256):
        bits = np.array([(code >> i) & 1 for i in range(8)], dtype=np.uint8)
        if int(np.sum(bits != np.roll(bits, 1))) <= 2:
            table[code] = n; n += 1
        else:
            table[code] = LBP_BINS - 1
    return table

_TABLE = _uniform_table()
_HOG = cv2.HOGDescriptor((TARGET, TARGET), (16, 16), (16, 16), (8, 8), 9)

def _lbp(gray):
    g = gray.astype(np.float32)
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, dtype=np.uint8)
    for i, (dr, dc) in enumerate(_OFFSETS):
        nb = g[1+dr:1+dr+c.shape[0], 1+dc:1+dc+c.shape[1]]
        code |= ((nb >= c).astype(np.uint8) << i)
    return _TABLE[code]

def extract(gray):
    img = cv2.resize(gray, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    lbp = _lbp(padded)
    cell = TARGET // GRID
    hists = []
    for r in range(GRID):
        for c in range(GRID):
            block = lbp[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            hists.append(np.bincount(block.ravel(), minlength=LBP_BINS))
    lbp_vec = np.concatenate(hists).astype(np.float32)
    hog_vec = _HOG.compute(img).ravel().astype(np.float32)
    v = np.concatenate([lbp_vec, hog_vec])
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

def hellinger(a, b):
    # 特征是 L2 归一化的：先 L2 归一化 sqrt 向量再点积，保证 BC ∈ [0,1]
    sa, sb = np.sqrt(a).astype(np.float64), np.sqrt(b).astype(np.float64)
    na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
    if na == 0.0 or nb == 0.0:
        return 1.0
    bc = float(np.dot(sa / na, sb / nb))
    return float(np.sqrt(max(0.0, 1.0 - bc)))

def parse_time(s):
    if not s:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
