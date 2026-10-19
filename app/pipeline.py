import os
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from app.scanner import scan_roots
from app.features import extract
from app.detector import expand_box
from app.store import pack_feature

def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None

def imwrite_unicode(path, img, quality=80):
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            buf.tofile(path)
        return ok
    except Exception:
        return False

def read_exif_taken_at(path):
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            return ex.get(36867) or ex.get(306)
    except Exception:
        return None

def make_thumb(img, thumb_path: Path):
    h, w = img.shape[:2]
    scale = 256 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    return imwrite_unicode(str(thumb_path), img)

def process_one(path, store, detector, clusterer, thumb_dir: Path):
    path = Path(path).as_posix()  # 统一正斜杠（直接调用时可能是反斜杠）
    st = os.stat(path)
    photo_id = store.upsert_photo(path, st.st_mtime, st.st_size)  # 已存在则取回 id
    img = imread_unicode(path)
    if img is None:
        store.update_photo_meta(photo_id, status="bad")
        return
    h, w = img.shape[:2]
    store.update_photo_meta(photo_id, width=w, height=h,
                            exif_taken_at=read_exif_taken_at(path))
    thumb_path = Path(thumb_dir) / f"{photo_id}.jpg"
    if make_thumb(img, thumb_path):
        store.update_photo_meta(photo_id, thumb_path=str(thumb_path))
    boxes = detector.detect(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for b in boxes:
        feature = None
        if b.clusturable:
            x1, y1, x2, y2 = expand_box(b.x, b.y, b.w, b.h, w, h)
            feature = pack_feature(extract(gray[y1:y2, x1:x2]))
        face_id = store.add_face(photo_id, b.x, b.y, b.w, b.h, b.quality, feature)
        if b.clusturable and clusterer is not None:
            clusterer.assign(face_id)
    store.update_photo_meta(photo_id, has_face=1 if boxes else 0)

def run_scan(roots, store, detector, clusterer, thumb_dir,
             progress=None, cancel=None):
    thumb_prefix = Path(thumb_dir).as_posix().rstrip("/") + "/"
    known_before = set(store.known_fingerprints().keys())
    todo = [p for p in scan_roots(roots, store) if not p.startswith(thumb_prefix)]
    total = len(todo)
    for i, path in enumerate(todo):
        if cancel is not None and cancel.is_set():
            # 取消：撤销本次新入库但未处理的行，保持"已处理才算入索引"，
            # 下次扫描这些照片仍会进入 todo
            fresh = [p for p in todo[i:] if p not in known_before]
            if fresh:
                store.con.executemany("DELETE FROM photos WHERE path=?",
                                      [(p,) for p in fresh])
                store.con.commit()
            return
        try:
            process_one(path, store, detector, clusterer, Path(thumb_dir))
        except OSError:
            raise  # 磁盘满等：上抛给 ScanWorker 提示并停止（spec §8）
        except Exception:
            store.con.execute("UPDATE photos SET status='bad' WHERE path=?", (path,))
            store.con.commit()
        if progress:
            progress(i + 1, total)
