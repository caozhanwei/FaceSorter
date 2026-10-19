import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from app.scanner import scan_roots
from app.features import extract
from app.detector import expand_box
from app.store import pack_feature
from app.facecrop import aligned_crop, rect_crop, FACE_SIZE

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

def save_face_crop(img, box, face_id, face_dir: Path):
    """清晰头像：有五点行 → 模板对齐裁剪；否则矩形裁剪。取自原图非缩略图。"""
    if face_dir is None:
        return
    if box.row is not None and len(box.row) >= 14:
        crop = aligned_crop(img, box.row, FACE_SIZE)
    else:
        crop = rect_crop(img, box, FACE_SIZE)
    face_dir = Path(face_dir)
    face_dir.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(str(face_dir / f"{face_id}.jpg"), crop, quality=88)

def process_one(path, store, detector, clusterer, thumb_dir: Path,
                face_dir: Path = None):
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
            vec = None
            embed = getattr(detector, "embed", None)
            if embed is not None:
                try:
                    vec = embed(img, b)
                except Exception:
                    vec = None  # 模型损坏等：降级 LBP，扫描不中断
            if vec is None:
                x1, y1, x2, y2 = expand_box(b.x, b.y, b.w, b.h, w, h)
                vec = extract(gray[y1:y2, x1:x2])
            feature = pack_feature(vec)
        face_id = store.add_face(photo_id, b.x, b.y, b.w, b.h, b.quality, feature)
        save_face_crop(img, b, face_id, face_dir)
        if b.clusturable and clusterer is not None:
            clusterer.assign(face_id)
    store.update_photo_meta(photo_id, has_face=1 if boxes else 0)


# ---- 多线程扫描：计算（解码/检测/嵌入/编码）并行，提交（Store 写+聚类）按序单线程 ----
# 提交顺序=todo 顺序，聚类分配与串行完全一致（乱序完成只影响计算，不影响结果）

def _thumb_bytes(img):
    h, w = img.shape[:2]
    scale = 256 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def _face_crop_array(img, box):
    if box.row is not None and len(box.row) >= 14:
        return aligned_crop(img, box.row, FACE_SIZE)
    return rect_crop(img, box, FACE_SIZE)


def _compute_one(path, detector, engine_lock, want_crops):
    """工作线程负载：纯计算/IO，不碰 Store。非 OSError 异常降级 bad。"""
    st = os.stat(path)  # 磁盘级错误直接上抛终止扫描（与串行一致）
    try:
        img = imread_unicode(path)
        if img is None:
            return {"stat": st, "bad": True}
        h, w = img.shape[:2]
        payload = {"stat": st, "w": w, "h": h,
                   "exif": read_exif_taken_at(path),
                   "thumb": _thumb_bytes(img), "faces": []}

        def _det(i):
            if engine_lock is None:
                return detector.detect(i)
            with engine_lock:
                return detector.detect(i)

        boxes = _det(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for b in boxes:
            feature = None
            if b.clusturable:
                vec = None
                embed = getattr(detector, "embed", None)
                if embed is not None:
                    try:
                        if engine_lock is None:
                            vec = embed(img, b)
                        else:
                            with engine_lock:
                                vec = embed(img, b)
                    except Exception:
                        vec = None  # 模型损坏等：降级 LBP，扫描不中断
                if vec is None:
                    x1, y1, x2, y2 = expand_box(b.x, b.y, b.w, b.h, w, h)
                    vec = extract(gray[y1:y2, x1:x2])
                feature = pack_feature(vec)
            crop = _face_crop_array(img, b) if want_crops else None
            payload["faces"].append(
                (b.x, b.y, b.w, b.h, b.quality, b.clusturable, feature, crop))
        return payload
    except OSError:
        raise
    except Exception:
        return {"stat": st, "bad": True}


def _commit_one(store, clusterer, path, payload, thumb_dir, face_dir):
    st = payload["stat"]
    photo_id = store.upsert_photo(path, st.st_mtime, st.st_size)
    if payload.get("bad"):
        store.update_photo_meta(photo_id, status="bad")
        return
    store.update_photo_meta(photo_id, width=payload["w"], height=payload["h"],
                            exif_taken_at=payload["exif"])
    if payload.get("thumb") is not None:
        thumb_path = Path(thumb_dir) / f"{photo_id}.jpg"
        try:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.write_bytes(payload["thumb"])
            store.update_photo_meta(photo_id, thumb_path=str(thumb_path))
        except OSError:
            pass  # 缩略图写失败不阻断索引（与 imwrite_unicode 吞错一致）
    if face_dir is not None:
        Path(face_dir).mkdir(parents=True, exist_ok=True)
    has_face = 0
    for (x, y, w, h, quality, clusturable, feature, crop) in payload["faces"]:
        face_id = store.add_face(photo_id, x, y, w, h, quality, feature)
        if face_dir is not None and crop is not None:
            imwrite_unicode(str(Path(face_dir) / f"{face_id}.jpg"), crop,
                            quality=88)
        if clusturable and clusterer is not None:
            clusterer.assign(face_id)
        has_face = 1
    store.update_photo_meta(photo_id, has_face=has_face)


def _run_parallel(todo, store, detector, clusterer, thumb_dir, face_dir,
                  progress, cancel, workers, known_before):
    total = len(todo)
    # onnxruntime 会话并发 Run 安全；cv2 系（YuNet/SFace/Haar）未承诺线程安全须加锁
    engine_lock = None if getattr(detector, "thread_safe", False) \
        else threading.Lock()
    want_crops = face_dir is not None
    window = max(2, workers * 2)  # 提交窗口：乱序完成至多积压 window 份，限内存
    pending = {}
    next_submit = 0

    def _refill(ex):
        nonlocal next_submit
        while next_submit < total and len(pending) < window:
            pending[next_submit] = ex.submit(
                _compute_one, todo[next_submit], detector, engine_lock,
                want_crops)
            next_submit += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        _refill(ex)
        for i in range(total):
            if cancel is not None and cancel.is_set():
                # 取消：未提交入库的新行删除，保持"已处理才算入索引"下次续扫
                fresh = [p for p in todo[i:] if p not in known_before]
                if fresh:
                    store.con.executemany("DELETE FROM photos WHERE path=?",
                                          [(p,) for p in fresh])
                    store.con.commit()
                ex.shutdown(wait=False, cancel_futures=True)
                return
            payload = pending.pop(i).result()  # OSError 上抛终止扫描
            _commit_one(store, clusterer, todo[i], payload, thumb_dir,
                        face_dir)
            _refill(ex)
            if progress:
                progress(i + 1, total)

def run_scan(roots, store, detector, clusterer, thumb_dir, face_dir=None,
             progress=None, cancel=None, workers=1):
    skip_prefixes = [Path(thumb_dir)]
    if face_dir is not None:
        skip_prefixes.append(Path(face_dir))
    known_before = set(store.known_fingerprints().keys())
    todo = scan_roots(roots, store, skip_prefixes)
    total = len(todo)
    if workers and workers > 1 and total:
        _run_parallel(todo, store, detector, clusterer, Path(thumb_dir),
                      Path(face_dir) if face_dir is not None else None,
                      progress, cancel, workers, known_before)
        return
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
            process_one(path, store, detector, clusterer, Path(thumb_dir),
                        Path(face_dir) if face_dir is not None else None)
        except OSError:
            raise  # 磁盘满等：上抛给 ScanWorker 提示并停止（spec §8）
        except Exception:
            store.con.execute("UPDATE photos SET status='bad' WHERE path=?", (path,))
            store.con.commit()
        if progress:
            progress(i + 1, total)
