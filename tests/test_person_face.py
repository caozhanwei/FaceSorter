# -*- coding: utf-8 -*-
"""人物平均脸：对齐裁剪的像素均值（纯函数，UI 线程外计算）。"""
import numpy as np, cv2
from app.person_face import average_face


def _crop(path, bgr):
    cv2.imencode(".jpg", np.full((160, 160, 3), bgr, np.uint8))[1].tofile(str(path))


def test_average_face_pixel_mean(tmp_path):
    _crop(tmp_path / "1.jpg", (100, 20, 0))
    _crop(tmp_path / "2.jpg", (20, 100, 0))
    avg = average_face([tmp_path / "1.jpg", tmp_path / "2.jpg"])
    assert avg.shape[:2] == (160, 160)
    px = avg[0, 0]
    assert abs(int(px[0]) - 60) <= 3 and abs(int(px[1]) - 60) <= 3 and int(px[2]) <= 3


def test_average_face_none_when_no_usable(tmp_path):
    assert average_face([]) is None
    assert average_face([tmp_path / "nope.jpg"]) is None


def test_average_face_skips_missing_then_caps(tmp_path):
    # 缺失文件不占采样名额；凑满 max_n 张可用即止（第 3 张不参与）
    _crop(tmp_path / "1.jpg", (100, 0, 0))
    _crop(tmp_path / "2.jpg", (0, 100, 0))
    _crop(tmp_path / "3.jpg", (0, 0, 200))
    avg = average_face([tmp_path / "missing.jpg", tmp_path / "1.jpg",
                        tmp_path / "2.jpg", tmp_path / "3.jpg"], max_n=2)
    px = avg[0, 0]
    # 只有 1、2 号参与（B=100 与 G=100），3 号的 R=200 不得混入
    assert abs(int(px[0]) - 50) <= 3 and abs(int(px[1]) - 50) <= 3 and int(px[2]) <= 3
