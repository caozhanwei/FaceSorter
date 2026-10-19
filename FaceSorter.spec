# -*- mode: python ; coding: utf-8 -*-
# A 版：YuNet+SFace（主线升级，UpgradeCode 沿用 b0a966f0-…）
from PyInstaller.utils.hooks import collect_all

datas = [
    ('models/face_detection_yunet_2023mar.onnx', 'models'),
    ('models/face_recognition_sface_2021dec.onnx', 'models'),
    ('models/LICENSES.txt', 'models'),
]
binaries = []
hiddenimports = []
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceSorter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceSorter',
)
