# -*- coding: utf-8 -*-
"""安装包测试：双变体 MSI（A=YuNet 主线升级 / B=ArcFace 并排）元数据 + 模型清单 + 解包三查。
未构建时跳过（与 tests/assets 真实照片用例同策略）。
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("win32com.client")

_REPO = Path(__file__).resolve().parent.parent

PRODUCT_VERSION = "1.1.0"  # 与 build-msi.bat PRODUCTVERSION 同步

VARIANTS = {
    "FaceSorter.msi": {
        "product": "FaceSorter",
        "upgrade_code": "b0a966f0-e08a-471b-968a-82bfcc114858",  # 沿用 1.0.x → 可覆盖升级
        "exe": "FaceSorter.exe",
        "models": ["face_detection_yunet_2023mar.onnx",
                   "face_recognition_sface_2021dec.onnx"],
    },
    "FaceSorter-ArcFace.msi": {
        "product": "FaceSorter-ArcFace",
        "upgrade_code": "fef6d9a2-c610-41ed-a205-1ebeb50f2edc",  # 独立变体：与主线并排安装
        "exe": "FaceSorter-ArcFace.exe",
        "models": ["det_10g.onnx", "w600k_r50.onnx"],
    },
}

pytestmark = pytest.mark.parametrize("msi_name", list(VARIANTS))


def _msi_path(msi_name):
    return _REPO / "dist" / msi_name


def _query_single(db, sql):
    view = db.OpenView(sql)
    view.Execute(None)
    rec = view.Fetch()
    view.Close()
    return None if rec is None else rec.StringData(1)


@pytest.fixture()
def msi_db(msi_name):
    msi_path = _msi_path(msi_name)
    if not msi_path.exists():
        pytest.skip(f"{msi_path} 未构建，先运行 build.bat + build-msi.bat")
    # 本机 msi.dll 的 IDispatch 不解析名字（GetIDsOfNames 失败），
    # 必须走 EnsureModule 早期绑定 + 手动包装；Execute 的可选参数须显式传 None
    import win32com.client
    from win32com.client import gencache

    mod = gencache.EnsureModule("{000C1092-0000-0000-C000-000000000046}", 1033, 1, 0)
    assert mod is not None, "WindowsInstaller typelib 加载失败"
    raw = win32com.client.Dispatch("WindowsInstaller.Installer")
    installer = mod.Installer(raw._oleobj_)
    db = installer.OpenDatabase(str(msi_path), 0)  # 0 = 只读
    yield installer, db


def test_msi_product_metadata(msi_db, msi_name):
    _, db = msi_db
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'ProductName'") == VARIANTS[msi_name]["product"]
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'ProductVersion'") == PRODUCT_VERSION
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'Manufacturer'") == "FaceSorter"


def test_msi_is_per_user_no_admin(msi_db):
    # InstallScope="perUser"：ALLUSERS 空/缺 = 按用户安装，注册 HKCU，免管理员
    _, db = msi_db
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'ALLUSERS'") in ("", None)


def test_msi_upgrade_code_stable(msi_db, msi_name):
    # UpgradeCode 每变体固定不变，后续升级版 MSI 才能被 MajorUpgrade 识别
    _, db = msi_db
    value = _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'UpgradeCode'")
    assert value is not None
    assert value.lower() == "{%s}" % VARIANTS[msi_name]["upgrade_code"]


def test_msi_contains_app_and_models(msi_db, msi_name):
    # heat 收割后主 exe + 引擎模型必须都在 File 表里（引擎随包分发，勿依赖下载）
    _, db = msi_db
    view = db.OpenView("SELECT `File`.`FileName` FROM `File`")
    view.Execute(None)
    names = []
    while True:
        rec = view.Fetch()
        if rec is None:
            break
        names.append(rec.StringData(1).split("|")[-1])  # 8.3 短名|长名 取长名
    view.Close()
    assert VARIANTS[msi_name]["exe"] in names
    for m in VARIANTS[msi_name]["models"]:
        assert m in names


def test_msi_has_install_dir_dialog(msi_db):
    # 用户验收：安装时须可选安装位置 —— WixUI_InstallDir 提供欢迎页+目录选择页
    _, db = msi_db
    view = db.OpenView("SELECT `Dialog` FROM `Dialog`")
    view.Execute(None)
    names = set()
    while True:
        rec = view.Fetch()
        if rec is None:
            break
        names.add(rec.StringData(1))
    view.Close()
    assert "InstallDirDlg" in names
    assert "WelcomeDlg" in names


def test_admin_image_extraction(msi_name):
    # msiexec /a 管理员解包 = 端到端验证 cab/media 布线，无需真实安装
    msi_path = _msi_path(msi_name)
    if not msi_path.exists():
        pytest.skip(f"{msi_path} 未构建")
    exe_name = VARIANTS[msi_name]["exe"]
    with tempfile.TemporaryDirectory(prefix="fs_msi_") as tmp:
        proc = subprocess.run(
            ["msiexec", "/a", str(msi_path), "/qn", f"TARGETDIR={tmp}"],
            capture_output=True,
            timeout=600,
        )
        assert proc.returncode == 0, f"管理员解包失败 rc={proc.returncode}"
        extracted = list(Path(tmp).rglob(exe_name))
        assert extracted, f"解包产物中未找到 {exe_name}"
