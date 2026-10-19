# -*- coding: utf-8 -*-
"""安装包测试：MSI 元数据 + 文件清单 + 管理员解包三查（TDD 先于 wxs/build 脚本）。

dist/FaceSorter.msi 由 build-msi.bat 产出；未构建时跳过（与 tests/assets 真实照片用例同策略）。
"""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("win32com.client")

_REPO = Path(__file__).resolve().parent.parent
MSI_PATH = _REPO / "dist" / "FaceSorter.msi"

pytestmark = pytest.mark.skipif(
    not MSI_PATH.exists(), reason="dist/FaceSorter.msi 未构建，先运行 build-msi.bat"
)

UPGRADE_CODE = "b0a966f0-e08a-471b-968a-82bfcc114858"


def _query_single(db, sql):
    view = db.OpenView(sql)
    view.Execute(None)
    rec = view.Fetch()
    view.Close()
    return None if rec is None else rec.StringData(1)


@pytest.fixture()
def msi_db():
    # 本机 msi.dll 的 IDispatch 不解析名字（GetIDsOfNames 失败），
    # 必须走 EnsureModule 早期绑定 + 手动包装；Execute 的可选参数须显式传 None
    import win32com.client
    from win32com.client import gencache

    mod = gencache.EnsureModule("{000C1092-0000-0000-C000-000000000046}", 1033, 1, 0)
    assert mod is not None, "WindowsInstaller typelib 加载失败"
    raw = win32com.client.Dispatch("WindowsInstaller.Installer")
    installer = mod.Installer(raw._oleobj_)
    db = installer.OpenDatabase(str(MSI_PATH), 0)  # 0 = 只读
    yield installer, db


def test_msi_product_metadata(msi_db):
    _, db = msi_db
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'ProductName'") == "FaceSorter"
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'ProductVersion'") == "1.0.2"  # 与 build-msi.bat PRODUCTVERSION 同步
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'Manufacturer'") == "FaceSorter"


def test_msi_is_per_user_no_admin(msi_db):
    # InstallScope="perUser"：ALLUSERS 空/缺 = 按用户安装，注册 HKCU，免管理员
    _, db = msi_db
    assert _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'ALLUSERS'") in ("", None)


def test_msi_upgrade_code_stable(msi_db):
    # UpgradeCode 固定不变，后续升级版 MSI 才能被 MajorUpgrade 识别
    _, db = msi_db
    value = _query_single(db, "SELECT `Value` FROM `Property` WHERE `Property` = 'UpgradeCode'")
    assert value is not None
    assert value.lower() == "{%s}" % UPGRADE_CODE


def test_msi_contains_app_exe(msi_db):
    # heat 收割 dist\FaceSorter 后主 exe 必须在 File 表里
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
    assert "FaceSorter.exe" in names


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


def test_admin_image_extraction():
    # msiexec /a 管理员解包 = 端到端验证 cab/media 布线，无需真实安装
    with tempfile.TemporaryDirectory(prefix="fs_msi_") as tmp:
        proc = subprocess.run(
            ["msiexec", "/a", str(MSI_PATH), "/qn", f"TARGETDIR={tmp}"],
            capture_output=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"管理员解包失败 rc={proc.returncode}"
        extracted = list(Path(tmp).rglob("FaceSorter.exe"))
        assert extracted, "解包产物中未找到 FaceSorter.exe"
