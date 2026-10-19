@echo off
rem 用法: build.bat [yunet^|arcface]（默认 yunet）
setlocal
cd /d %~dp0
call .venv\Scripts\activate
pip install -r requirements-dev.txt -r requirements.txt
if /i "%1"=="arcface" (
  pyinstaller --noconfirm --clean FaceSorterArcFace.spec
  echo === dist\FaceSorter-ArcFace\FaceSorter-ArcFace.exe 就绪 ===
) else (
  pyinstaller --noconfirm --clean FaceSorter.spec
  echo === dist\FaceSorter\FaceSorter.exe 就绪 ===
)
