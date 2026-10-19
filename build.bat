@echo off
setlocal
cd /d %~dp0
call .venv\Scripts\activate
pip install -r requirements-dev.txt
pyinstaller --noconfirm --clean --windowed --onedir --name FaceSorter ^
  --collect-all cv2 main.py
echo === dist\FaceSorter\FaceSorter.exe 就绪 ===
