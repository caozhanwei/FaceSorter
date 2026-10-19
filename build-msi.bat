@echo off
rem FaceSorter 一键 MSI 打包：heat 收割 dist\FaceSorter -> candle -> light
rem 依赖：PyInstaller 产物 dist\FaceSorter\ 已存在（先跑 build.bat），WiX 3.14 在 %WIX%
setlocal
set WIX=G:\work\tools\wix314\tools
set PRODUCTVERSION=1.0.1
set OBJ=installer\obj

if not exist "%WIX%\candle.exe" (
  echo [ERROR] WiX toolchain not found at %WIX% & exit /b 1
)
if not exist "dist\FaceSorter\FaceSorter.exe" (
  echo [ERROR] dist\FaceSorter\FaceSorter.exe missing, run build.bat first & exit /b 1
)
if not exist "%OBJ%" mkdir "%OBJ%"

echo [1/3] heat harvest dist\FaceSorter ...
"%WIX%\heat.exe" dir dist\FaceSorter -gg -sfrag -srd -dr INSTALLFOLDER -cg FaceSorterFiles -var var.SourceDir -out "%OBJ%\FaceSorterFiles.wxs"
if errorlevel 1 goto :fail

echo [2/3] candle compile ...
"%WIX%\candle.exe" -dProductVersion=%PRODUCTVERSION% -dSourceDir=dist\FaceSorter installer\FaceSorter.wxs "%OBJ%\FaceSorterFiles.wxs" -out "%OBJ%\\"
if errorlevel 1 goto :fail

echo [3/3] light link (per-user ICE 抑制) ...
"%WIX%\light.exe" -sice:ICE38 -sice:ICE43 -sice:ICE57 -sice:ICE61 -sice:ICE91 -sice:ICE64 -o dist\FaceSorter.msi "%OBJ%\FaceSorter.wixobj" "%OBJ%\FaceSorterFiles.wixobj"
if errorlevel 1 goto :fail

echo OK: dist\FaceSorter.msi
exit /b 0

:fail
echo [ERROR] MSI build failed
exit /b 1
