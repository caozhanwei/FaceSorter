@echo off
rem FaceSorter 一键 MSI 打包：heat 收割 dist\<ProductDir> -> candle -> light
rem 用法: build-msi.bat [yunet^|arcface]（默认 yunet = 主线 FaceSorter.msi）
rem 依赖：先跑 build.bat 产出对应 dist 目录；WiX 3.14 在 %WIX%
setlocal
if not defined WIX set WIX=G:\work\tools\wix314\tools
set PRODUCTVERSION=1.4.1
set OBJ=installer\obj

if /i "%1"=="arcface" (
  set PRODUCTNAME=FaceSorter-ArcFace
  set PRODUCTDIR=FaceSorter-ArcFace
  set INSTALLDIRNAME=FaceSorter-ArcFace
  set EXENAME=FaceSorter-ArcFace.exe
  set UPGRADECODE=fef6d9a2-c610-41ed-a205-1ebeb50f2edc
  set MSIFILE=dist\FaceSorter-ArcFace.msi
) else (
  set PRODUCTNAME=FaceSorter
  set PRODUCTDIR=FaceSorter
  set INSTALLDIRNAME=FaceSorter
  set EXENAME=FaceSorter.exe
  set UPGRADECODE=b0a966f0-e08a-471b-968a-82bfcc114858
  set MSIFILE=dist\FaceSorter.msi
)

if not exist "%WIX%\candle.exe" (
  echo [ERROR] WiX toolchain not found at "%WIX%" & exit /b 1
)
if not exist "dist\%PRODUCTDIR%\%EXENAME%" (
  echo [ERROR] dist\%PRODUCTDIR%\%EXENAME% missing, run build.bat first & exit /b 1
)
if not exist "%OBJ%" mkdir "%OBJ%"

echo [1/3] heat harvest dist\%PRODUCTDIR% ...
"%WIX%\heat.exe" dir dist\%PRODUCTDIR% -gg -sfrag -srd -dr INSTALLFOLDER -cg FaceSorterFiles -var var.SourceDir -out "%OBJ%\FaceSorterFiles.wxs"
if errorlevel 1 goto :fail

echo [2/3] candle compile ...
"%WIX%\candle.exe" -dProductVersion=%PRODUCTVERSION% -dSourceDir=dist\%PRODUCTDIR% -dProductName=%PRODUCTNAME% -dInstallDirName=%INSTALLDIRNAME% -dExeName=%EXENAME% -dUpgradeCode=%UPGRADECODE% installer\FaceSorter.wxs "%OBJ%\FaceSorterFiles.wxs" -out "%OBJ%\\"
if errorlevel 1 goto :fail

echo [3/3] light link (per-user ICE 抑制) ...
"%WIX%\light.exe" -ext WixUIExtension -cultures:zh-CN -sice:ICE38 -sice:ICE43 -sice:ICE57 -sice:ICE61 -sice:ICE91 -sice:ICE64 -o %MSIFILE% "%OBJ%\FaceSorter.wixobj" "%OBJ%\FaceSorterFiles.wixobj"
if errorlevel 1 goto :fail

echo OK: %MSIFILE%
exit /b 0

:fail
echo [ERROR] MSI build failed
exit /b 1
