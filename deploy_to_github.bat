@echo off
REM Script otomatis untuk deploy ke Streamlit Cloud (Windows)
REM Pastikan Anda sudah:
REM 1. Install Git (https://git-scm.com/download/win)
REM 2. Buat akun GitHub (https://github.com)
REM 3. Buat repository kosong di GitHub (https://github.com/new)

setlocal enabledelayedexpansion

echo ============================================
echo Deploy Cats vs Dogs Classifier ke Streamlit Cloud
echo ============================================
echo.

REM Input dari user
set /p GITHUB_USERNAME="Masukkan GitHub username Anda: "
set /p REPO_NAME="Masukkan nama repository GitHub (default: Project-Akhir-PSD): "
if "!REPO_NAME!"=="" set REPO_NAME=Project-Akhir-PSD
set /p USER_NAME="Masukkan nama Anda (untuk git config): "
set /p USER_EMAIL="Masukkan email Anda (untuk git config): "

echo.
echo Konfigurasi:
echo   GitHub Username: !GITHUB_USERNAME!
echo   Repository Name: !REPO_NAME!
echo   Git User: !USER_NAME! ^<!USER_EMAIL!^>
echo.

REM Cek apakah git terinstall
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git tidak terinstall atau tidak ada di PATH
    echo Install dari https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Inisialisasi git lokal
echo [1/6] Inisialisasi repository lokal...
git init
if errorlevel 1 (
    echo ERROR: Gagal menjalankan git init
    pause
    exit /b 1
)

REM Konfigurasi user
echo [2/6] Konfigurasi git user...
git config user.name "!USER_NAME!"
git config user.email "!USER_EMAIL!"

REM Add semua file
echo [3/6] Menambahkan semua file...
git add .
if errorlevel 1 (
    echo ERROR: Gagal menjalankan git add
    pause
    exit /b 1
)

REM Commit
echo [4/6] Melakukan commit...
git commit -m "Initial commit: Cats vs Dogs classifier dengan Streamlit demo"
if errorlevel 1 (
    echo ERROR: Gagal melakukan commit
    pause
    exit /b 1
)

REM Rename branch ke main
echo [5/6] Rename branch ke 'main'...
git branch -M main

REM Add remote dan push
echo [6/6] Push ke GitHub...
echo.
echo   URL Repository: https://github.com/!GITHUB_USERNAME!/!REPO_NAME!.git
echo.
git remote add origin https://github.com/!GITHUB_USERNAME!/!REPO_NAME!.git
if errorlevel 1 (
    echo ERROR: Gagal add remote
    pause
    exit /b 1
)

git push -u origin main
if errorlevel 1 (
    echo ERROR: Gagal push. Pastikan:
    echo   1. Repository sudah dibuat di GitHub
    echo   2. URL GitHub username dan nama repo benar
    echo   3. Anda sudah login GitHub di git (git credential manager)
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! Push ke GitHub selesai.
echo ============================================
echo.
echo Langkah selanjutnya:
echo 1. Buka https://streamlit.io/cloud
echo 2. Sign up / Login dengan GitHub
echo 3. Klik "New app" -> "Create app"
echo 4. Pilih repository: %GITHUB_USERNAME%/%REPO_NAME%
echo 5. Branch: main
echo 6. Main file path: app.py
echo 7. Klik "Deploy"
echo.
echo Tunggu 2-5 menit hingga aplikasi selesai di-deploy.
echo Anda akan mendapat URL publik seperti:
echo   https://%REPO_NAME%-lowercase.streamlit.app/
echo.
echo Bagikan URL tersebut ke dosen!
echo ============================================
pause
