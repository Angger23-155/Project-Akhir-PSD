@echo off
REM Script untuk membersihkan repository (hapus file-file sisa eksperimen)
REM dan push ulang ke GitHub

setlocal enabledelayedexpansion

echo ============================================
echo Membersihkan Repository - Hapus File Sisa
echo ============================================
echo.

REM File-file yang akan dihapus
set FILES_TO_DELETE=^
cv_results.txt ^
cv_results_pipeline.txt ^
nested_cv_results.txt ^
random_search_results.txt ^
random_search_lgbm_results.txt ^
random_search_selectk_results.txt ^
test_metrics.txt ^
test_metrics_best_pipeline.txt ^
test_metrics_lgbm.txt ^
test_metrics_selectk.txt ^
test_metrics_deploy.txt ^
model_pipeline_selectk.joblib

echo File yang akan dihapus:
echo.
for %%F in (%FILES_TO_DELETE%) do (
    echo   - %%F
)
echo.

set /p CONFIRM="Lanjutkan? (y/n): "
if /i not "!CONFIRM!"=="y" (
    echo Dibatalkan.
    exit /b 0
)

echo.
echo [1/4] Menghapus file-file...
for %%F in (%FILES_TO_DELETE%) do (
    if exist "%%F" (
        del /f /q "%%F"
        echo   Dihapus: %%F
    ) else (
        echo   Tidak ditemukan: %%F
    )
)

echo.
echo [2/4] Git add (stage changes)...
git add -A
if errorlevel 1 (
    echo ERROR: Gagal menjalankan git add
    pause
    exit /b 1
)

echo [3/4] Git commit...
git commit -m "Clean up: remove experiment result files"
if errorlevel 1 (
    echo ERROR: Gagal melakukan commit
    pause
    exit /b 1
)

echo [4/4] Git push...
git push origin main
if errorlevel 1 (
    echo ERROR: Gagal push
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! Repository sudah dibersihkan.
echo File-file sisa telah dihapus dan di-push ke GitHub.
echo ============================================
pause
