@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo Atualizacao local dos analytics Safety Chat
echo ==========================================
echo.

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" tools\update_analytics_local.py
) else (
  python tools\update_analytics_local.py
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Processo concluido. Revise as alteracoes no GitHub Desktop e faca commit/push.
) else (
  echo Processo falhou. Os artefatos atuais foram preservados ou restaurados.
)
echo.
pause
exit /b %EXIT_CODE%
