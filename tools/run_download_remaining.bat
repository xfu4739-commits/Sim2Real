@echo off
setlocal
cd /d "%~dp0.."

echo Source:      G:\我的云端硬盘
echo Sim full:    C:\Users\10115\Datasets\Sim2Real-Fire\sim_full
echo Destination: C:\Users\10115\Datasets\Sim2Real-Fire\sim_archives_remaining
echo.

python tools\download_remaining_archives.py --execute --robocopy --smallest-first --workers 4
pause
