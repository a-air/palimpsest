@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
python refiner.py score --raw examples\raw_2026-03_delivery.md --node examples\node_example.json
echo.
pause