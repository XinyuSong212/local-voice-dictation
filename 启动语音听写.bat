@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动语音听写工具...
python voice_input.py
pause
