@echo off
REM Local AI Assistant - Masaustu uygulamasini baslatir.
REM Cift tiklayarak calistirabilirsiniz.
cd /d "%~dp0"
start "" "venv\Scripts\pythonw.exe" -m src.desktop
