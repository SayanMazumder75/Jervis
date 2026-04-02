@echo off
echo ============================================================
echo   J.A.R.V.I.S  --  Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not installed or not in PATH.
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [1/4] Python found.

:: Install pip packages
echo  [2/4] Installing packages...
pip install SpeechRecognition pyttsx3 pyautogui psutil

:: Try PyAudio normally first
echo  [3/4] Installing PyAudio...
pip install pyaudio
if errorlevel 1 (
    echo  PyAudio direct install failed. Trying pipwin...
    pip install pipwin
    pipwin install pyaudio
)

echo  [4/4] Creating required folders...
if not exist "core" mkdir core
if not exist "ui"   mkdir ui
if not exist "logs" mkdir logs

:: Create __init__.py files if missing
if not exist "core\__init__.py" type nul > "core\__init__.py"
if not exist "ui\__init__.py"   type nul > "ui\__init__.py"

echo.
echo ============================================================
echo   Setup complete!  Run JARVIS with:
echo     python main.py
echo ============================================================
echo.
pause