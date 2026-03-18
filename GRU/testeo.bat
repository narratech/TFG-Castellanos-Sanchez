@echo off
setlocal

REM -----------------------------
REM 1) Nombre del entorno virtual
REM -----------------------------
set VENV_NAME=venv

REM -----------------------------
REM 2) Activar entorno
REM -----------------------------
call "%VENV_NAME%\Scripts\activate.bat"

REM -----------------------------
REM 3) Ejecutar autoencoder.py
REM -----------------------------
REM echo Ejecutando autoencoder.py...
REM python -u src/autoencoder.py --csv realset.csv --epochs 500

if ERRORLEVEL 1 (
    echo Error ejecutando autoencoder.py. Pulsa Intro para salir.
    pause
    exit /b 1
)

REM -----------------------------
REM 4) Ejecutar gru_tester.py
REM -----------------------------
echo Ejecutando gru_tester.py...
python -u src/gru_tester.py --csv realset.csv --onehot True

if ERRORLEVEL 1 (
    echo Error ejecutando gru_tester.py. Pulsa Intro para salir.
    pause
    exit /b 1
)

pause
endlocal