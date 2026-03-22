@echo off
setlocal

REM -----------------------------
REM 1) Nombre del entorno virtual
REM -----------------------------
set VENV_NAME=venv

REM -----------------------------
REM 2) Comprobar si el entorno no existe
REM -----------------------------
if not exist "%VENV_NAME%\Scripts\python.exe" (
    echo El entorno no exite, ejecuta import_dependencies.bat
    pause
    exit /b 1
)

REM -----------------------------
REM 3) Activar entorno
REM -----------------------------
call "%VENV_NAME%\Scripts\activate.bat"

REM -----------------------------
REM 4) Ejecutar autoencoder.py
REM -----------------------------
echo Ejecutando autoencoder.py...
python -u src/autoencoder.py

if ERRORLEVEL 1 (
    echo Error ejecutando autoencoder.py. Pulsa Intro para salir.
    pause
    exit /b 1
)

echo.
echo Secuencias sinteticas generadas correctamente.

REM -----------------------------
REM 5) Ejecutar gru_train_from_csv.py
REM -----------------------------
echo Ejecutando gru_train_from_csv.py...
python -u src/gru_train_from_csv.py
if ERRORLEVEL 1 (
    echo Error ejecutando gru_train_from_csv.py. Pulsa Intro para salir.
    pause
    exit /b 1
)

echo.
echo Modelo creado correctamente.
pause
endlocal