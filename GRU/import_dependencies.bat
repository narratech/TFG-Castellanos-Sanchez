@echo off
setlocal

REM -----------------------------
REM 1) Nombre del entorno virtual
REM -----------------------------
set VENV_NAME=venv

REM -----------------------------
REM 2) Crear el entorno si no existe
REM -----------------------------
if not exist "%VENV_NAME%\Scripts\python.exe" (
    echo Creando entorno virtual...
    python -m venv %VENV_NAME%
) else (
    echo Entorno virtual ya existe.
)

REM -----------------------------
REM 3) Activar entorno
REM -----------------------------
call "%VENV_NAME%\Scripts\activate.bat"

REM -----------------------------
REM 4) Comprobar e instalar pandas
REM -----------------------------
echo import importlib.util> temp_check.py
echo import sys>> temp_check.py
echo import subprocess>> temp_check.py
echo def install(pkg):>> temp_check.py
echo     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])>> temp_check.py
echo def is_installed(pkg):>> temp_check.py
echo     return importlib.util.find_spec(pkg) is not None>> temp_check.py
echo if not is_installed("pandas"):>> temp_check.py
echo     print("Instalando pandas...")>> temp_check.py
echo     install("pandas")>> temp_check.py
echo else:>> temp_check.py
echo     print("pandas ya instalado.")>> temp_check.py

python temp_check.py
if ERRORLEVEL 1 (
    echo Error instalando pandas. Pulsa Intro para salir.
    pause
    del temp_check.py
    exit /b 1
)

del temp_check.py

REM -----------------------------
REM 5) Comprobar e instalar torch
REM -----------------------------
echo import importlib.util> temp_check.py
echo import sys>> temp_check.py
echo import subprocess>> temp_check.py
echo def install(pkg):>> temp_check.py
echo     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])>> temp_check.py
echo def is_installed(pkg):>> temp_check.py
echo     return importlib.util.find_spec(pkg) is not None>> temp_check.py
echo if not is_installed("torch"):>> temp_check.py
echo     print("Instalando torch...")>> temp_check.py
echo     install("torch")>> temp_check.py
echo else:>> temp_check.py
echo     print("torch ya instalado.")>> temp_check.py

python temp_check.py
if ERRORLEVEL 1 (
    echo Error instalando torch. Pulsa Intro para salir.
    pause
    del temp_check.py
    exit /b 1
)

del temp_check.py

REM -----------------------------
REM 6) Comprobar e instalar matplotlib
REM -----------------------------
echo import importlib.util> temp_check.py
echo import sys>> temp_check.py
echo import subprocess>> temp_check.py
echo def install(pkg):>> temp_check.py
echo     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])>> temp_check.py
echo def is_installed(pkg):>> temp_check.py
echo     return importlib.util.find_spec(pkg) is not None>> temp_check.py
echo if not is_installed("matplotlib"):>> temp_check.py
echo     print("Instalando matplotlib...")>> temp_check.py
echo     install("matplotlib")>> temp_check.py
echo else:>> temp_check.py
echo     print("matplotlib ya instalado.")>> temp_check.py

python temp_check.py
if ERRORLEVEL 1 (
    echo Error instalando matplotlib. Pulsa Intro para salir.
    pause
    del temp_check.py
    exit /b 1
)
del temp_check.py


REM -----------------------------
REM 7) Comprobar e instalar scikit-learn
REM -----------------------------
echo import importlib.util> temp_check.py
echo import sys>> temp_check.py
echo import subprocess>> temp_check.py
echo def install(pkg):>> temp_check.py
echo     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])>> temp_check.py
echo def is_installed(pkg):>> temp_check.py
echo     return importlib.util.find_spec(pkg) is not None>> temp_check.py
echo if not is_installed("sklearn"):>> temp_check.py
echo     print("Instalando scikit-learn...")>> temp_check.py
echo     install("scikit-learn")>> temp_check.py
echo else:>> temp_check.py
echo     print("scikit-learn ya instalado.")>> temp_check.py

python temp_check.py
if ERRORLEVEL 1 (
    echo Error instalando scikit-learn. Pulsa Intro para salir.
    pause
    del temp_check.py
    exit /b 1
)
del temp_check.py

REM -----------------------------
REM 8) Comprobar e instalar onnx
REM -----------------------------
echo import importlib.util> temp_check.py
echo import sys>> temp_check.py
echo import subprocess>> temp_check.py
echo def install(pkg):>> temp_check.py
echo     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])>> temp_check.py
echo def is_installed(pkg):>> temp_check.py
echo     return importlib.util.find_spec(pkg) is not None>> temp_check.py
echo if not is_installed("onnx"):>> temp_check.py
echo     print("Instalando onnx...")>> temp_check.py
echo     install("onnx")>> temp_check.py
echo else:>> temp_check.py
echo     print("onnx ya instalado.")>> temp_check.py

python temp_check.py
if ERRORLEVEL 1 (
    echo Error instalando onnx. Pulsa Intro para salir.
    pause
    del temp_check.py
    exit /b 1
)

del temp_check.py


REM -----------------------------
REM 9) Comprobar e instalar onnxscript
REM -----------------------------
echo import importlib.util> temp_check.py
echo import sys>> temp_check.py
echo import subprocess>> temp_check.py
echo def install(pkg):>> temp_check.py
echo     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])>> temp_check.py
echo def is_installed(pkg):>> temp_check.py
echo     return importlib.util.find_spec(pkg) is not None>> temp_check.py
echo if not is_installed("onnxscript"):>> temp_check.py
echo     print("Instalando onnxscript...")>> temp_check.py
echo     install("onnxscript")>> temp_check.py
echo else:>> temp_check.py
echo     print("onnxscript ya instalado.")>> temp_check.py

python temp_check.py
if ERRORLEVEL 1 (
    echo Error instalando onnxscript. Pulsa Intro para salir.
    pause
    del temp_check.py
    exit /b 1
)

del temp_check.py

echo Entorno de python creado
pause
endlocal