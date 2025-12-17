@echo off
REM ==========================================
REM  Windows Launcher
REM ==========================================

set SCRIPT_NAME=branch1.main.onestep

echo Avvio onestep.py ...

python -m %SCRIPT_NAME% ^
    --train_path "branch1/data_enhanced/training_EC_WN/train_lowfrict.csv" ^
    --val_path "branch1/data_enhanced/training_EC_WN/val_lowfrict.csv" ^
    --test_path "branch1/data_enhanced/training_EC_WN/test_lowfrict.csv" ^
    --verbose ^
    --patience 50 ^
    --wd 1e-4 ^
    --batch_size 256 ^
    --optimizer_type "adam" ^
    --scheduler_type "onecycle" ^
    --loss_type "mse" ^
    --epochs 500 ^
    --lr 1e-3 ^
    --integrator "rk4" ^
    --dynlayer "ude_d_constant" ^
    --activation "softplus" ^
    --version "v_220_Matlearn" ^
    --sin_cos_transform ^
    --sample_weights ^
    --training

REM Controllo errori
if %errorlevel% neq 0 (
    echo.
    echo [ERRORE] Lo script ha riscontrato un problema.
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Script terminato correttamente.