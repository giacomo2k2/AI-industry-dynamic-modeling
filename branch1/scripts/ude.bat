
@echo off
REM ==========================================
REM  Windows Launcher
REM ==========================================

set SCRIPT_NAME=branch1.main.ude

echo Avvio ude.py ...

python -m %SCRIPT_NAME% ^
    --train_path "branch1/data_enhanced/training_EC_WN/train_lowfrict.csv" ^
    --val_path "branch1/data_enhanced/training_EC_WN/val_lowfrict.csv" ^
    --test_path "branch1/data_enhanced/training_EC_WN/test_lowfrict.csv" ^
    --patience 15 ^
    --wd 1e-5 ^
    --batch_size 32 ^
    --optimizer_type "adam" ^
    --scheduler_type "plateau" ^
    --loss_type "mse" ^
    --epochs 2000 ^
    --lr 1e-3 ^
    --activation "softplus" ^
    --version "v1000wn" ^
    --sin_cos_transform ^
    --verbose ^
    --show_plots ^
    --training

REM Controllo errori
if %errorlevel% neq 0 (
    echo.
    echo [ERRORE] Lo script ha riscontrato un problema.
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Script terminato correttamente.
