
@echo off
REM ==========================================
REM  Windows Launcher
REM ==========================================

set SCRIPT_NAME=branch1.main.rnn

echo Avvio rnn.py ...

python -m %SCRIPT_NAME% ^
    --train_path "branch1/data_enhanced/training/train_lowfrict.csv" ^
    --val_path "branch1/data_enhanced/training/val_lowfrict.csv" ^
    --test_path "branch1/data_enhanced/training/test_lowfrict.csv" ^
    --verbose ^
    --patience 5 ^
    --wd 1e-5 ^
    --batch_size 256 ^
    --optimizer_type "adam" ^
    --scheduler_type "plateau" ^
    --loss_type "mse" ^
    --curriculum_steps 2 4 ^
    --epochs_per_stage 2 2 ^
    --lrs 1e-3 5e-4 ^
    --integrator "rk4" ^
    --dynlayer "nn" ^
    --activation "softplus" ^
    --fixed_l1 ^
    --fixed_l2 ^
    --fixed_m1 ^
    --fixed_m2 ^
    --fixed_b1 ^
    --version "exp_norm" ^
    --sin_cos_transform ^
    --normalization ^
    --training

REM Controllo errori
if %errorlevel% neq 0 (
    echo.
    echo [ERRORE] Lo script ha riscontrato un problema.
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Script terminato correttamente.
