
@echo off
REM ==========================================
REM  Windows Launcher
REM ==========================================

set SCRIPT_NAME=branch1.main.rnn

echo Avvio rnn.py ...

python -m %SCRIPT_NAME% ^
    --train_path "branch1/data_enhanced/training_EC_WN/train_lowfrict.csv" ^
    --val_path "branch1/data_enhanced/training_EC_WN/val_lowfrict.csv" ^
    --test_path "branch1/data_enhanced/training_EC_WN/test_lowfrict.csv" ^
    --verbose ^
    --patience 20 ^
    --wd 1e-5 ^
    --batch_size 1024 ^
    --optimizer_type "adam" ^
    --scheduler_type "onecycle" ^
    --loss_type "mae" ^
    --curriculum_steps 2 ^
    --epochs_per_stage 2 ^
    --lrs 5e-3 ^
    --integrator "rk4" ^
    --dynlayer "nn" ^
    --activation "softplus" ^
    --fixed_l1 ^
    --fixed_l2 ^
    --fixed_m1 ^
    --fixed_m2 ^
    --fixed_b1 ^
    --version "v169NN" ^
    --sin_cos_transform ^
    --sample_weights

REM Controllo errori
if %errorlevel% neq 0 (
    echo.
    echo [ERRORE] Lo script ha riscontrato un problema.
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Script terminato correttamente.
