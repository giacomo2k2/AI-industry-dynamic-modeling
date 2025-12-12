@echo off
setlocal EnableDelayedExpansion

REM ========================================================
REM  AUTOMATED EXPERIMENT RUNNER (ROBUST MODE)
REM ========================================================

set SCRIPT_NAME=branch1.main.rnn
set /a fail_count=0
set /a total_runs=0

REM --- PARAMETRI FISSI ---
set COMMON_ARGS=--train_path "branch1/data_enhanced/training/train_nofrict.csv" --val_path "branch1/data_enhanced/training/val_nofrict.csv" --test_path "branch1/data_enhanced/training/test_nofrict.csv" --training --verbose --patience 15 --wd 1e-5 --loss_type mse --optimizer_type adam --scheduler_type plateau --integrator rk4 --activation softplus --fixed_m1 --fixed_m2 
REM --- LOOP PRINCIPALE ---
FOR %%L in (ode) DO (
    
    set /a version_counter=80
    
    echo.
    echo ========================================================
    echo  STARTING BATCH FOR DYNAMIC LAYER: %%L
    echo ========================================================

    REM --- LOOP STEPS ---
    FOR %%C in ("4 8 16 32") DO (
        
        REM --- LOOP LR ---
        FOR %%R in ("1e-2 5e-3 5e-3 1e-3") DO (
            
            REM --- LOOP EPOCHS ---
            FOR %%E in ("30 20 10 5") DO (
                
                REM --- LOOP BATCH SIZE ---
                FOR %%B in (512 1024) DO (
                    
                    FOR %%S in (true false) DO (
                        if "%%S"=="true" (set SINCOS_FLAG=--sin_cos_transform) else (set SINCOS_FLAG=)

                        REM Logica Norm (solo per NN)
                        set NORM_CASES=false 
                        if "%%L"=="nn" set NORM_CASES=false true

                        FOR %%N in (!NORM_CASES!) DO (
                            if "%%N"=="true" (set NORM_FLAG=--normalization) else (set NORM_FLAG=)

                            set "CURRENT_VER=v!version_counter!"
                            set /a total_runs+=1
                            
                            echo.
                            echo [RUN !total_runs!] Layer: %%L - Ver: !CURRENT_VER! - Batch: %%B - SinCos: %%S - Norm: %%N
                            
                            REM ESECUZIONE
                            python -m %SCRIPT_NAME% ^
                                %COMMON_ARGS% ^
                                --dynlayer %%L ^
                                --version "!CURRENT_VER!" ^
                                --curriculum_steps %%~C ^
                                --lrs %%~R ^
                                --epochs_per_stage %%~E ^
                                --batch_size %%B ^
                                !SINCOS_FLAG! ^
                                !NORM_FLAG!

                            REM CHECK ERRORE (Senza uscire)
                            if !errorlevel! neq 0 (
                                echo.
                                echo [WARNING] Run failed! Logging error and continuing...
                                set /a fail_count+=1
                            ) else (
                                echo [SUCCESS] Run completed.
                            )

                            set /a version_counter+=1
                        )
                    )
                )
            )
        )
    )
)

echo.
echo ========================================================
echo  BATCH COMPLETED.
echo  Total Runs: !total_runs!
echo  Failures:   !fail_count!
echo  Check 'batch_errors_log.txt' for details.
echo ========================================================
pause