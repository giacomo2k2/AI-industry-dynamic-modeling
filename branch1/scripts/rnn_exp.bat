@echo off
setlocal

:: --- CONFIGURAZIONE COMUNE ---
set SCRIPT_PATH=branch1.main.rnn
set COMMON_ARGS=--epochs 100 --patience 15 --lr 0.01 --batch_size 512 --wd 0.0 --optimizer adam --verbose --training

:: Directory base dei dati
set DATA_DIR=branch1/data_enhanced/training

:: Definisco i gruppi di flag per fissare i parametri
:: 1. Qui fisso tutto TRANNE l1 (quindi imparo solo l1)
set FLAGS_ONLY_L1=--fixed_l2 --fixed_m1 --fixed_m2 --fixed_b1 --fixed_b2

:: 2. Qui fisso tutto TRANNE b1 e b2 (quindi imparo solo b1 e b2)
set FLAGS_LEARN_B1_B2=--fixed_l1 --fixed_l2 --fixed_m1 --fixed_m2

:: 3. Learning only b1
set FLAGS_ONLY_B1=--fixed_l1 --fixed_l2 --fixed_m1 --fixed_m2 --fixed_b2

:: 4. Learning m1 and l2
set FLAGS_LEARN_M1_L2=--fixed_l1 --fixed_m2 --fixed_b1 --fixed_b2

echo ==========================================
echo AVVIO SUITE ESPERIMENTI MULTI-DATASET
echo ==========================================

:: --- PARTE 1: Imparare solo L1 su 3 dataset diversi ---

:: 1.1 Dataset NO Friction
echo.
echo "Dataset: NO Friction | Target: Learn L1"
python -m %SCRIPT_PATH% ^
    %COMMON_ARGS% ^
    --train_path "%DATA_DIR%/train_nofrict.csv" ^
    --val_path "%DATA_DIR%/val_nofrict.csv" ^
    --test_path "%DATA_DIR%/test_nofrict.csv" ^
    --version "v_nofrict_only_b1" ^
    --model_name "model.pth" ^
    %FLAGS_ONLY_B1%

:: 1.2 Dataset LOW Friction
echo.
echo "Dataset: LOW Friction | Target: Learn L1"
python -m %SCRIPT_PATH% ^
    %COMMON_ARGS% ^
    --train_path "%DATA_DIR%/train_lowfrict.csv" ^
    --val_path "%DATA_DIR%/val_lowfrict.csv" ^
    --test_path "%DATA_DIR%/test_lowfrict.csv" ^
    --version "v_lowfrict_only_l1" ^
    --model_name "model.pth" ^
    %FLAGS_ONLY_L1%

:: 1.3 Dataset HIGH Friction
echo.
echo "Dataset: HIGH Friction | Target: Learn L1"
python -m %SCRIPT_PATH% ^
    %COMMON_ARGS% ^
    --train_path "%DATA_DIR%/train_highfrict.csv" ^
    --val_path "%DATA_DIR%/val_highfrict.csv" ^
    --test_path "%DATA_DIR%/test_highfrict.csv" ^
    --version "v_highfrict_only_l1" ^
    --model_name "model.pth" ^
    %FLAGS_ONLY_L1%


:: --- PARTE 2: Imparare più di un parametro su Low Friction ---

echo.
echo "Dataset: LOW Friction | Target: Learn b1 & b2"
python-m %SCRIPT_PATH% ^
    %COMMON_ARGS% ^
    --train_path "%DATA_DIR%/train_lowfrict.csv" ^
    --val_path "%DATA_DIR%/val_lowfrict.csv" ^
    --test_path "%DATA_DIR%/test_lowfrict.csv" ^
    --version "v_lowfrict_learn_b1b2" ^
    --model_name "model_friction.pth" ^
    %FLAGS_LEARN_B1_B2%

echo.
echo "Dataset: LOW Friction | Target: Learn m1 & l2"
python-m %SCRIPT_PATH% ^
    %COMMON_ARGS% ^
    --train_path "%DATA_DIR%/train_lowfrict.csv" ^
    --val_path "%DATA_DIR%/val_lowfrict.csv" ^
    --test_path "%DATA_DIR%/test_lowfrict.csv" ^
    --version "v_lowfrict_learn_m1l2" ^
    --model_name "model_friction.pth" ^
    %FLAGS_LEARN_M1_L2%

echo.
echo "Dataset: LOW Friction | Target: Learn all params"
python-m %SCRIPT_PATH% ^
    %COMMON_ARGS% ^
    --train_path "%DATA_DIR%/train_lowfrict.csv" ^
    --val_path "%DATA_DIR%/val_lowfrict.csv" ^
    --test_path "%DATA_DIR%/test_lowfrict.csv" ^
    --version "v_lowfrict_learn_all" ^
    --model_name "model_friction.pth" ^

echo.
echo ==========================================
echo TUTTI GLI ESPERIMENTI COMPLETATI
echo ==========================================
pause