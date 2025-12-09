@echo off
REM Script di ude_main.py

setlocal enabledelayedexpansion

REM Configurazione percorsi dati
set "train_path=branch1/data_enhanced/training/train_lowfrict.csv"
set "val_path=branch1/data_enhanced/training/val_lowfrict.csv"
set "test_path=branch1/data_enhanced/training/test_lowfrict.csv"

REM Configurazione training
set "epochs=500"
set "patience=10"
set "lr=1e-3"
set "wd=1e-4"
set "batch_size=512"

REM Configurazione output
set "model_path=branch1/models/deepNN/best_deepNN.pth"

REM Esegui lo script Python
python -m branch1.ude_main ^
    --train_path !train_path! ^
    --val_path !val_path! ^
    --test_path !test_path! ^
    --epochs !epochs! ^
    --patience !patience! ^
    --lr !lr! ^
    --wd !wd! ^
    --batch_size !batch_size! ^
    --model_path !model_path! ^
    --verbose ^
    --show_plots 
    REM --training ^