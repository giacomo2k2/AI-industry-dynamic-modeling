@echo off
python -m branch1.main ^
    --train_path data/training/train_lowfrict.csv ^
    --val_path data/training/val_lowfrict.csv
