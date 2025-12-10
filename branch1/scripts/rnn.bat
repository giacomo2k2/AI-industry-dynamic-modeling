@echo off

python -m branch1.main.rnn ^
    --train_path "branch1/data_enhanced/training/train_nofrict.csv" ^
    --val_path "branch1/data_enhanced/training/val_nofrict.csv" ^
    --test_path "branch1/data_enhanced/training/test_nofrict.csv" ^
    --model_path "branch1/models/integrator/" ^
    --model_name "integrator_model.pth" ^
    --version "NeuralUDEv4_nofrict" ^
    --epochs 40 ^
    --patience 3 ^
    --lr 0.01 ^
    --wd 0.0 ^
    --batch_size 256 ^
    --optimizer "adam" ^
    --l1_ref 1.0 ^
    --l2_ref 5.0 ^
    --m1_ref 1.0 ^
    --m2_ref 1.0 ^
    --b1_ref 1.0 ^
    --b2_ref 0.5 ^
    --training ^
    --verbose ^
    --show_plots ^
    --fixed_l1 ^
    --fixed_l2 ^
    --fixed_m2 ^
    --fixed_m1 


pause