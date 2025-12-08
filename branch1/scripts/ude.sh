python -m branch1.ude_main \
    --train_path branch1/data_enhanced/training/train_lowfrict.csv \
    --val_path branch1/data_enhanced/training/val_lowfrict.csv \
    --test_path branch1/data_enhanced/training/test_lowfrict.csv \
    --model_path branch1/models/deepNN/best_deepNN.pth \
    --epochs 500 \
    --lr 0.001 \
    --batch_size 512 \
    --wd 0.0001 \
    --patience 15 \
    --verbose \
    --show_plots \
    # --training \