import argparse
import os
import sys
import pandas as pd
from typing import Tuple
    

def main(
    input_csv: str,
    output_dir: str,
    tag: str,
    split_ratios: Tuple[float, float, float],
):

    df = pd.read_csv(input_csv)

    # split the id_pendulum column into train, val, test based on split_ratios
    unique_ids = df['id_pendulum'].unique()
    n_total = len(unique_ids)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total) 
    n_test = n_total - n_train - n_val

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]

    train_df = df[df['id_pendulum'].isin(train_ids)]
    print(f"Train IDs ({len(train_ids)})")

    val_df = df[df['id_pendulum'].isin(val_ids)]
    print(f"Validation IDs ({len(val_ids)})")

    test_df = df[df['id_pendulum'].isin(test_ids)]
    print(f"Test IDs ({len(test_ids)})")

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, f"train_{tag}.csv"), index=False)
    print(f"Saved train_{tag}.csv with {len(train_df)} rows.")

    val_df.to_csv(os.path.join(output_dir, f"val_{tag}.csv"), index=False)
    print(f"Saved val_{tag}.csv with {len(val_df)} rows.")

    test_df.to_csv(os.path.join(output_dir, f"test_{tag}.csv"), index=False)
    print(f"Saved test_{tag}.csv with {len(test_df)} rows.")
    

if __name__ == "__main__":

    csv_paths = [
        ("branch1/data_enhanced/processed_EC_WN/double_pendulum_highfrict.csv", "highfrict"),
        ("branch1/data_enhanced/processed_EC_WN/double_pendulum_lowfrict.csv", "lowfrict"),
        ("branch1/data_enhanced/processed_EC_WN/double_pendulum_nofrict.csv", "nofrict"),
    ]

    output_dir = "branch1/data_enhanced/training_EC_WN"
    split_ratios = (0.64, 0.16, 0.20)

    for csv_path, tag in csv_paths:
        main(
            csv_path,
            output_dir,
            tag,
            split_ratios,
        )