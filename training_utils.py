import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.2, shuffle: bool = False, sim_type: str = ''):
    
    trainval_df, test_df = train_test_split(df, train_size = train_ratio, random_state=42, shuffle=shuffle)
    train_df, val_df = train_test_split(trainval_df, test_size = val_ratio, random_state=42, shuffle=shuffle)

    train_df.to_csv('data/training/train_' + sim_type + '.csv', index=False)
    val_df.to_csv('data/training/val_' + sim_type + '.csv', index=False)
    test_df.to_csv('data/training/test_' + sim_type + '.csv', index=False)

    return train_df, val_df, test_df


def main():

    files = [
        'double_pendulum_lowfrict.csv',
        'double_pendulum_nofrict.csv',
        'double_pendulum_highfrict.csv']
    
    for f in files:
        print(f"Splitting dataset for {f}...")
        df = pd.read_csv('data/processed/' + f)
        sim_type = f.split('_')[2].split('.')[0]
        split_dataset(df, sim_type=sim_type, shuffle=True)
    

if __name__ == "__main__":
    main()