import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from data_inspection import show_bins, show_status_stats, show_energy_stats


PLOT = True

# ========================================================
# ================== MODIFY THE DATASET ==================
# ========================================================


def round_to_pi(df: pd.DataFrame):
    """
    Wrap angles to the interval [-pi, pi]
    """
    # ATTENTION: watchout the module, in NN it may not work.
    o = df[['o1', 'o2']].to_numpy() + np.pi
    o = np.mod(o, 2*np.pi) - np.pi
    df['o1'], df['o2'] = o[:,0], o[:,1]

    return df

def energy_check(df, tol=1e-2):
    # Compute energy differences per pendulum in one shot
    df = df.sort_values(['id_pendulum']).copy()

    # Group-wise difference
    df['dE'] = df.groupby('id_pendulum')['E'].diff()

    # Identify pendulums whose energy difference is greater than the positive threshold
    bad = df.loc[df['dE'] > tol, 'id_pendulum'].unique()
    
    # Filter valid rows exluding the ones that are in the bad list
    valid_df = df[~df['id_pendulum'].isin(bad)].copy()
    

    # Remap IDs 0..N-1 compactly
    if not valid_df.empty:
        id_map = {old: new for new, old in enumerate(valid_df['id_pendulum'].unique())}
        valid_df['id_pendulum'] = valid_df['id_pendulum'].map(id_map)
    else:
        valid_df = df.drop(df.index)  # keep columns, empty
    
    return valid_df.drop(columns='dE'), list(bad)

def cut_dataset(df:pd.DataFrame, threshold:float):
    """
    Keep onlty the time steps lower than a time threshold
    """
    
    return df[df['time_step'] < threshold]



def kullback_leibler(p, q, eps=1e-10):
    """
    Compute the Kullback-Leibler divergence D_KL(P || Q) between two probability distributions.
    
    ---
    Args:
        p (np.ndarray): First probability distribution.
        q (np.ndarray): Second probability distribution.
        eps (float): Small value to avoid log(0).
    """
    p = p + eps
    q = q + eps
    return np.sum(p * np.log(p / q))

def optimize_threshold(df:pd.DataFrame ):
    """
    Optimize the threshold based on Kullback-Leibler divergence between the 
    distribution of binned angles and a uniform distribution.
    
    ---
    Args:
        distr_bins (np.ndarray): Histogram bins of the angle distributions.
        uniform_bins (np.ndarray): Histogram bins of a uniform distribution.
    """
    best_threshold = None
    best_kl_divergence = float('inf')
    kl_divergences = []
    thresholds_vec = np.linspace(0.0, 10.0, 1000)

    for threshold in thresholds_vec:
        df_temp = cut_dataset (df, threshold=threshold)
        
        # produce bins
        states = df_temp[['o1', 'o1_dot', 'o2', 'o2_dot']].to_numpy()
        distr_bins, _ = np.histogram(states[:, 3], bins=50, range=(-np.pi, np.pi), density=True)
        uniform_bins = np.ones_like(distr_bins) / distr_bins.size

        kl_divergence = kullback_leibler(distr_bins, uniform_bins)
        
        kl_divergences.append(kl_divergence)

        if kl_divergence < best_kl_divergence:
            best_kl_divergence = kl_divergence
            best_threshold = threshold

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_vec, kl_divergences)
    plt.title('Kullback-Leibler Divergence vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Kullback-Leibler Divergence')
    plt.grid()
    plt.tight_layout()
    plt.show()

    return best_threshold

# ========================================================
# ================== MAIN PROCESSING =====================
# ========================================================

def process_data(path:str, threshold, plot: bool = PLOT):
    
    sim_type = path.split('_')[2].split('.')[0]
    print(f"Processing {sim_type} data...")
    
    # Read the csv in the data raw folder
    df = pd.read_csv(f'data/raw/{path}')

    
    processed_df, discarded = energy_check(df)

    processed_df = round_to_pi(processed_df)

    if PLOT:
        show_bins(processed_df, id=-1, save=True, filename=sim_type)

        show_status_stats(processed_df, filename=sim_type)   

        show_energy_stats(processed_df, filename=sim_type)

    threshold = optimize_threshold(processed_df)
    print(f"Optimal threshold for {sim_type}: {threshold}")
    processed_df = cut_dataset(processed_df, threshold=threshold)


    if PLOT:
        show_bins(processed_df, id=-1, save=False)

    # Save processed dataframe to csv
    #processed_df.to_csv(f'data/processed/{path}', index=False)

    # Print process terminated and number of pendulums kept
    n_total = df['id_pendulum'].nunique()
    n_kept = processed_df['id_pendulum'].nunique()
    print(f"Processing terminated. Kept {n_kept} out of {n_total} pendulums.")

    return processed_df


# ========================================================
# ================== RUN THE PROCESSING ==================
# ========================================================

def main():
    
    thresholds = [
        10.0, 
        6.0, 
        2.0
    ]

    df_nof = process_data('double_pendulum_nofrict.csv', threshold = thresholds[0])
    df_lowf = process_data('double_pendulum_lowfrict.csv', threshold = thresholds[1])
    df_highf = process_data('double_pendulum_highfrict.csv', threshold = thresholds[2])

if __name__ == '__main__':
    main()
  