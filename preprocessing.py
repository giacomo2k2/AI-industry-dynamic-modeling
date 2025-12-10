import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from data_inspection import show_bins, show_status_stats, show_energy_stats

# Set global plot style
plt.rc('font', family='serif')

# Global plotting flag
PLOT = True

# ========================================================
# ================== MODIFY THE DATASET ==================
# ========================================================


def round_to_pi(df: pd.DataFrame):
    """
    Wrap angles to the interval [-pi, pi].

    ---
    Args:
        df (pd.DataFrame): Input dataframe containing angle data.

    ---
    Returns:
        pd.DataFrame: Dataframe with angles wrapped to [-pi, pi].
    """
    # ATTENTION: watchout the module, in NN it may not work.
    q = df[['q1', 'q2']].to_numpy() + np.pi
    q = np.mod(q, 2*np.pi) - np.pi
    df['q1'], df['q2'] = q[:,0], q[:,1]

    return df

def energy_check(df, tol=1e-2):
    """
    Keep only energy-conservative pendulums.

    ---
    Args:
        df (pd.DataFrame): Input dataframe containing energy data.
        tol (float, optional): Tolerance threshold for energy difference.
    
    ---
    Returns:
        pd.DataFrame: Filtered dataframe with only energy-conservative pendulums.
        list: List of discarded pendulum IDs.
    """
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

def cut_dataset(df:pd.DataFrame, threshold:float) -> pd.DataFrame:
    """
    Keep onlty the time steps lower than a time threshold.

    ---
    Args:
        df (pd.DataFrame): Input dataframe containing time steps.
        threshold (float): Time threshold to cut the dataset.
    
    ---
    Returns:
        pd.DataFrame: Dataframe filtered by the time threshold.
    """
    
    return df[df['time_step'] < threshold]



def kullback_leibler(p, q, eps=1e-10, lambda_val=1.0, threshold_penalty = 0):
    """
    Compute the Kullback-Leibler divergence D_KL(P || Q) between two probability distributions.
    
    ---
    Args:
        p (np.ndarray): real probability distribution.
        q (np.ndarray): approximated probability distribution.
        eps (float, optional): Small value to avoid log(0).
        lambda_val (float, optional): Weight for the threshold penalty term.
        threshold_penalty (float, optional): Penalty term based on the threshold.

    ---
    Returns:
        float: Kullback-Leibler divergence with penalty term.
    """
    p = p + eps
    q = q + eps

    return np.sum(p * np.log(p / q)) + lambda_val * threshold_penalty**2

def optimize_threshold(df:pd.DataFrame) -> float:
    """
    Optimize the threshold based on Kullback-Leibler divergence between the 
    distribution of binned angles and a uniform distribution.
    
    ---
    Args:
        distr_bins (np.ndarray): Histogram bins of the angle distributions.
        uniform_bins (np.ndarray): Histogram bins of a uniform distribution.
    
    ---
    Returns:
        float: Optimal threshold value that minimizes the KL divergence.
    """
    best_threshold = None
    best_kl_divergence = float('inf')
    kl_vector = []
    max_time = df['time_step'].to_numpy()[-1]
    thresholds_vec = np.linspace(0.0, max_time, num=100)



    for threshold in thresholds_vec:
        if threshold == 0.0:
            kl_vector.append(np.nan)
            continue
        
        df_temp = cut_dataset(df, threshold=threshold)

        # produce bins
        states = df_temp[['q1', 'q2']].to_numpy()
        kl_divergence = 0

        for i in range(states.shape[1]):
            distr_bins, _ = np.histogram(states[:, i], bins=200, range=(-np.pi, np.pi), density=True)
            uniform_bins = np.ones_like(distr_bins) / distr_bins.size
            q = distr_bins / distr_bins.sum()
            kl_divergence += kullback_leibler(q = q, p = uniform_bins, lambda_val= 0.5, threshold_penalty= (max_time - threshold)/max_time)
            
        kl_vector.append(kl_divergence)

        if kl_divergence < best_kl_divergence:
            best_kl_divergence = kl_divergence
            best_threshold = threshold



    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_vec, kl_vector, label='KL Divergence')
    plt.scatter(best_threshold, best_kl_divergence, linestyle='', marker='o', color='red', label='Optimal Threshold')
    plt.title('Kullback-Leibler Divergence vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Kullback-Leibler Divergence')
    plt.grid()
    plt.tight_layout()
    plt.show()

    return best_threshold

def add_white_noise(df: pd.DataFrame, std: float = 0.01) -> pd.DataFrame:
    """
    Add white Gaussian noise to the state variables of the dataframe.
    
    ---
    Args:
        df (pd.DataFrame): The input dataframe containing state variables.
        std (float): Standard deviation of the Gaussian noise to be added.

    ---
    Returns:
        pd.DataFrame: Dataframe with added noise to state variables.
    """
    noisy_df = df.copy()
    state_columns = ['q1', 'q2', 'q1_dot', 'q2_dot']
    
    for col in state_columns:
        noise = np.random.normal(0, std, size=noisy_df.shape[0])
        noisy_df[col] += noise
    
    return noisy_df

# ========================================================
# ================== MAIN PROCESSING =====================
# ========================================================

def process_data(path:str, 
                 plot: bool = PLOT,
                 raw_path: str = 'data/raw/',
                 processed_path: str = 'data/processed/') -> pd.DataFrame:
    
    sim_type = path.split('_')[2].split('.')[0]
    print(f"Processing {sim_type} data...")
    
    # Read the csv in the data raw folder
    df = pd.read_csv(f'{raw_path}{path}')

    
    processed_df, discarded = energy_check(df)
    processed_df = add_white_noise(processed_df)
    #processed_df = round_to_pi(processed_df)


    status_cols = ['q1', 'q1_dot', 'q1_dot_dot', 'q2', 'q2_dot', 'q2_dot_dot']
    status_names = [r'$q_1$', r'$\dot{q}_1$', r'$\ddot{q}_1$', r'$q_2$', r'$\dot{q}_2$',  r'$\ddot{q}_2$']
    vlim = processed_df[status_cols].abs().max().to_numpy()


    if plot:
        show_bins(processed_df, cols=status_cols, names=status_names, vlim=vlim, save=True, filename=sim_type)

        show_status_stats(processed_df, cols=status_cols, names=status_names, filename=sim_type)   

        show_energy_stats(processed_df, filename=sim_type)

    threshold = optimize_threshold(processed_df)

    print(f"Optimal threshold for {sim_type}: {threshold}")
    processed_df = cut_dataset(processed_df, threshold=threshold)


    if plot:
        show_bins(processed_df, cols=status_cols, names=status_names, vlim=vlim, save=False)

    # Save processed dataframe to csv
    processed_df.to_csv(f'{processed_path}{path}', index=False)

    # Print process terminated and number of pendulums kept
    n_total = df['id_pendulum'].nunique()
    n_kept = processed_df['id_pendulum'].nunique()
    print(f"Processing terminated. Kept {n_kept} out of {n_total} pendulums.")

    return processed_df

# ========================================================
# ================== RUN THE PROCESSING ==================
# ========================================================

def main():
    

    # df_nof = process_data(path = 'double_pendulum_nofrict.csv', 
    #                       raw_path= 'branch1/data_enhanced/raw/',
    #                       processed_path= 'branch1/data_enhanced/processed/')
    # df_lowf = process_data(path = 'double_pendulum_lowfrict.csv',
    #                        raw_path= 'branch1/data_enhanced/raw/',
    #                        processed_path= 'branch1/data_enhanced/processed/')
    # df_highf = process_data(path = 'double_pendulum_highfrict.csv',
    #                         raw_path= 'branch1/data_enhanced/raw/',
    #                         processed_path= 'branch1/data_enhanced/processed/')
    

    df_nof = process_data(path = 'double_pendulum_nofrict.csv', 
                          raw_path= 'data/raw/',
                          processed_path= 'data/processed/')
    df_lowf = process_data(path = 'double_pendulum_lowfrict.csv',
                           raw_path= 'data/raw/',
                           processed_path= 'data/processed/')
    df_highf = process_data(path = 'double_pendulum_highfrict.csv',
                            raw_path= 'data/raw/',
                            processed_path= 'data/processed/')

if __name__ == '__main__':
    main()
  