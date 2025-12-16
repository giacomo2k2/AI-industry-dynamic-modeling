import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from data_inspection import show_bins, show_status_stats, show_energy_stats
import os

# Set global plot style
plt.rc('font', family='serif')

# Global plotting flag
PLOT = False

# ========================================================
# ================== MODIFY THE DATASET ==================
# ========================================================



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

def cut_dataset(df:pd.DataFrame, threshold:float, min_threshold:float = None) -> pd.DataFrame:
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
    if min_threshold is not None:
        return df[(df['time_step'] >= min_threshold) & (df['time_step'] < threshold)]
    else:
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

def optimize_threshold(
        df:pd.DataFrame,
        save_path:str = None,
        show_plot:bool = False
    ) -> float:
    """
    Optimize the threshold based on Multivariate Kullback-Leibler divergence 
    on the full phase space (q1, q2, q1_dot, q2_dot).
    """
    best_threshold = None
    best_kl_divergence = float('inf')
    kl_vector = []
    
    # 1. Prepare global parameters
    max_time = df['time_step'].to_numpy()[-1]
    thresholds_vec = np.linspace(0.0, max_time, num=50) # Reduced num for performance

    # 2. Define global bounds for the histogram to keep bins consistent across loops
    # Angles are always -pi to pi
    # Velocities: we take the max absolute value seen in the whole dataset
    max_v1 = df['q1_dot'].abs().max()
    max_v2 = df['q2_dot'].abs().max()
    
    # Define ranges for: [q1, q2, q1_dot, q2_dot]
    # We add a small buffer (+0.1) to velocity bounds to ensure no data falls on the edge
    bounds = [
        [-np.pi, np.pi], 
        [-np.pi, np.pi], 
        [-max_v1 - 0.1, max_v1 + 0.1], 
        [-max_v2 - 0.1, max_v2 + 0.1]
    ]

    # 3. Main Optimization Loop
    for threshold in thresholds_vec:
        if threshold == 0.0:
            kl_vector.append(np.nan)
            continue
        
        # Cut dataset
        df_temp = cut_dataset(df, threshold=threshold)
        
        # Extract full state: q1, q2, q1_dot, q2_dot
        states = df_temp[['q1', 'q2', 'q1_dot', 'q2_dot']].to_numpy()
        
        # 4. Compute N-Dimensional Histogram
        # CRITICAL: bins must be small (e.g., 10 to 15) to avoid memory explosion in 4D
        # bins=12 means 12^4 = 20,736 bins. 
        H, edges = np.histogramdd(states, bins=12, range=bounds, density=True)
        
        # Normalize Histogram to get Q (Observed Distribution)
        # We flatten the array to treat it as a single probability vector
        q = H.flatten()
        q = q / (q.sum() + 1e-10) # Normalize sum to 1
        
        # 5. Define Target Distribution P (Uniform Reference)
        # In a fully mixed ergodic system, we compare against a flat distribution
        p = np.ones_like(q) / q.size
        
        # Calculate KL
        # We calculate one single scalar KL for the whole 4D space
        current_kl = kullback_leibler(
            q=q, 
            p=p, 
            lambda_val=0.01, 
            threshold_penalty=(max_time - threshold) / max_time
        )
            
        kl_vector.append(current_kl)

        if current_kl < best_kl_divergence:
            best_kl_divergence = current_kl
            best_threshold = threshold



    fig = plt.figure(figsize=(10, 6))
    plt.plot(thresholds_vec, kl_vector, label='KL Divergence')
    plt.scatter(best_threshold, best_kl_divergence, linestyle='', marker='o', color='red', label='Optimal Threshold')
    plt.title('Kullback-Leibler Divergence vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Kullback-Leibler Divergence')
    plt.grid()
    plt.tight_layout()
    if show_plot:
        plt.show()
    if save_path:
        fig.savefig(save_path+'.png')
        

    return best_threshold

def add_white_noise(df: pd.DataFrame, std: float = 0.1) -> pd.DataFrame:
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

def process_data(
        file_path:str, 
        processed_folder: str,
        plot: bool = PLOT,
        energy_cut : bool = True,
        white_noise: bool = False,
        round_pi: bool = False
    ) -> pd.DataFrame:

    plot_folder = os.path.join('plots', processed_folder)
    os.makedirs(plot_folder, exist_ok=True)
    print("Plots folder: ", plot_folder)

    print("Resulting datsets will be saved in: ", processed_folder)
    os.makedirs(processed_folder, exist_ok=True)

    file_name = os.path.basename(file_path).split('.csv')[0]
    print(f"File name: {file_name}")
    
    sim_type = file_path.split('_')[2].split('.')[0]
    print(f"Processing {sim_type} data...")
    
    # Read the csv in the data raw folder
    df = pd.read_csv(file_path)

    if energy_cut:
        processed_df, discarded = energy_check(df)
    

    if white_noise:
        processed_df = add_white_noise(processed_df)

    # if round_pi:
    #     processed_df = round_to_pi(processed_df)



    status_cols = ['q1', 'q1_dot', 'q1_dot_dot', 'q2', 'q2_dot', 'q2_dot_dot']
    status_names = [r'$q_1$', r'$\dot{q}_1$', r'$\ddot{q}_1$', r'$q_2$', r'$\dot{q}_2$',  r'$\ddot{q}_2$']
    vlim = processed_df[status_cols].abs().max().to_numpy()

    vlim[0] = np.pi
    vlim[3] = np.pi
    if plot:
        show_bins(processed_df, cols=status_cols, names=status_names, vlim=vlim, save=True, filename=sim_type)

        show_status_stats(processed_df, cols=status_cols, names=status_names, filename=sim_type)   

        show_energy_stats(processed_df, filename=sim_type)

    threshold_path = os.path.join(plot_folder, file_name+'_threshold')
    print("Saving optmized threshold plot in : ", threshold_path)
    threshold = optimize_threshold(
        df = processed_df,
        save_path= threshold_path,
        show_plot= plot
    )

    print(f"Optimal threshold for {sim_type}: {threshold}")
    processed_df = cut_dataset(processed_df, threshold=threshold)

    #show_status_stats(processed_df, cols=status_cols, names=status_names, filename=sim_type)   

    if plot:
        show_bins(processed_df, cols=status_cols, names=status_names, vlim=vlim, save=False)

    # Save processed dataframe to csv
    save_path = os.path.join(processed_folder, file_name)
    os.makedirs(processed_folder, exist_ok=True)
    processed_df.to_csv(save_path, index=False)

    # Print process terminated and number of pendulums kept
    n_total = df['id_pendulum'].nunique()
    n_kept = processed_df['id_pendulum'].nunique()
    print(f"Processing terminated. Kept {n_kept} out of {n_total} pendulums.")

    return processed_df

# ========================================================
# ================== RUN THE PROCESSING ==================
# ========================================================

def main(
        raw_folder: str,
        processed_folder: str,
        energy_cut : bool,
        white_noise: bool,
        round_pi: bool
):
    

    # df_nof = process_data(path = 'double_pendulum_nofrict.csv', 
    #                       raw_path= 'branch1/data_enhanced/raw/',
    #                       processed_path= 'branch1/data_enhanced/processed/')
    # df_lowf = process_data(path = 'double_pendulum_lowfrict.csv',
    #                        raw_path= 'branch1/data_enhanced/raw/',
    #                        processed_path= 'branch1/data_enhanced/processed/')
    # df_highf = process_data(path = 'double_pendulum_highfrict.csv',
    #                         raw_path= 'branch1/data_enhanced/raw/',
    #                         processed_path= 'branch1/data_enhanced/processed/')

    mode_string = ''


    for i , mod in enumerate([energy_cut, white_noise, round_pi]):
        if mod:
            mode_string += '_' + ['EC', 'WN', 'RP'][i]


    processed_folder = os.path.join(os.path.dirname(processed_folder), os.path.basename(processed_folder) + mode_string)


    for f in os.listdir(raw_folder):
        # Load processed data
        file_path = os.path.join(raw_folder, f)

        df = process_data(file_path = file_path, 
                          processed_folder= processed_folder,
                          energy_cut= energy_cut,
                          white_noise= white_noise,
                          round_pi= round_pi)


    # df_nof = process_data(path = 'double_pendulum_nofrict.csv', 
    #                       raw_path= raw_path,
    #                       processed_path= processed_path,
    #                       energy_cut= energy_cut,
    #                       white_noise= white_noise,
    #                       round_pi= round_pi)
    
    # df_lowf = process_data(path = 'double_pendulum_lowfrict.csv',
    #                        raw_path= raw_path,
    #                        processed_path= processed_path,
    #                        energy_cut= energy_cut,
    #                        white_noise= white_noise,
    #                        round_pi= round_pi)
    
    # df_highf = process_data(path = 'double_pendulum_highfrict.csv',
    #                         raw_path= raw_path,
    #                         processed_path= processed_path,
    #                         energy_cut= energy_cut,
    #                         white_noise= white_noise,
    #                         round_pi= round_pi)

if __name__ == '__main__':

    main(
        raw_folder= './data/raw/double_pendulum_nofrict.csv',
        processed_folder= './data/processed/double_pendulum_nofrict.csv',
        energy_cut = True,
        white_noise=True,
        round_pi= False
    )
  