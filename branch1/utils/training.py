import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch import nn
from models import DoublePendulum
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import os


# ===============================================
# ============= Dataset ========================
# ===============================================

def create_dataset( files = [
                            'double_pendulum_lowfrict.csv',
                            'double_pendulum_nofrict.csv',
                            'double_pendulum_highfrict.csv'],
                    data_path = 'branch1/data_enhanced/processed/',
                    splits_path = 'branch1/data_enhanced/training/'):

    '''
    Docstring for create_dataset

    Inputs:
    - files: list of str
        List of filenames to split into train, val, test sets.
    - data_path: str
        Path where the data files are located.
    - splits_path: str  
        Path where to save the splitted datasets.
    Attention: be careful to have the simulation type in the filename as 'double_pendulum_<sim_type>.csv'
    as the function uses it to name the splitted files.

    Outputs:
    - None
    The function saves the splitted datasets as CSV files in the specified splits_path.
    '''
    for f in files:
        print(f"Splitting dataset for {f}...")
        df = pd.read_csv(data_path + f)
        sim_type = f.split('_')[2].split('.')[0]
        split_dataset(df, sim_type=sim_type, shuffle=False, splits_path=splits_path)

                   
def split_dataset(df: pd.DataFrame, 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.2, 
                  shuffle: bool = False, 
                  sim_type: str = '',
                  splits_path: str = 'data/training/') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Splits the dataset into training, validation, and test sets based on unique pendulum IDs.
    Inputs:
    - df: pd.DataFrame
        The input dataframe containing the dataset.
    - train_ratio: float
        The proportion of the dataset to include in the training set.
    - val_ratio: float
        The proportion of the dataset to include in the validation set.
    - shuffle: bool
        Whether to shuffle the dataset before splitting.
    - sim_type: str
        A string indicating the type of simulation (used for naming the output files).
    - splits_path: str
        The path where to save the splitted datasets.
    Outputs:
    - train_df: pd.DataFrame
        The training set dataframe. 
    - val_df: pd.DataFrame
        The validation set dataframe.
    - test_df: pd.DataFrame
        The test set dataframe.
    '''
    
    trainval_df, test_df = train_test_split(df, train_size = train_ratio, random_state=42, shuffle=shuffle)
    train_df, val_df = train_test_split(trainval_df, test_size = val_ratio, random_state=42, shuffle=shuffle)

    train_df.to_csv(f'{splits_path}train_' + sim_type + '.csv', index=False)
    val_df.to_csv(f'{splits_path}val_' + sim_type + '.csv', index=False)
    test_df.to_csv(f'{splits_path}test_' + sim_type + '.csv', index=False)

    return train_df, val_df, test_df


def pull_data_by_ids(data_path: str = '', 
                     data_splitter: str = 'id_pendulum',
                     xcols = ['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step'],
                     ycols = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2'],
                     modify_func = None):

    xdims = len(xcols)
    ydims = len(ycols)

    # Load data
    data = pd.read_csv(data_path)

    if modify_func is not None:
        data = modify_func(data)

    # Build X and Y
    ids = data[data_splitter].unique().tolist()
    evolution_length = data[data[data_splitter] == ids[0]].shape[0]


    X = np.zeros((len(ids), evolution_length, xdims)) # dim -> (n_pendulums, evolution_length, Xdims)
    Y = np.zeros((X.shape[0], X.shape[1], ydims)) # dim -> (n_pendulums, evolution_length, Ydims)

    for i, pid in enumerate(ids):
        each_data = data[data[data_splitter] == pid]
        X[i, :, :] = each_data[xcols].to_numpy()
        Y[i, :, :] = each_data[ycols].to_numpy()
    
    #convert all to float32
    X = X.astype(np.float32)    
    Y = Y.astype(np.float32)

    return X, Y

def get_dominant_frequencies(signal_data, dt, top_k=32):
        """
        signal_data: array numpy delle posizioni (es. theta1) [N_samples]
        dt: passo temporale (es. 0.01s)
        top_k: quante frequenze vogliamo estrarre
        """
        n = len(signal_data)
        
        # 1. Calcola la FFT
        freqs = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.abs(np.fft.rfft(signal_data))
        
        # 2. Ignora la frequenza 0 (il bias/media statica)
        freqs = freqs[1:]
        fft_vals = fft_vals[1:]
        
        # 3. Trova gli indici delle frequenze con ampiezza maggiore
        #argsort ordina crescente, quindi prendiamo gli ultimi k e invertiamo
        top_indices = np.argsort(fft_vals)[-top_k:][::-1]
        
        dominant_freqs = freqs[top_indices]
        
        return torch.tensor(dominant_freqs, dtype=torch.float32)

class KNGLDataset(Dataset):

    '''
    Docstring for KNGLDataset
    Custom Dataset for loading pendulum data for training neural networks.
    Inputs:
    - inputs: np.ndarray
        Array of input features with shape (n_pendulums, evolution_length, 5).
    - targets: np.ndarray
        Array of target values with shape (n_pendulums, evolution_length, 4).
    - time_steps: int
        Number of time steps to consider for each sample.
    - transform: callable, optional
        Optional transform to be applied on a sample.
    Outputs:
    - sample: torch.FloatTensor
        Tensor of input features for a given sample.
    - target: torch.FloatTensor
        Tensor of target values for a given sample.
    
    Methods:
    - __len__: returns the total number of samples in the dataset.
    - __getitem__: retrieves a sample and its corresponding target based on the index.

    
    '''
    def __init__(self, 
                 inputs, # (pendulums, evolution_length, 5)
                 targets, # (pendulums, evolution_length, 4)
                 horizon=32,
                 transform=None,
                 X_feature_positions_map=None,
                 Y_feature_position_map=None,
                 normalization_map=None,
                 fourier_features=-1):
        

   
        self.inputs = torch.FloatTensor(inputs)  # Converti in tensor
        self.targets = torch.FloatTensor(targets)

        self.X_feature_positions_map = X_feature_positions_map
        self.Y_feature_position_map = Y_feature_position_map
        self.normalization_map = normalization_map

        self.transform = transform

        self.horizon = horizon
        
        self.evolution_length = self.inputs.shape[1]
        self.n_pendulums = self.inputs.shape[0]

        # Normalize input and output if normalization map is provided
        if self.normalization_map:

            # normalize inputs
            for feature, pos in X_feature_positions_map.items():
                if feature in normalization_map: # normalize only what has been normalized
                    self.inputs[:, :, pos] = (self.inputs[:, :, pos] - normalization_map[feature]['mean']) / (normalization_map[feature]['std'] + 1e-8)

            # normalize targets
            for feature, pos in Y_feature_position_map.items():
                if feature in normalization_map: # normalize only what has been normalized
                    self.targets[:, :, pos] = (self.targets[:, :, pos] - normalization_map[feature]['mean']) / (normalization_map[feature]['std'] + 1e-8)

        # Add Fourier features if specified (only on inputs)
        if fourier_features>0:

            if 'q1' in X_feature_positions_map:
                
                q1_pos = X_feature_positions_map['q1']
                q1_np = self.inputs[:, :, q1_pos].numpy()
                q1_freqs = get_dominant_frequencies(q1_np.flatten(), dt=0.01, top_k=fourier_features)

                self.inputs = torch.cat([self.inputs,
                                         torch.sin(2 * np.pi * q1_freqs.unsqueeze(0).unsqueeze(0) * self.inputs[:, :, q1_pos].unsqueeze(-1)),
                                         torch.cos(2 * np.pi * q1_freqs.unsqueeze(0).unsqueeze(0) * self.inputs[:, :, q1_pos].unsqueeze(-1))], dim=-1)
            
            elif 'q1_sin' in X_feature_positions_map and 'q1_cos' in X_feature_positions_map:

                q1_sin_pos = X_feature_positions_map['q1_sin']
                q1_cos_pos = X_feature_positions_map['q1_cos']
                q1_angles = torch.atan2(self.inputs[:, :, q1_sin_pos], self.inputs[:, :, q1_cos_pos]).numpy()
                q1_freqs = get_dominant_frequencies(q1_angles.flatten(), dt=0.01, top_k=fourier_features)

                q1_angles_tensor = torch.atan2(self.inputs[:, :, q1_sin_pos], self.inputs[:, :, q1_cos_pos]).unsqueeze(-1)
                
                self.inputs = torch.cat([self.inputs,
                                         torch.sin(2 * np.pi * q1_freqs.unsqueeze(0).unsqueeze(0) * q1_angles_tensor),
                                         torch.cos(2 * np.pi * q1_freqs.unsqueeze(0).unsqueeze(0) * q1_angles_tensor)], dim=-1)

            if 'q2' in X_feature_positions_map:
                
                q2_pos = X_feature_positions_map['q2']
                q2_np = self.inputs[:, :, q2_pos].numpy()
                q2_freqs = get_dominant_frequencies(q2_np.flatten(), dt=0.01, top_k=fourier_features)

                self.inputs = torch.cat([self.inputs,
                                         torch.sin(2 * np.pi * q2_freqs.unsqueeze(0).unsqueeze(0) * self.inputs[:, :, q2_pos].unsqueeze(-1)),
                                         torch.cos(2 * np.pi * q2_freqs.unsqueeze(0).unsqueeze(0) * self.inputs[:, :, q2_pos].unsqueeze(-1))], dim=-1)
            
            elif 'q2_sin' in X_feature_positions_map and 'q2_cos' in X_feature_positions_map:

                q2_sin_pos = X_feature_positions_map['q2_sin']
                q2_cos_pos = X_feature_positions_map['q2_cos']
                q2_angles = torch.atan2(self.inputs[:, :, q2_sin_pos], self.inputs[:, :, q2_cos_pos]).numpy()
                q2_freqs = get_dominant_frequencies(q2_angles.flatten(), dt=0.01, top_k=fourier_features)

                q2_angles_tensor = torch.atan2(self.inputs[:, :, q2_sin_pos], self.inputs[:, :, q2_cos_pos]).unsqueeze(-1)
                
                self.inputs = torch.cat([self.inputs,
                                         torch.sin(2 * np.pi * q2_freqs.unsqueeze(0).unsqueeze(0) * q2_angles_tensor),
                                         torch.cos(2 * np.pi * q2_freqs.unsqueeze(0).unsqueeze(0) * q2_angles_tensor)], dim=-1)

    
    def __len__(self): # il data loader con __len__ sa che può chiamare len-1 volte  
        return self.n_pendulums * (self.evolution_length - self.horizon)  # Numero totale di campioni (size 0)
    
    def __getitem__(self, idx):

        pendulum_idx = idx // (self.evolution_length - self.horizon)
        time_idx = idx % (self.evolution_length - self.horizon)

        sample = self.inputs[pendulum_idx, time_idx:time_idx+self.horizon]
        target = self.targets[pendulum_idx, time_idx:time_idx+self.horizon]
        
        if self.transform:
            sample = self.transform(sample)

        #print(f"Sample shape: {sample.shape}, Target shape: {target.shape}")

        #print("\nSAMPLE")
        #print(sample)
        #print("\nTARGET")
        #print(target)
        
        return sample, target  # Ritorna tupla (input, target)
    
# ===============================================
# ============= Modify  data ====================
# ===============================================

def normalize_data(data, data_min=None, data_max=None):
    '''
    Docstring for normalize_data
    Normalizes the input data to the range [0, 1].
    Inputs:
    - data: np.ndarray
        The input data to be normalized.
    - data_min: np.ndarray or None
        Minimum values for each feature. If None, computed from data.
    - data_max: np.ndarray or None
        Maximum values for each feature. If None, computed from data.
    Outputs:
    - normalized_data: np.ndarray
        The normalized data.
    - data_min: np.ndarray
        The minimum values used for normalization.
    - data_max: np.ndarray
        The maximum values used for normalization.
    '''
    if data_min is None:
        data_min = data.min(axis=0)
    if data_max is None:
        data_max = data.max(axis=0)
    normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
    return normalized_data, data_min, data_max

def turn_to_cosine_sine(X_dataset, cols=['q1', 'q2']):
    '''
    Transforms angular columns in the dataset to their sine and cosine representations.
    '''
    X_transformed = X_dataset.copy()
    for col in cols:
        angles = X_transformed[col].values
        X_transformed[f'{col}_sin'] = np.sin(angles)
        X_transformed[f'{col}_cos'] = np.cos(angles)
    
    return X_transformed

def fourier_features(X_dataset, cols=['q1', 'q2'], top_k=16, dt=0.01):
    '''
    Adds Fourier features for specified angular columns in the dataset.
    '''
    X_transformed = X_dataset.copy()
    for col in cols:
        angles = X_transformed[col].values
        dominant_freqs = get_dominant_frequencies(angles, dt=dt, top_k=top_k).numpy()
        
        for i, freq in enumerate(dominant_freqs):
            X_transformed[f'{col}_fourier_sin_{i}'] = np.sin(2 * np.pi * freq * angles)
            X_transformed[f'{col}_fourier_cos_{i}'] = np.cos(2 * np.pi * freq * angles)

    return X_transformed
# ===============================================
# ================= Metrics =====================
# ===============================================

def avg_param_difference (true_params: torch.Tensor, 
                      pred_params: torch.Tensor) -> float:
    '''
    Docstring for param_difference
    Computes the mean absolute difference between true and predicted parameters.
    Inputs:
    - true_params: torch.Tensor
        Tensor of true parameter values.
    - pred_params: torch.Tensor
        Tensor of predicted parameter values.
    Outputs:
    - mean_diff: float
        The mean absolute difference between true and predicted parameters.
    '''
    with torch.no_grad():
        diff = torch.abs(true_params - pred_params)
        mean_diff = diff.mean().item()
    return mean_diff

def max_param_difference (true_params: torch.Tensor, 
                          pred_params: torch.Tensor) -> float:
    '''
    Docstring for max_param_difference
    Computes the maximum absolute difference between true and predicted parameters.
    Inputs:
    - true_params: torch.Tensor
        Tensor of true parameter values.
    - pred_params: torch.Tensor
        Tensor of predicted parameter values.
    Outputs:
    - max_diff: float
        The maximum absolute difference between true and predicted parameters.
    '''
    with torch.no_grad():
        diff = torch.abs(true_params - pred_params)
        max_diff = diff.max().item()
    return max_diff

def average_discrepancy_perc(true_params: torch.Tensor, 
                             pred_params: torch.Tensor) -> float:
    '''
    Docstring for average_discrepancy_perc
    Computes the average percentage discrepancy between true and predicted parameters.
    Inputs:
    - true_params: torch.Tensor
        Tensor of true parameter values.
    - pred_params: torch.Tensor
        Tensor of predicted parameter values.
    Outputs:
    - avg_perc_diff: float
        The average percentage discrepancy between true and predicted parameters.
    '''
    with torch.no_grad():
        perc_diff = torch.abs((true_params - pred_params) / (true_params + 1e-8)) * 100.0
        avg_perc_diff = perc_diff.mean().item()
    return avg_perc_diff

# ===============================================
# ============= Training ========================
# ===============================================

class EarlyStopping:

    '''
    Docstring for EarlyStopping
    Early stopping utility to stop training when validation loss does not improve.
    Inputs:
    - patience: int
        Number of epochs to wait for improvement before stopping.
    Outputs:
    - None

    Methods:
    - step: checks if validation loss has improved and updates state.
    - restore_best_weights: restores the model to the best weights observed during training.

    '''
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_state = None
        self.wait = 0
        self.stopped_epoch = 0

    def step(self, val_loss, model):
        improved = val_loss < self.best_loss
        if improved:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch += 1
            return True  # stop
        return False

    def restore_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def get_adam_effective_lr(optimizer, param):
    '''
    Calculate the effective learning rate for a given parameter in Adam optimizer.
    Inputs:
    - optimizer: torch.optim.Adam
        The Adam optimizer instance.
    - param: torch.nn.Parameter
        The parameter for which to calculate the effective learning rate.
    Outputs:
    - effective_lr: torch.Tensor or None
        The effective learning rate for the parameter, or None if state is not initialized.

    '''
    state = optimizer.state[param]
    if len(state) == 0:
        return None
    
    lr = optimizer.param_groups[0]['lr']
    beta1, beta2 = optimizer.param_groups[0]['betas']
    eps = optimizer.param_groups[0]['eps']
    
    step = state['step']
    bias_correction1 = 1 - beta1 ** step
    exp_avg_sq = state['exp_avg_sq']
    
    effective_lr = lr / bias_correction1 / (torch.sqrt(exp_avg_sq) + eps)
    return effective_lr

def train_nn_model(
    model,
    device,
    
    ds_train,
    ds_val,

    early_stopper, 
    epochs=100,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=32,
    optimizer_type = 'adam',
    scheduler_type = 'plateau',
    loss_type = 'mse',
    
    verbose=False,

    show_grad = False,
    show_lrs = False,
    **fit_params,
):
    # --- Model to device ---
    model.to(device)

    # --- Data Loaders ---
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    
    # --- Optimizer and Scheduler ---
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"[ERROR] Optimizer '{optimizer_type}' not recognized")
        print("\tAvailable: \n\t -'adam' \n\t -'adamw'")
        exit()
    
    scheduler = None
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=lr, 
            steps_per_epoch=len(train_loader), 
            epochs=epochs, 
            pct_start=0.3, 
            div_factor=25.0, 
            final_div_factor=1e4
        )
    else:
        print(f"[ERROR] Scheduler '{scheduler_type}' not recognized")
        print("\tAvailable: \n\t -'plateau' \n\t -'onecycle'")
        exit()

    # --- Loss Function ---
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'physics':
        loss_fn = physics_loss
    elif loss_type == 'mae':
        loss_fn = nn.L1Loss()
    else:
        print(f"[ERROR] Loss function '{loss_type}' not recognized")
        print("\tAvailable: \n\t -'mse' \n\t -'physics'")
        exit()

    # --- Training Loop ---
    history = {"loss": [], "val_loss": []}
    with tqdm(total=epochs, desc="Training", unit="epoch", disable=not verbose) as pbar:
        for epoch in range(epochs):
            
            model.train()
            running_loss = 0.0
            n_batches = 0

            for i, (xb, yb) in enumerate(train_loader):
                
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                y_pred = model(xb)

                if loss_type == 'mse':
                    loss = loss_fn(y_pred, yb) 
                elif loss_type == 'mae':
                    loss = loss_fn(y_pred, yb)
                elif loss_type == 'physics':
                    loss = loss_fn(y_pred, yb, model=model)

                loss.backward()

                if show_grad: # debug on gradients
                    if i == 0:
                        print(f"Gradients for batch {i}:")
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                print(f"{name}: {param.grad.norm().item()}")
                            else:
                                print(f"{name}: None")
                if show_lrs: # debug on effective learning rates
                    if i == 0:
                        print(f"Effective Learing Rates for batch {i}:")
                        for name, param in model.named_parameters():
                            eff_lr = get_adam_effective_lr(optimizer, param)
                            if eff_lr is not None:
                                print(f"{name}: {eff_lr.mean()}")
                            else:
                                print(f"{name}: None")
                            
                optimizer.step()

                if scheduler_type == 'onecycle':
                    scheduler.step()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(n_batches, 1)
            history["loss"].append(epoch_loss)

            # Validation
            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        y_pred = model(xb)

                        if loss_type == 'mse':
                            vloss = loss_fn(y_pred, yb) 
                        elif loss_type == 'mae':
                            vloss = loss_fn(y_pred, yb)
                        elif loss_type == 'physics':
                            vloss = loss_fn(y_pred, yb, model=model)

                        val_running_loss += vloss.item()
                        val_batches += 1
                val_loss = val_running_loss / max(val_batches, 1)
                history["val_loss"].append(val_loss)

                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)

                # Early stopping
                stop = early_stopper.step(val_loss, model)
                if stop:
                    pbar.set_postfix({
                        'status': 'Early Stopped',
                        'loss': f'{epoch_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}'
                    })
                    early_stopper.restore_best_weights(model)
                    break

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{epoch_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")

    return history



def plot_training_history(history=None,
                          figsize=None, 
                          print_final_scores=True,
                          save_path=None,
                          show_plot=False):
    
    fig = plt.figure(figsize=figsize)
    
    for metric in history.keys():
        plt.plot(history[metric], label=metric)
    
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.keys()) > 0:
        plt.legend()
    
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()

    if show_plot:
        plt.show()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path+'.png')
    
    if print_final_scores:
        trl = history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history:
            vll = history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)

# ===============================================
# ============= Evaluation ======================
# ===============================================

def evaluate_model(model, 
                   ds_test, 
                   loss_fn, 
                   device="cpu", 
                   batch_size=32):
    
    model.to(device)
    model.eval()

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    n_batches = 0

    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    
    return avg_loss, y_pred, yb

def evaluate_lagrangian_model(model, 
                   ds_test, 
                   loss_fn, 
                   device="cpu", 
                   batch_size=32):
    
    model.to(device)
    model.eval()

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    n_batches = 0

    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred, _ = model(xb)
            loss = loss_fn(y_pred, yb)
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    

    return avg_loss

def predict_model(model, 
                  X_test,
                  device="cpu"):
    
    model.to(device)
    model.eval()

    test_loader = DataLoader(X_test, shuffle=False)
    predictions = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            y_pred = model(xb)
            predictions.append(y_pred.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions

# =============================================================
# ===================== Losses ================================
# =============================================================

def physics_loss(y_pred, y_true, model = None, horizon = 1): 
    
    #eliminate cosine jump
    q_pred = y_pred[:, :, [0, 2]]
    q_true = y_true[:, :, [0, 2]]
    
    cos_loss = nn.MSELoss()(torch.cos(q_pred), torch.cos(q_true))
    sin_loss = nn.MSELoss()(torch.sin(q_pred), torch.sin(q_true))
    
    v_pred = y_pred[:, :, [1, 3]]
    v_true = y_true[:, :, [1, 3]]

    vel_loss = nn.SmoothL1Loss(beta=1.0)(v_pred, v_true)
    
    #Loss on angles and velocity weighted not to obscure 
    base_loss = 10 * (cos_loss + sin_loss) + 1 * vel_loss

    if model is not None:
        dyn = model.f 
        #print("I am here")
        # Regolarizzazione L2 sui parametri (corretto per tensori)
        reg_friction = torch.exp(dyn.logb1**2).mean() + torch.exp(dyn.logb2**2).mean()
        reg_length   = torch.exp(dyn.logl1**2).mean() + torch.exp(dyn.logl2**2).mean()
        reg_masses   = torch.exp(dyn.logm1**2).mean() + torch.exp(dyn.logm2**2).mean()

        # Pesi della regolarizzazione (lambda)
        if horizon < 10:
            lambda_frict = 1  
            lambda_len   = 1
            lambda_masses = 1
        else:
            lambda_frict = 0.01  
            lambda_len   = 0.01 
            lambda_masses = 0.01
        #print(f"Regularization terms: Friction: {reg_friction.item():.6f}, Length: {reg_length.item():.6f}, Masses: {reg_masses.item():.6f}")
        #print(f"Parameters: b1: {torch.exp(dyn.logb1).item():.4f}, b2: {torch.exp(dyn.logb2).item():.4f}, l1: {torch.exp(dyn.logl1).item():.4f}, l2: {torch.exp(dyn.logl2).item():.4f}, m1: {torch.exp(dyn.logm1).item():.4f}, m2: {torch.exp(dyn.logm2).item():.4f}")

        total_loss = base_loss + lambda_frict * reg_friction + lambda_len * reg_length + lambda_masses * reg_masses
        
        return total_loss
    
    else:
        return base_loss


class LagrangianLoss(nn.Module):
    def __init__(self, m1, m2, l1, l2, b1, b2, X_scale, acc_scale, g=9.81):
        super().__init__()
        # Salviamo i parametri fisici
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.b1, self.b2 = b1, b2
        self.g = g

        self.register_buffer('X_scale', torch.tensor(X_scale, dtype=torch.float32))
        self.register_buffer('acc_scale', torch.tensor(acc_scale, dtype=torch.float32))

    def forward(self, inputs, predictions):
        """
        inputs:      Tensore (N, 4) -> [q1, q2, q1_dot, q2_dot] (STATO CORRENTE)
        predictions: Tensore (N, 2) -> [q1_ddot, q2_ddot]       (ACCELERAZIONE PREDETTA)
        
        Nota: Se la tua rete predice anche q_dot, estrai solo le accelerazioni dalle predictions.
        """
        #print(f"input: {inputs.shape}")
        #print(f"predictions: {predictions.shape}")

        inputs_phys = inputs * self.X_scale
        
        # 1. ESTRAZIONE VARIABILI DAGLI INPUT (q e q_dot attuali)
        # Queste servono per calcolare le matrici M, C, G
        q1 = inputs_phys[:, :, 0]
        q2 = inputs_phys[:, :, 1]
        q1_dot = inputs_phys[:, :, 2]
        q2_dot = inputs_phys[:, :, 3]

        # 2. ESTRAZIONE VARIABILI DALLE PREDIZIONI (q_ddot futura)
        # Queste sono le incognite che la rete sta cercando di indovinare
        # Assumiamo che prediction sia [q1_ddot, q2_ddot]
        # Se predici anche velocità, adatta gli indici (es. predictions[:, 2:])
        q1_ddot_pred = predictions[:, :, 0]
        q2_ddot_pred = predictions[:, :, 1]

        q1_ddot_phys = q1_ddot_pred * self.acc_scale[0]
        q2_ddot_phys = q2_ddot_pred * self.acc_scale[1]

        # --- 3. COSTRUZIONE MATRICI FISICHE (uguale a prima) ---
        # Nota importante: Usiamo q1, q2, q1_dot DAGLI INPUT!
        
        # Parametri locali per brevità
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        # Matrice Inerzia M(q_input)
        M00 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * torch.cos(q2)
        M01 = m2 * l2**2 + m2 * l1 * l2 * torch.cos(q2)
        M10 = M01
        M11 = m2 * l2**2 

        # Coriolis C(q_input, q_dot_input)
        term_c = m2 * l1 * l2 * torch.sin(q2)
        C0 = -term_c * (2 * q1_dot + q2_dot) * q2_dot
        C1 = term_c * q1_dot**2

        # Damping D(q_dot_input)
        D0 = self.b1 * q1_dot
        D1 = self.b2 * q2_dot

        # Gravità G(q_input)
        G0 = (m1 + m2) * g * l1 * torch.sin(q1) + m2 * g * l2 * torch.sin(q1 + q2)
        G1 = m2 * g * l2 * torch.sin(q1 + q2)

        # --- 4. CALCOLO RESIDUO LAGRANGIANO ---
        # Equazione: M * q_ddot + C + D + G = 0 (assumendo torque u=0)
        # Attenzione alle dimensioni: M è matrice, q_ddot è vettore.
        
        # Riga 1 del sistema
        # M00*acc1 + M01*acc2 + C0 + D0 + G0 = 0 ?
        res_1 = (M00 * q1_ddot_phys) + (M01 * q2_ddot_phys) + C0 + D0 + G0
        
        # Riga 2 del sistema
        # M10*acc1 + M11*acc2 + C1 + D1 + G1 = 0 ?
        res_2 = (M10 * q1_ddot_phys) + (M11 * q2_ddot_phys) + C1 + D1 + G1

        # NOTA SUI SEGNI:
        # La dinamica standard è M*q_ddot + C + G = u.
        # Se u=0, allora M*q_ddot + C + G = 0.
        # Controlla i segni di C, D, G nel tuo sistema. 
        # Nel codice precedente calcolavamo b = u - C - D - G e poi acc = M_inv * b.
        # Quindi M*acc = -C -D -G  =>  M*acc + C + D + G = 0. È corretto.

        torque_scale = (self.m1 + self.m2) * self.g * self.l1 
        # Aggiungiamo un epsilon per evitare divisioni per zero
        torque_scale = torque_scale + 1e-6

        # Normalizziamo i residui
        res_1_norm = res_1 / torque_scale
        res_2_norm = res_2 / torque_scale

        loss_physics = torch.mean(res_1_norm**2) + torch.mean(res_2_norm**2)
        
        return loss_physics
    
# ============================================================
# ================ Loading and Saving Models =================
# ============================================================

def save_model(model, history: dict, model_path: str):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(model.state_dict(), model_path)

    np.save(model_path + '_history.npy', history)

def load_model(model, model_path):
    
    model.load_state_dict(torch.load(model_path))

    history = np.load(model_path + '_history.npy', allow_pickle='TRUE').item()
    
    return model, history

def log_recipe(recipe: dict, log_path: str):
    
    with open(log_path, 'w') as f:
        for key, value in recipe.items():
            f.write(f'{key}: {value}\n')




            
