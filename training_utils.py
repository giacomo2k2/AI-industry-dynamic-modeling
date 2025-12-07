import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.2, shuffle: bool = False, sim_type: str = ''):
    
    trainval_df, test_df = train_test_split(df, train_size = train_ratio, random_state=42, shuffle=shuffle)
    train_df, val_df = train_test_split(trainval_df, test_size = val_ratio, random_state=42, shuffle=shuffle)

    train_df.to_csv('data/training/train_' + sim_type + '.csv', index=False)
    val_df.to_csv('data/training/val_' + sim_type + '.csv', index=False)
    test_df.to_csv('data/training/test_' + sim_type + '.csv', index=False)

    return train_df, val_df, test_df

class EarlyStopping:
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

class KNGLDataset(Dataset):
    def __init__(self, 
                 inputs, # (pendulums, evolution_length, 5)
                 targets, # (pendulums, evolution_length, 4)
                 time_steps=32,
                 transform=None):
   
        self.inputs = torch.FloatTensor(inputs)  # Converti in tensor
        self.targets = torch.FloatTensor(targets)
        
        self.transform = transform
        self.time_steps = time_steps
        
        self.evolution_length = self.inputs.shape[1]
        self.n_pendulums = self.inputs.shape[0]
    
    def __len__(self): # il data loader con __len__ sa che può chiamare len-1 volte  
        return self.n_pendulums * (self.evolution_length - self.time_steps)  # Numero totale di campioni (size 0)
    
    def __getitem__(self, idx):

        pendulum_idx = idx // (self.evolution_length - self.time_steps)
        time_idx = idx % (self.evolution_length - self.time_steps)

        sample = self.inputs[pendulum_idx, time_idx:time_idx+self.time_steps]
        target = self.targets[pendulum_idx, time_idx:time_idx+self.time_steps]
        
        if self.transform:
            sample = self.transform(sample)

        #print(f"Sample shape: {sample.shape}, Target shape: {target.shape}")

        #print("\nSAMPLE")
        #print(sample)
        #print("\nTARGET")
        #print(target)
        
        return sample, target  # Ritorna tupla (input, target)

def train_nn_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    loss_fn,
    early_stopper, 
    verbose=False,
    patience=10,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    shuffle=True,
    device="cpu",
    **fit_params,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds_train = KNGLDataset(X_train.astype(np.float32), y_train.astype(np.float32), time_steps=8)
    ds_val = KNGLDataset(X_val.astype(np.float32), y_val.astype(np.float32), time_steps=8)

    early_stopping = early_stopper
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    history = {"loss": [], "val_loss": []}

    # tqdm solo sulle EPOCHE
    with tqdm(total=epochs, desc="Training", unit="epoch", disable=not verbose) as pbar:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                y_pred = model(xb)

                if i == 0 and epoch % 10 == 0 and verbose:
                    print(f"\EPOCH {epoch+1} - BATCH {i+1}")
                    print("Difference between prediction and target (only first el of batch):")
                    print(y_pred[0] - yb[0])

                loss = loss_fn(y_pred, yb)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(n_batches, 1)
            history["loss"].append(epoch_loss)

            # Validazione
            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        y_pred = model(xb)
                        vloss = loss_fn(y_pred, yb)
                        val_running_loss += vloss.item()
                        val_batches += 1
                val_loss = val_running_loss / max(val_batches, 1)
                history["val_loss"].append(val_loss)

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

                # Aggiorna tqdm con loss corrente
                pbar.set_postfix({
                    'loss': f'{epoch_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})

            pbar.update(1)  # Avanza la barra
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")

    return history


def plot_training_history(
        history=None,
        figsize=None, 
        print_final_scores=True):
    
    plt.figure(figsize=figsize)
    
    for metric in history.keys():
        plt.plot(history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history:
            vll = history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)


def main():

    files = [
        'double_pendulum_lowfrict.csv',
        'double_pendulum_nofrict.csv',
        'double_pendulum_highfrict.csv']
    
    for f in files:
        print(f"Splitting dataset for {f}...")
        df = pd.read_csv('data/processed/' + f)
        sim_type = f.split('_')[2].split('.')[0]
        split_dataset(df, sim_type=sim_type, shuffle=False)
    

if __name__ == "__main__":
    main()