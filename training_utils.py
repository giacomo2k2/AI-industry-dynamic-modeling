import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

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
    """
    X, y: torch.Tensor
    loss_fn: callable (y_pred, y_true) -> loss
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds_train = TensorDataset(X_train, y_train)
    ds_val = TensorDataset(X_val, y_val)
    early_stopping = early_stopper
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False) # loads an array of minibatches of said size
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train() # sets the model to training mode
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader: #it is iterating on minibatches not single input single output
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad() # set gradients to zero
            y_pred = model(xb) # forward pass
            loss = loss_fn(y_pred, yb) 
            loss.backward() # backward pass : does everything itself
            optimizer.step() # it is to update the Adam optimizer, learning rate for instance
            running_loss += loss.item() # accumulate loss given from this minibatch
            n_batches += 1 #states that a minibatch has been processed

        epoch_loss = running_loss / max(n_batches, 1) # compute the average loss per epoch
        history["loss"].append(epoch_loss)

        if verbose:
            msg = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}"

        # validazione + early stopping
        if val_loader is not None:
            model.eval() # sets the model to evaluation mode: inference
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

            if verbose:
                msg += f" - val_loss: {val_loss:.4f}"
                print(msg)

            # early stopping con restore_best_weights=True
            stop = early_stopping.step(val_loss, model)
            if stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                early_stopping.restore_best_weights(model)
                break
        else:
            if verbose:
                print(msg)

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