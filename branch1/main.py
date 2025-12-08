import torch
import pandas as pd
import numpy as np
from training_utils import train_nn_model, plot_training_history, EarlyStopping, physics_loss
from architecture import DynLayer, EulerModel, RK4Model
from torch import nn
import argparse
    
def main(train_path: str, 
         val_path: str):

    # --- Device setup ---

    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if my_device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_reserved(my_device)/1e9} GB")

    # --- Train Data ---

    # Load training data
    tr_data = pd.read_csv(train_path)

    # Ground truth
    m1_gt = tr_data['m1'].iloc[0]
    m2_gt = tr_data['m2'].iloc[0]
    l1_gt = tr_data['l1'].iloc[0]
    l2_gt = tr_data['l2'].iloc[0]
    b1_gt = tr_data['b1'].iloc[0]
    b2_gt = tr_data['b2'].iloc[0]

    # Build X_train and y_train

    tr_ids = tr_data['id_pendulum'].unique().tolist()
    tr_evolution_length = tr_data[tr_data['id_pendulum'] == tr_ids[0]].shape[0]

    X_train = np.zeros((len(tr_ids), tr_evolution_length, 5)) # dim -> (n_pendulums, evolution_length, 5)

    for i, pid in enumerate(tr_ids):
        pend_data = tr_data[tr_data['id_pendulum'] == pid]
        X_train[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
    
    y_train = X_train[:, :, :4].copy() # dim -> (n_pendulums, evolution_length, 4)

    # --- Val Data ---
    
    # Load validation data
    val_data = pd.read_csv(val_path)

    # Build X_val and y_val

    val_ids = val_data['id_pendulum'].unique().tolist()
    val_evolution_length = val_data[val_data['id_pendulum'] == val_ids[0]].shape[0]

    X_val = np.zeros((len(val_ids), val_evolution_length, 5))

    for i, pid in enumerate(val_ids):
        pend_data = val_data[val_data['id_pendulum'] == pid]
        X_val[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()

    y_val = X_val[:, :, :4].copy()
    
    # --- Model ---

    # Dynamic layer for pendulum dynamics
    dPend = DynLayer(
        b1_ref=0.1,
        fixed_b1=0.3, 
        b2_ref=0.1,
        fixed_b2 = 0.3,
        l1_ref=1.0, 
        fixed_l1 = 1.2,
        l2_ref=5, 
        fixed_l2=None,
        m1_ref=1.0, 
        fixed_m1 = 2.2,
        m2_ref=1.0,
        fixed_m2=1.8 
        )

    # Integration layer
    integration = RK4Model(dPend)

    # --- Loss function ---

    #loss_fn = physics_loss 
    loss_fn = nn.MSELoss()

    # --- Training ---

    curriculum_steps = [4, 8, 16, 32] # steps of horizon (dynamic during training)
    epochs_per_stage = [10, 10, 10, 10] 
    lrs = [1e-2, 1e-2, 1e-2, 1e-2]       

    early_stopper = EarlyStopping(patience=10)

    history_full = {"loss": [], "val_loss": []}

    for stage, (steps, eps, lr) in enumerate(zip(curriculum_steps, epochs_per_stage, lrs)):
        
        print(f"\n--- STAGE {stage+1}: Horizon {steps} steps, LR {lr} ---")
        
        # Stampa parametri imparati finora
        print(f"\nParams attuali: m1={dPend.get_m1().item():.3f}, l1={dPend.get_l1().item():.3f}, b1={dPend.get_b1().item():.3f}")
        print(f"\nParams attuali: m2={dPend.get_m2().item():.3f}, l2={dPend.get_l2().item():.3f}, b2={dPend.get_b2().item():.3f}")
        
        # Reset Early Stopping 
        early_stopper = EarlyStopping(patience=15)
        
        # Training
        hist = train_nn_model(
            model=integration,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            loss_fn=loss_fn,
            early_stopper=early_stopper,
            epochs=eps,
            lr=lr,
            weight_decay=0.0,
            batch_size=512, 
            device=my_device,
            verbose=True,
            horizon=steps 
        )
        
        # Concatena la storia per il plot finale
        if "loss" in hist:
            history_full["loss"].extend(hist["loss"])
            history_full["val_loss"].extend(hist["val_loss"])


    print("Parameters learned: ")
    print(f"\n\tb1: {dPend.get_b1().item():.4f}, b2: {dPend.get_b2().item():.4f}")
    print(f"\tl1: {dPend.get_l1().item():.4f}, l2: {dPend.get_l2().item():.4f}")
    print(f"\tm1: {dPend.get_m1().item():.4f}, m2: {dPend.get_m2().item():.4f}")  

    print("Ground truth parameters: ")
    print(f"\n\tb1: {b1_gt:.4f}, b2: {b2_gt:.4f}")
    print(f"\tl1: {l1_gt:.4f}, l2: {l2_gt:.4f}")
    print(f"\tm1: {m1_gt:.4f}, m2: {m2_gt:.4f}\n")

    
    # --- Plot training history ---
    
    plot_training_history(history = history_full, figsize=(6,10))

if __name__ == "__main__":

    # extract data paths from main arguments
    parser = argparse.ArgumentParser()

    # Argomenti opzionali con flag
    parser.add_argument('--train_path', type=str, default='data/training/train_lowfrict.csv')
    parser.add_argument('--val_path', type=str, default= 'data/training/val_lowfrict.csv')
    # parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    # parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    # parser.add_argument('--device', type=str, default='cuda', help='Device')
    # parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    main(train_path=args.train_path,
         val_path=args.val_path)

# 1. Architecture for the estimation of constant parameters

    # Architectures:
    # a. RNN-based architecture
    # b. UDEs


# 2 Architecture for the estimation of time-varying parameters

# 3. Architecture for the digital twin model

# ideas for later: 
# 1. Check for the regularization in the custom loss function: might be better adding all parameters [!!!]
# 2. Normalize?


# N.B. time step seems ok for Runge - Kutta integration 
# N.B. main problem in velocity estimation
# N.B. check for initialization
