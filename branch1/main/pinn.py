import torch
import pandas as pd
import numpy as np
from branch1.utils.training import (train_nn_model, plot_training_history, EarlyStopping, KNGLDataset, 
                                    save_model, load_model, evaluate_model, predict_model, LagrangianLoss,
                                    train_lagrangian_nn_model, evaluate_lagrangian_model)
from branch1.utils.architecture import DynLayer, EulerModel, RK4Model, DeepNN
from branch1.utils.plots import plot_matrices_predictions, compare_results, analyze_pendulum_dynamics
from branch1.utils.architecture import device_setup
from torch import nn
import argparse

def main(train_path: str, 
         val_path: str,
         test_path: str,
         training: bool,
         epochs: int,
         batch_size: int,
         lr: float,
         wd: float,
         patience: int,
         lambda_c: float,
         verbose: bool,
         model_path: str,
         show_plots: bool):
    
    # --- Device setup ---

    my_device = device_setup()

    # --- Train Data ---

    # Load training data
    tr_data = pd.read_csv(train_path)

    # Compute the acc scale from the data set
    acc_scale = np.max(np.abs(tr_data[['q1_dot_dot', 'q2_dot_dot']].to_numpy()), axis=0) + 1e-6

    # Extract pendulum features
    m1 = tr_data['m1'].unique().tolist()[0]
    m2 = tr_data['m2'].unique().tolist()[0]
    l1 = tr_data['l1'].unique().tolist()[0]
    l2 = tr_data['l2'].unique().tolist()[0]
    b1 = tr_data['b1'].unique().tolist()[0]
    b2 = tr_data['b2'].unique().tolist()[0]
    dt = tr_data['time_step'].iloc[1] - tr_data['time_step'].iloc[0]

    # Build X_train and y_train
    tr_ids = tr_data['id_pendulum'].unique().tolist()
    tr_evolution_length = tr_data[tr_data['id_pendulum'] == tr_ids[0]].shape[0]

    X_train = np.zeros((len(tr_ids), tr_evolution_length, 5)) # dim -> (n_pendulums, evolution_length, 5)
    y_train = np.zeros((X_train.shape[0], X_train.shape[1], 4)) # dim -> (n_pendulums, evolution_length, 4)

    for i, pid in enumerate(tr_ids):
        pend_data = tr_data[tr_data['id_pendulum'] == pid]
        X_train[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
        y_train[i, :, :] = pend_data[['q1', 'q2', 'q1_dot', 'q2_dot']].to_numpy()

    X_scale = np.max(np.abs(X_train), axis=(0, 1)) + 1e-6
    y_scale = np.max(np.abs(y_train), axis=(0, 1)) + 1e-6

    X_train_norm = X_train / X_scale
    y_train_norm = y_train / y_scale

    # --- Val Data ---
    
    # Load validation data
    val_data = pd.read_csv(val_path)

    # Build X_val and y_val
    val_ids = val_data['id_pendulum'].unique().tolist()
    val_evolution_length = val_data[val_data['id_pendulum'] == val_ids[0]].shape[0]

    X_val = np.zeros((len(val_ids), val_evolution_length, 5))
    y_val = np.zeros((X_val.shape[0], X_val.shape[1], 4))

    X_val_norm = X_val / X_scale
    y_val_norm = y_val / y_scale

    for i, pid in enumerate(val_ids):
        pend_data = val_data[val_data['id_pendulum'] == pid]
        X_val[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
        y_val[i, :, :] = pend_data[['q1', 'q2', 'q1_dot', 'q2_dot']].to_numpy()

    # --- Test Data ---

    # Load test data
    test_data = pd.read_csv(test_path)

    # Build X_test and y_test
    test_ids = test_data['id_pendulum'].unique().tolist()
    test_evolution_length = test_data[test_data['id_pendulum'] == test_ids[0]].shape[0]
    
    X_test = np.zeros((len(test_ids), test_evolution_length, 5))
    y_test = np.zeros((X_test.shape[0], X_test.shape[1], 4))

    X_test_norm = X_test / X_scale
    y_test_norm = y_test / y_scale

    for i, pid in enumerate(test_ids):
        pend_data = test_data[test_data['id_pendulum'] == pid]
        X_test[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
        y_test[i, :, :] = pend_data[['q1', 'q2', 'q1_dot', 'q2_dot']].to_numpy()
    
    
    # --- Model ---

    #building of deepNN
    deepNN = DeepNN(input_dim=4, 
                    output_dim=4, 
                    hidden_dims=[16, 32, 32, 16])
    
    model = RK4Model(f=deepNN, additional_derivatives=True)

    # --- Loss function ---

    loss_fn = nn.MSELoss()
    lagrangian_loss = LagrangianLoss(
        m1=m1,
        m2=m2,
        l1=l1,
        l2=l2,
        b1=b1,
        b2=b2,
        X_scale = X_scale,
        acc_scale = acc_scale
    )

    # --- Training or Loading ---
    curriculum_steps = [4, 8, 16, 32] # steps of horizon (dynamic during training)
    epochs_per_stage = [30, 30, 30, 30] 
    lrs = [1e-2, 1e-2, 1e-2, 1e-2]       
    lambda_cs = [0, 0.01, 0.1, 0.5]

    early_stopper = EarlyStopping(patience=10)

    history_full = {"loss": [], "val_loss": []}

    # Training
    if training:
        print("sono qui")

        for stage, (steps, eps, lr, lambda_cc) in enumerate(zip(curriculum_steps, epochs_per_stage, lrs, lambda_cs)):

            early_stopper = EarlyStopping(patience=patience)
                
            ds_train = KNGLDataset(X_train_norm.astype(np.float32), y_train_norm.astype(np.float32), time_steps=steps)
            ds_val = KNGLDataset(X_val_norm.astype(np.float32), y_val_norm.astype(np.float32), time_steps=steps)
            ds_test = KNGLDataset(X_test_norm.astype(np.float32), y_test_norm.astype(np.float32), time_steps=steps)

            hist = train_lagrangian_nn_model(
                model=model,
                ds_train=ds_train,
                ds_val=ds_val,
                loss_fn=loss_fn,
                early_stopper=early_stopper,
                epochs=eps,
                lr=lr,
                weight_decay=wd,
                batch_size=batch_size, 
                device=my_device,
                verbose=verbose,
                lagrangian_loss=lagrangian_loss,
                lambda_c = lambda_cc,
                show_grad=True,
                show_lrs=True       
            )

        save_model(model= model, 
                    history= hist, 
                    model_path=model_path)

    # Loading
    else:

        deepNN, hist = load_model(model=model, model_path=model_path)

    
    # --- Evaluation ---

    test_loss = evaluate_lagrangian_model(model=model, ds_test=ds_test, loss_fn=loss_fn, device=my_device)
    print(f"\nTest loss is: {test_loss:.4f}")
    
    # --- Plot training history ---

    if show_plots:
        
        plot_training_history(history = hist , figsize=(10,6))

        analyze_pendulum_dynamics(model=model,
                                    X_test=X_test.astype(np.float32),
                                    y_test=y_test.astype(np.float32),
                                    X_scale=X_scale,
                                    y_scale=y_scale,
                                    dt=dt,
                                    device=my_device)

if __name__ == "__main__":

    # extract data paths from main arguments
    parser = argparse.ArgumentParser()

    # Argomenti opzionali con flag
    parser.add_argument('--train_path', type=str, default='branch1/data_enhanced/training/train_lowfrict.csv')
    parser.add_argument('--val_path', type=str, default='branch1/data_enhanced/training/val_lowfrict.csv')
    parser.add_argument('--test_path', type=str, default='branch1/data_enhanced/training/test_lowfrict.csv')

    parser.add_argument('--training', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lambda_c', type=float, default=0.001)

    parser.add_argument('--verbose', action='store_true', default=True)

    parser.add_argument('--model_path', type=str, default='branch1/models/PINN/best_PINN.pth')

    parser.add_argument('--show_plots', action='store_true', default=True)
    
    args = parser.parse_args()

    main(train_path=args.train_path,
         val_path=args.val_path,
         test_path=args.test_path,
         training=args.training,
         epochs=args.epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         wd=args.wd,
         lambda_c=args.lambda_c,
         patience=args.patience,
         verbose=args.verbose,
         model_path=args.model_path,
         show_plots=args.show_plots)

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
