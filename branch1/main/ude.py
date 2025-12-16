import torch
import pandas as pd
import numpy as np
from branch1.utils.training import (train_nn_model, plot_training_history, EarlyStopping, physics_loss, split_dataset, 
                                    KNGLDataset, save_model, load_model, evaluate_model, predict_model, pull_data_by_ids,
                                    average_discrepancy_perc, avg_param_difference, max_param_difference, log_recipe,
                                    turn_to_cosine_sine)       
from branch1.utils.architecture import DynLayer, EulerModel, RK4Model, DeepNN, device_setup
from branch1.utils.plots import plot_matrices_predictions, plot_dynamics_predictions
from torch import nn
import argparse
import os
import sys
import traceback

def main(
        train_path: str, 
        val_path: str,
        test_path: str,

        sin_cos_transform: bool,
        normalization: bool,
        fourier_features: bool,

        training: bool,
        patience: int,
        wd: float,
        batch_size: int,
        optimizer_type: str,
        scheduler_type: str,
        loss_type: str,

        epochs: int,
        lr: float,

        activation: str,

        version: str,

        verbose: bool,
        show_plots: bool
    ):

    # --- Training recipe ---

    training_recipe = {
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,

        'sin_cos_transform': sin_cos_transform,
        'normalization': normalization,
        'fourier_features': fourier_features,

        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'patience': patience,
        'wd': wd,
        'optimizer_type': optimizer_type,
        'scheduler_type': scheduler_type,
        'loss_type': loss_type,

        'activation': activation,

        'version': version,
    }

    # --- Save paths ---

    exp_folder = os.path.join('branch1/models', 'entries', version)
    os.makedirs(exp_folder, exist_ok=True) # create experiment folder if not exists

    model_path = os.path.join(exp_folder, 'deep_nn.pth')
    history_path = os.path.join(exp_folder, 'history.npy')
    log_path = os.path.join(exp_folder, 'training_recipe.txt')    
    
    plots_folder = os.path.join(exp_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # --- Device setup ---

    my_device = device_setup()

    # --- Sin Cos Transform ---
    if sin_cos_transform:           # use sin-cos transformation for angles 
        modify_func = turn_to_cosine_sine
        X_train_columns = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step']
        Y_train_columns = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2'] # TODO why not q1 q2?
        
    else:
        modify_func = None
        X_train_columns = ['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']
        Y_train_columns = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']
        

    # --- Train Data ---

    X_train, Y_train = pull_data_by_ids(data_path=train_path,
                                        data_splitter='id_pendulum',
                                        xcols = X_train_columns,
                                        ycols = Y_train_columns,
                                        modify_func = modify_func)
    


    # --- Val Data ---
    
    X_val, Y_val = pull_data_by_ids(data_path=val_path,
                                    data_splitter='id_pendulum',
                                    xcols = X_train_columns,
                                    ycols = Y_train_columns,
                                    modify_func = modify_func)
    


    # --- Test Data ---

    X_test, Y_test = pull_data_by_ids(data_path=test_path,
                                      data_splitter='id_pendulum',
                                      xcols = X_train_columns,
                                      ycols = Y_train_columns,
                                      modify_func = modify_func)
    

    
    # --- Model ---

    hidden_dims = [32, 64, 64, 32, 16] # TODO meno layer alta dimensionalita
    output_dim = 10
    
    deepNN = DeepNN(input_dim=len(X_train_columns), 
                    output_dim=10, 
                    hidden_dims=hidden_dims) # TODO Softplus SILU

    # --- Loss function ---

    loss_fn = nn.MSELoss()

    # --- Training or Loading ---

    early_stopper = EarlyStopping(patience=patience)
        
    ds_train = KNGLDataset(
        X_train.astype(np.float32), 
        Y_train.astype(np.float32), 
        horizon=1
    )
    ds_val = KNGLDataset(
        X_val.astype(np.float32), 
        Y_val.astype(np.float32), 
        horizon=1
    )
    ds_test = KNGLDataset(
        X_test.astype(np.float32), 
        Y_test.astype(np.float32), 
        horizon=1
    )

    # Training
    if training:

        hist = train_nn_model(
            model=deepNN,
            device=my_device,

            ds_train=ds_train,
            ds_val=ds_val,

            early_stopper=early_stopper,
            epochs=epochs,
            lr=lr,
            weight_decay=wd,
            batch_size=batch_size,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            loss_type=loss_type,

            verbose=verbose, 
        )

        save_model(model= deepNN, 
                   history= hist, 
                   model_path = model_path)
        

        log_recipe(recipe = training_recipe, log_path =log_path)

    # Loading
    else:

        deepNN, hist = load_model(model=deepNN, model_path = model_path)

    
    # --- Evaluation ---

    test_loss, ypred_metrics, ytrue_metrics = evaluate_model(model=deepNN, ds_test=ds_test, loss_fn=loss_fn, device=my_device)
    print(f"\nTest loss is: {test_loss:.4f}")
    print(f"\nMetrics")
    print(f"\nAverage parameters difference: {avg_param_difference(true_params=ytrue_metrics, pred_params=ypred_metrics)}")
    print(f"\nMax parameters difference: {max_param_difference(true_params=ytrue_metrics, pred_params=ypred_metrics)}")
    print(f"\nAverage discrepancy percentage: {average_discrepancy_perc(true_params=ytrue_metrics, pred_params=ypred_metrics)}")
    
    # --- Plot training history ---
        
    plot_training_history(
        history = hist,
        figsize=(10,6), 
        save_path = history_path,
        show_plot=show_plots
    )

    # --- Compare true vs predicted parameters of a certain pendulum ---
    
    for i in range(5):

        X_test_i = X_test[i,:,:]
        Y_test_i = Y_test[i,:,:]
        
        Y_pred_i = predict_model(model=deepNN,
                                X_test=X_test_i,
                                device=my_device)
        
        mat_comp_path_i = os.path.join(plots_folder, f'mat_comp_{i}.png')

        plot_matrices_predictions(
            y_true=Y_test_i, 
            y_pred=Y_pred_i,
            save_path = mat_comp_path_i,
            show_plot=show_plots
        )

    # --- Compare dynamics ---

    if show_plots:

        states_col = ['q1', 'q1_dot', 'q2', 'q2_dot']
        dynamics_col = ['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot']
        
        St_In_test2, Y_test2 = pull_data_by_ids(data_path=test_path,
                                      data_splitter='id_pendulum',
                                      xcols = states_col + X_train_columns,
                                      ycols = dynamics_col,
                                      modify_func = modify_func)
        
        Xstates_test2 = St_In_test2[:,:, :len(states_col)]  
        Xinputs_test2 = St_In_test2[:,:, len(states_col):]
        
        for i in range(5):

            evolution_i_path = os.path.join(plots_folder, f'evolution_comparison_{i}.png')

            plot_dynamics_predictions(states = Xstates_test2[i,:,:],
                                      inputs =  Xinputs_test2[i,:,:],
                                      true_states_dot = Y_test2[i,:,:], 
                                      model = deepNN,
                                      save_path = evolution_i_path, 
                                      device=my_device)
    
if __name__ == "__main__":

    # extract data paths from main arguments
    parser = argparse.ArgumentParser()

    # Argomenti opzionali con flag
    parser.add_argument('--train_path', type=str, default='branch1/data_enhanced/training/train_lowfrict.csv')
    parser.add_argument('--val_path', type=str, default='branch1/data_enhanced/training/val_lowfrict.csv')
    parser.add_argument('--test_path', type=str, default='branch1/data_enhanced/training/test_lowfrict.csv')
    
    parser.add_argument('--sin_cos_transform', action='store_true', default=False)
    parser.add_argument('--normalization', action='store_true', default=False)
    parser.add_argument('--fourier_features', type=int, default=-1)

    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--loss_type', type=str, default='mse')

    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--activation', type=str, default='relu')

    parser.add_argument('--version', type=str, default='v1')

    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--show_plots', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.training:
        print("Training mode")
    else:
        print("Evaluation mode")
    
    try:
        main(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,

            sin_cos_transform=args.sin_cos_transform,
            normalization=args.normalization,
            fourier_features=args.fourier_features,

            training=args.training,
            patience=args.patience,
            wd=args.wd,
            batch_size=args.batch_size,
            optimizer_type=args.optimizer_type,
            scheduler_type=args.scheduler_type,
            loss_type=args.loss_type,
            
            epochs=args.epochs,
            lr=args.lr,

            activation=args.activation,

            version=args.version,

            verbose=args.verbose,
            show_plots=args.show_plots    
        )
    except Exception as e:
        print("\n" + "!"*60)
        print(f" [CRITICAL ERROR] Run failed for version {args.version} in entries prediction")
        print("!"*60 + "\n")
        
        # Stampa l'errore a video
        traceback.print_exc()

        # --- LOGGING SU FILE (Fondamentale per batch notturni) ---
        # Crea un file 'error_log.txt' che accumula tutti gli errori
        with open("batch_errors_log.txt", "a") as f:
            f.write(f"\n{'='*30}\n")
            f.write(f"TIMESTAMP: {pd.Timestamp.now()}\n")
            f.write(f"CONFIG: Ver={args.version}, BS={args.batch_size}, SinCos={args.sin_cos_transform}, Norm={args.normalization}\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write("-" * 10 + " TRACEBACK " + "-" * 10 + "\n")
            traceback.print_exc(file=f)
            f.write(f"{'='*30}\n")
        
        # Usciamo con codice di errore 1 così il .bat sa che è fallito (ma continua)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[USER INTERRUPT] Stopping script via Ctrl+C...")
        sys.exit(0)

