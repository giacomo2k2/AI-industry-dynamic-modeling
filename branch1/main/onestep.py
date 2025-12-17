import torch
import pandas as pd
import numpy as np
from branch1.utils.training import (train_nn_model, plot_training_history, EarlyStopping, physics_loss, KNGLDataset, pull_data_by_ids,
                                    save_model, load_model, evaluate_model, predict_model, average_discrepancy_perc, avg_param_difference, 
                                    max_param_difference, log_recipe, turn_to_cosine_sine)
from branch1.utils.architecture import DynLayer, EulerModel, RK4Model, device_setup, DynUDE, DynNN, DynUDE_Informed, UDE_More_Informed, UDE_D_constant
from branch1.utils.plots import plot_evolutions_comparison, plot_matrices_predictions
from torch import nn
import argparse
from models import DoublePendulum
from simulation import Simulator
import os
import traceback
import sys
    
def main(
        train_path: str, 
        val_path: str,
        test_path: str,

        sin_cos_transform: bool,
        normalization: bool,
        fourier_features: int,
        sample_weights: bool,

        training: bool,
        patience: int,
        wd: float,
        batch_size: int,
        optimizer_type: str,
        scheduler_type: str,
        loss_type: str,

        epochs: int,
        lr: float,

        integrator: str,
        
        dynlayer: str,
        activation: str,
        
        fixed_physical_params: dict,
        physical_params_ref: dict,

        version: str,
        
        verbose: bool,
        show_plots: bool
    ):

    # --- Recipe ---
    
    training_recipe = {
            'tran_path': train_path,
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

            'dynlayer': dynlayer,
            'activation': activation,
        }
    
    if training:
        print("\n--- TRAINING RECIPE ---\n")
        for key, value in training_recipe.items():
            print(f"{key}: {value}")
        print()
    else:
        print("\n--- EVALUATION MODE ---\n")

    # --- Save paths ---

    exp_folder = os.path.join('branch1/models', 'onestep', dynlayer, version)
    os.makedirs(exp_folder, exist_ok=True) # create experiment folder if not exists

    model_path = os.path.join(exp_folder, 'rnn.pth')
    history_path = os.path.join(exp_folder, 'history.npy')
    log_path = os.path.join(exp_folder, 'training_recipe.txt')    
    
    plots_folder = os.path.join(exp_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # --- Device setup ---

    my_device = device_setup()

    # --- Ground truth ---
    
    tr_data = pd.read_csv(train_path)

    d11_gt = tr_data['D11'].iloc[0]
    d22_gt = tr_data['D22'].iloc[0]

    print(f"\nGround truth physical parameters from training set: D11 = {d11_gt}, D22 = {d22_gt}\n")
    # --- Sin-Cos, Normalization, Fourier Features ---

    if normalization:  # compute normalization map from training data and put into a dict
        normalization_map = {
            'q1_dot': {
                'mean': tr_data['q1_dot'].mean(),
                'std': tr_data['q1_dot'].std()
            },
            'q2_dot': {
                'mean': tr_data['q2_dot'].mean(),
                'std': tr_data['q2_dot'].std()
            },
            'q1_dot_dot': {
                'mean': tr_data['q1_dot_dot'].mean(),
                'std': tr_data['q1_dot_dot'].std()
            },
            'q2_dot_dot': {
                'mean': tr_data['q2_dot_dot'].mean(),
                'std': tr_data['q2_dot_dot'].std()
            }
        }
    else:
        normalization_map = None

    if sin_cos_transform:           # use sin-cos transformation for angles 
        modify_func = turn_to_cosine_sine
        X_train_columns = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot']
        X_feature_positions_map = {
            'q1_sin': 0,
            'q1_cos': 1,
            'q1_dot': 2,
            'q2_sin': 3,
            'q2_cos': 4,
            'q2_dot': 5,
        }
    else:
        modify_func = None
        X_train_columns = ['q1', 'q1_dot', 'q2', 'q2_dot']
        X_feature_positions_map = {
            'q1': 0,
            'q1_dot': 1,
            'q2': 2,
            'q2_dot': 3,
        }
    Y_train_columns = ['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot']
    Y_feature_position_map = {
        'q1_dot': 0,
        'q1_dot_dot': 1,
        'q2_dot': 2,
        'q2_dot_dot': 3
    }
    
    if fourier_features>0:
        extra_features_len = 2 * 2 * fourier_features  # q1 and q2 each with sin and cos components
    else:
        extra_features_len = 0    

    # --- Train Data ---

    X_train, Y_train =  pull_data_by_ids(data_path=train_path, 
                                         data_splitter = 'id_pendulum',
                                         xcols = X_train_columns,
                                         ycols = Y_train_columns,
                                         modify_func = modify_func) 
    
    # --- Sample Weights ---

    train_weights = None 
    
    if sample_weights:

        print("Number of samples in training data:", X_train.shape)
        
        state_vars = X_train_columns.copy()
        
        weights = np.ones(X_train.shape[0]*X_train.shape[1], dtype=np.float32) # inizializziamo i pesi a 1
        
        num_bins = 10

        for var in state_vars:
            
            var_idx = X_feature_positions_map[var]

            var_values = X_train[:, :, var_idx]

            var_values = var_values.flatten()

            counts, edges = np.histogram(var_values, bins=num_bins)
            
            bin_indices = np.digitize(var_values, edges[1:-1])
            
            sample_counts = counts[bin_indices]
            
            inv_freq = 1.0 / (sample_counts + 1e-6)
            
            weights *= inv_freq

        weights /= np.mean(weights)
        
        weights = np.clip(weights, a_min=0.1, a_max=10.0) # clipping to avoid extreme weights for a single outlier

        # reshape weights to match number of sequences
        weights = weights.reshape(X_train.shape[0], X_train.shape[1])
        
        # Assegniamo il risultato a train_weights
        train_weights = weights

        
    # --- Val Data ---
    
    X_val, Y_val =  pull_data_by_ids(data_path=val_path, 
                                         data_splitter = 'id_pendulum',
                                         xcols = X_train_columns,
                                         ycols = Y_train_columns,
                                         modify_func = modify_func) 
    
    X_test_columns = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step'] if sin_cos_transform else ['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']
    # --- Test Data ---
    X_test, Y_test =  pull_data_by_ids(data_path=test_path, 
                                       data_splitter = 'id_pendulum',
                                       xcols = X_test_columns,
                                       ycols = Y_train_columns,
                                       modify_func = modify_func)
    
    # --- Dynamic Layer ---

    # Every DynLayer will output derivatives (NOT NORMALIZED) in the following order 
    Y_feature_DynLayer_positions_map = {  #TODO: controllare che sia giusto 
            'q1_dot': 0,
            'q1_dot_dot': 1,
            'q2_dot': 2,
            'q2_dot_dot': 3
    }

    if physical_params_ref:
        print(f"\nUsing reference physical parameters for informed UDEs: {physical_params_ref}\n")
        physical_params_ref = None

    if physical_params_ref is None:
        print("Abbiamo scofitto la mafia")

    if dynlayer == 'nn':
        dyn_layer = DynNN(
            input_dim=len(X_train_columns) + extra_features_len,
            output_dim=2,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            normalization = normalization,
            normalization_map = normalization_map,
            activation=activation
        )

    elif dynlayer == 'ude':
        dyn_layer = DynUDE(
            input_dim=len(X_train_columns) + extra_features_len,
            output_dim=10,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            normalization = normalization,
            normalization_map = normalization_map,
            activation=activation
        )
    elif dynlayer == 'informed_ude':
        dyn_layer = DynUDE_Informed(
            input_dim=len(X_train_columns) + extra_features_len,
            output_dim=9,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            normalization = normalization,
            normalization_map = normalization_map,
            activation=activation
        )
    
    elif dynlayer == 'more_informed_ude':
        dyn_layer = UDE_More_Informed(
            input_dim=len(X_train_columns) + extra_features_len,
            output_dim=8,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            normalization = normalization,
            normalization_map = normalization_map,
            activation=activation
        )

    elif dynlayer == 'ude_d_constant':
        dyn_layer = UDE_D_constant(
            input_dim=len(X_train_columns) + extra_features_len,
            output_dim=6,

            d11_ref= 0.35,
            d22_ref= 0.35,

            fixed_d11 =  None,
            fixed_d22 =  None,

            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            normalization = normalization,
            normalization_map = normalization_map,
            activation=activation
        )
        
    else:
        print(f"\n\t[ERROR] '{dynlayer}' not recognized as a valid input for dynlayer \n")
        print("\tAvailable: \n\t -'nn' \n\t -'ude' \n\t -'informed_ude' \n\t -'ude_d_constant' \n")
        exit()


    # --- Integrator ---

    physical_state_positions_map = {
        'q1': 0,
        'q1_dot': 1,
        'q2': 2,
        'q2_dot': 3
    }

    # --- Training ---

    if training:
        
        early_stopper = EarlyStopping(patience=patience)

        # Datasets
        ds_train = KNGLDataset(
            X_train.astype(np.float32), 
            Y_train.astype(np.float32), 
            horizon=1,
            normalization_map=normalization_map,
            X_feature_positions_map=X_feature_positions_map,
            Y_feature_position_map=Y_feature_position_map,
            fourier_features=fourier_features,
            weights=train_weights
        )

        ds_val = KNGLDataset(
            X_val.astype(np.float32), 
            Y_val.astype(np.float32), 
            horizon=1,
            normalization_map=normalization_map,
            X_feature_positions_map=X_feature_positions_map,
            Y_feature_position_map=Y_feature_position_map,
            fourier_features=fourier_features,
            weights=None
        )
            
        # Training
        hist = train_nn_model(
            model=dyn_layer,
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

            verbose=verbose
        )
                
        # Save final model
        
        print("\nSaving model to path: ", model_path)
        save_model(model= dyn_layer, 
                   history= hist, 
                   model_path=model_path)
        
        # Log training recipe

        print("\nLogging training recipe to path: ", log_path)
        log_recipe(recipe = training_recipe, log_path = log_path)

                
    else:

        # Load pre-trained model
        print("\nLoading model from path: ", model_path)
        model, hist = load_model(model = dyn_layer, 
                                         model_path = model_path)
        
        model.to(my_device)
        model.eval()
                
    # --- Plot training history ---
    
    print("\nSaving training history plot to path: ", history_path)
    plot_training_history(
        history = hist, 
        figsize=(10,6),
        save_path=history_path,
        show_plot=show_plots
    )

    # --- Evaluation comparing matrices ---
    
    if dynlayer == 'ude' or dynlayer == 'informed_ude' or dynlayer == 'more_informed_ude' or dynlayer == 'ude_d_constant':

        X_matrices_test, Y_matrices_test = pull_data_by_ids(
            data_path=test_path,
            data_splitter='id_pendulum',
            xcols = X_train_columns, 
            ycols = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2'],
            modify_func = modify_func
        )

        for i in range(10):

            X_matrices_test_i_np = X_matrices_test[i,:,:]
            Y_matrices_test_i_np = Y_matrices_test[i,:,:]

            X_matrices_test_i = torch.Tensor(X_matrices_test_i_np).to(my_device)
            
            M, C, D, G = dyn_layer.predict_matrices(X_matrices_test_i)
            C = C.squeeze(-1)
            G = G.squeeze(-1)

            Y_matrices_pred_i_np = np.zeros_like(Y_matrices_test_i_np)
            Y_matrices_pred_i_np[:,0] = M[:,0,0].detach().cpu().numpy()  # M11
            Y_matrices_pred_i_np[:,1] = M[:,0,1].detach().cpu().numpy()  # M12
            Y_matrices_pred_i_np[:,2] = M[:,1,0].detach().cpu().numpy()  # M21
            Y_matrices_pred_i_np[:,3] = M[:,1,1].detach().cpu().numpy()  # M22
            Y_matrices_pred_i_np[:,4] = C[:,0].detach().cpu().numpy()    # c1
            Y_matrices_pred_i_np[:,5] = C[:,1].detach().cpu().numpy()    # c2
            Y_matrices_pred_i_np[:,6] = D[:,0,0].detach().cpu().numpy()  # D11
            Y_matrices_pred_i_np[:,7] = D[:,1,1].detach().cpu().numpy()  # D22
            Y_matrices_pred_i_np[:,8] = G[:,0].detach().cpu().numpy()    # G1
            Y_matrices_pred_i_np[:,9] = G[:,1].detach().cpu().numpy()    # G2
            
            mat_comp_path_i = os.path.join(plots_folder, f'mat_comp_{i}.png')

            plot_matrices_predictions(
                y_true=Y_matrices_test_i_np, 
                y_pred=Y_matrices_pred_i_np,
                save_path = mat_comp_path_i,
                show_plot=show_plots
            )

    print("\n--- Evaluating evolutions on test set ---\n")
    # --- Evaluating an evolution with predicted parameters ---

    if sin_cos_transform:           # use sin-cos transformation for angles 
        Y_test_columns = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot']
        Y_test_feature_position_map = {
            'q1_sin': 0,
            'q1_cos': 1,
            'q1_dot': 2,
            'q2_sin': 3,
            'q2_cos': 4,
            'q2_dot': 5
        }
    else:
        Y_test_columns = ['q1', 'q1_dot', 'q2', 'q2_dot']
        Y_test_feature_position_map = {
            'q1': 0,
            'q1_dot': 1,
            'q2': 2,
            'q2_dot': 3
        }
    
    X_test_feature_positions_map = {
            'q1_sin': 0,
            'q1_cos': 1,
            'q1_dot': 2,
            'q2_sin': 3,
            'q2_cos': 4,
            'q2_dot': 5,
            'time_step': 6
        }
    
    if integrator == 'rk4':
        model = RK4Model(
            f = dyn_layer,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_test_feature_positions_map,
            Y_feature_position_map = Y_test_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            physical_state_positions_map = physical_state_positions_map,
            normalization = normalization,
            normalization_map = normalization_map,
            auxiliary_input= (fourier_features>0)
        )
    elif integrator == 'euler':
        model = EulerModel(
            f = dyn_layer,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_test_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            physical_state_positions_map=physical_state_positions_map,
            normalization=normalization,
            normalization_map = normalization_map
        )
    else:
        print(f"\n\t[ERROR] '{integrator}' not recognized as a valid input for integrator \n")
        print("\tAvailable: \n\t -'rk4' \n\t -'euler' \n")
        exit()

    model.to(my_device)
    model.eval()

    X_test_evo, Y_test_evo =  pull_data_by_ids(data_path=test_path, 
                                       data_splitter = 'id_pendulum',
                                        xcols = X_test_columns,
                                        ycols = Y_test_columns,
                                        modify_func = modify_func)
    
    
    for i in range(10):  # evaluate on 3 different sequences

        X_test_i_np = X_test_evo[i:i+1,:,:]
        Y_test_i_np = Y_test_evo[i,:,:]   

        X_test_i = torch.Tensor(X_test_i_np).to(my_device)
        Y_pred_i = model(X_test_i).detach().cpu().numpy()[0,:,:]

        evolution_i_path = os.path.join(plots_folder, f'evolution_comparison_{i}.png')
        
        print(f"\nSaving evolution comparison plot for sequence {i} to path: ", plots_folder)
        plot_evolutions_comparison(
            true_states = Y_test_i_np,
            predicted_states = Y_pred_i,
            save_path = evolution_i_path,
            show_plot = show_plots,
            sin_cos_transform=sin_cos_transform
        )

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

    parser.add_argument('--sample_weights', action='store_true', default=False)

    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--loss_type', type=str, default='mse')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--fixed_d11', action='store_true', default=None)
    parser.add_argument('--fixed_d22', action='store_true', default=None)

    parser.add_argument('--d11_ref', type=float, default=5.0)
    parser.add_argument('--d22_ref', type=float, default=5.0)

    parser.add_argument('--integrator', type=str, default='rk4')
    
    parser.add_argument('--dynlayer', type=str, default='ude')
    parser.add_argument('--activation', type=str, default='relu')

    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--model_path', type=str, default='branch1/models/integrator/')
    parser.add_argument('--model_name', type=str, default='integrator_model.pth')

    parser.add_argument('--show_plots', action='store_true', default=False)
    
    args = parser.parse_args()

    fixed_physical_params = {
        'fixed_d11': args.fixed_d11,
        'fixed_d22': args.fixed_d22
    }
    
    physical_params_ref = {
        'd11_ref': args.d11_ref,
        'd22_ref': args.d22_ref
    }

    try:
        main(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        
        sin_cos_transform=args.sin_cos_transform,
        normalization=args.normalization,
        fourier_features=args.fourier_features,
        sample_weights=args.sample_weights,

        training=args.training,
        patience=args.patience,
        wd=args.wd,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        scheduler_type=args.scheduler_type,
        loss_type=args.loss_type,

        epochs=args.epochs,
        lr=args.lr,

        integrator=args.integrator,

        dynlayer=args.dynlayer,
        activation=args.activation,

        fixed_physical_params=fixed_physical_params,
        physical_params_ref=physical_params_ref,

        version=args.version,
        
        verbose=args.verbose,
        show_plots=args.show_plots
    )
    
    except Exception as e:
        print("\n" + "!"*60)
        print(f" [CRITICAL ERROR] Run failed for version {args.version} layer {args.dynlayer}")
        print("!"*60 + "\n")
        
        # Stampa l'errore a video
        traceback.print_exc()

        # --- LOGGING SU FILE (Fondamentale per batch notturni) ---
        # Crea un file 'error_log.txt' che accumula tutti gli errori
        with open("batch_errors_log.txt", "a") as f:
            f.write(f"\n{'='*30}\n")
            f.write(f"TIMESTAMP: {pd.Timestamp.now()}\n")
            f.write(f"CONFIG: Layer={args.dynlayer}, Ver={args.version}, BS={args.batch_size}, SinCos={args.sin_cos_transform}, Norm={args.normalization}\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write("-" * 10 + " TRACEBACK " + "-" * 10 + "\n")
            traceback.print_exc(file=f)
            f.write(f"{'='*30}\n")
        
        # Usciamo con codice di errore 1 così il .bat sa che è fallito (ma continua)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[USER INTERRUPT] Stopping script via Ctrl+C...")
        sys.exit(0)

    
         
