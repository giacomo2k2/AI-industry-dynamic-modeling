import torch
import pandas as pd
import numpy as np
from branch1.utils.training import (train_nn_model, plot_training_history, EarlyStopping, physics_loss, KNGLDataset, pull_data_by_ids,
                                    save_model, load_model, evaluate_model, predict_model, average_discrepancy_perc, avg_param_difference, 
                                    max_param_difference, log_recipe, turn_to_cosine_sine)
from branch1.utils.architecture import DynLayer, EulerModel, RK4Model, device_setup, DynUDE, DynNN, DynUDE_Informed
from branch1.utils.plots import plot_evolutions_comparison
from torch import nn
import argparse
from models import DoublePendulum
from simulation import Simulator
import os
import traceback # <--- Aggiungi questo import in alto o qui
import sys
    
def main(
        train_path: str, 
        val_path: str,
        test_path: str,

        sin_cos_transform: bool,
        normalization: bool,
        fourier_features: int,

        training: bool,
        patience: int,
        wd: float,
        batch_size: int,
        optimizer_type: str,
        scheduler_type: str,
        loss_type: str,

        curriculum_steps: list,
        epochs_per_stage: list,
        lrs: list,

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

            'curriculum_steps': curriculum_steps,
            'epochs_per_stage': epochs_per_stage,
            'lrs': lrs,
            'batch_size': batch_size,
            'patience': patience,
            'wd': wd,
            'optimizer_type': optimizer_type,
            'scheduler_type': scheduler_type,
            'loss_type': loss_type,

            'integrator': integrator,
            
            'dynlayer': dynlayer,
            'activation': activation,
            
            'fixed_physical_params': fixed_physical_params,
            'physical_params_ref': physical_params_ref
        }
    
    if training:
        print("\n--- TRAINING RECIPE ---\n")
        for key, value in training_recipe.items():
            print(f"{key}: {value}")
        print()
    else:
        print("\n--- EVALUATION MODE ---\n")

    # --- Save paths ---

    exp_folder = os.path.join('branch1/models', 'integrator', dynlayer, version)
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
    m1_gt = tr_data['m1'].iloc[0]
    m2_gt = tr_data['m2'].iloc[0]
    l1_gt = tr_data['l1'].iloc[0]
    l2_gt = tr_data['l2'].iloc[0]
    b1_gt = tr_data['b1'].iloc[0]
    b2_gt = tr_data['b2'].iloc[0]

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
        X_train_columns = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step']
        Y_train_columns = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot'] # TODO why not q1 q2?
        X_feature_positions_map = {
            'q1_sin': 0,
            'q1_cos': 1,
            'q1_dot': 2,
            'q2_sin': 3,
            'q2_cos': 4,
            'q2_dot': 5,
            'time_step': 6
        }
        Y_feature_position_map = {
            'q1_sin': 0,
            'q1_cos': 1,
            'q1_dot': 2,
            'q2_sin': 3,
            'q2_cos': 4,
            'q2_dot': 5
        }
    else:
        modify_func = None
        X_train_columns = ['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']
        Y_train_columns = ['q1', 'q1_dot', 'q2', 'q2_dot']
        X_feature_positions_map = {
            'q1': 0,
            'q1_dot': 1,
            'q2': 2,
            'q2_dot': 3,
            'time_step': 4
        }
        Y_feature_position_map = {
            'q1': 0,
            'q1_dot': 1,
            'q2': 2,
            'q2_dot': 3
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

    # --- Val Data ---
    
    X_val, Y_val =  pull_data_by_ids(data_path=val_path, 
                                         data_splitter = 'id_pendulum',
                                         xcols = X_train_columns,
                                         ycols = Y_train_columns,
                                         modify_func = modify_func) 
    
    # --- Test Data ---
    X_test, Y_test =  pull_data_by_ids(data_path=test_path, 
                                       data_splitter = 'id_pendulum',
                                       xcols = X_train_columns,
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

    if dynlayer == 'ode':
        dyn_layer = DynLayer(
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
            normalization = normalization,
            normalization_map = normalization_map,

            b1_ref = physical_params_ref['b1_ref'],
            b2_ref = physical_params_ref['b2_ref'],
            l1_ref = physical_params_ref['l1_ref'], 
            l2_ref = physical_params_ref['l2_ref'], 
            m1_ref = physical_params_ref['m1_ref'], 
            m2_ref = physical_params_ref['m2_ref'],

            fixed_b1 = b1_gt if fixed_physical_params['fixed_b1'] else None, 
            fixed_b2 = b2_gt if fixed_physical_params['fixed_b2'] else None,
            fixed_l1 = l1_gt if fixed_physical_params['fixed_l1'] else None,
            fixed_l2 = l2_gt if fixed_physical_params['fixed_l2'] else None,
            fixed_m1 = m1_gt if fixed_physical_params['fixed_m1'] else None,
            fixed_m2 = m2_gt if fixed_physical_params['fixed_m2'] else None, 
        )

    elif dynlayer == 'nn':
        dyn_layer = DynNN(
            input_dim=len(X_train_columns) - 1 + extra_features_len,
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
            input_dim=len(X_train_columns) - 1 + extra_features_len,
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
            input_dim=len(X_train_columns) - 1 + extra_features_len,
            output_dim=9,
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
        print("\tAvailable: \n\t -'ode' \n\t -'nn' \n\t -'ude' \n\t -'informed_ude' \n")
        exit()

    # --- Integrator ---

    physical_state_positions_map = {
        'q1': 0,
        'q1_dot': 1,
        'q2': 2,
        'q2_dot': 3
    }

    if integrator == 'rk4':
        model = RK4Model(
            f = dyn_layer,
            sin_cos_transform = sin_cos_transform,
            X_feature_positions_map = X_feature_positions_map,
            Y_feature_position_map = Y_feature_position_map,
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
            Y_feature_position_map = Y_feature_position_map,
            Y_feature_DynLayer_positions_map = Y_feature_DynLayer_positions_map,
            physical_state_positions_map=physical_state_positions_map,
            normalization=normalization,
            normalization_map = normalization_map
        )
    else:
        print(f"\n\t[ERROR] '{integrator}' not recognized as a valid input for integrator \n")
        print("\tAvailable: \n\t -'rk4' \n\t -'euler' \n")
        exit()



    # --- Training ---

    if training:

        history_full = {"loss": [], "val_loss": []}

        for stage, (steps, eps, lr) in enumerate(zip(curriculum_steps, epochs_per_stage, lrs)):
            
            print(f"\n--- STAGE {stage+1}: Horizon {steps} steps, LR {lr} ---")
            
            if dynlayer == 'ode':
                print(f"\nParams attuali: m1={dyn_layer.get_m1().item():.3f}, l1={dyn_layer.get_l1().item():.3f}, b1={dyn_layer.get_b1().item():.3f}")
                print(f"\nParams attuali: m2={dyn_layer.get_m2().item():.3f}, l2={dyn_layer.get_l2().item():.3f}, b2={dyn_layer.get_b2().item():.3f}")
            
            # Reset Early Stopping 
            early_stopper = EarlyStopping(patience=patience)

            # Datasets
            ds_train = KNGLDataset(
                X_train.astype(np.float32), 
                Y_train.astype(np.float32), 
                horizon=steps,
                normalization_map=normalization_map,
                X_feature_positions_map=X_feature_positions_map,
                Y_feature_position_map=Y_feature_position_map,
                fourier_features=fourier_features
            )

            ds_val = KNGLDataset(
                X_val.astype(np.float32), 
                Y_val.astype(np.float32), 
                horizon=steps,
                normalization_map=normalization_map,
                X_feature_positions_map=X_feature_positions_map,
                Y_feature_position_map=Y_feature_position_map,
                fourier_features=fourier_features
            )
            
            # Training
            hist = train_nn_model(
                model=model,
                device=my_device,
                
                ds_train=ds_train,
                ds_val=ds_val,
                
                early_stopper=early_stopper,
                epochs=eps,
                lr=lr,
                weight_decay=wd,
                batch_size=batch_size, 
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                loss_type=loss_type,

                verbose=verbose
            )
            
            # Concatena la storia per il plot finale
            if "loss" in hist:

                history_full["loss"].extend(hist["loss"])
                history_full["val_loss"].extend(hist["val_loss"])

                if dynlayer == 'ode':
                    #preparo i parametri per il calcolo delle discrepanze
                    pendulum_params = torch.tensor([
                        dyn_layer.get_m1(),
                        dyn_layer.get_m2(),
                        dyn_layer.get_l1(),
                        dyn_layer.get_l2(),
                        dyn_layer.get_b1(),
                        dyn_layer.get_b2()
                    ])
                    # metto i parametri gt in un tensore per il calcolo delle discrepanze
                    gt_params = torch.tensor([
                        m1_gt,
                        m2_gt,
                        l1_gt,
                        l2_gt,
                        b1_gt,
                        b2_gt
                    ], dtype=torch.float32)
                    
                    avg_diff = avg_param_difference(pendulum_params, gt_params)
                    max_diff = max_param_difference(pendulum_params, gt_params)
                    perc_diff = average_discrepancy_perc(pendulum_params, gt_params)
                
                    print(f"\nAverage parameters difference: {avg_diff}")
                    print(f"\nMax parameters difference: {max_diff}")
                    print(f"\nAverage discrepancy percentage: {perc_diff}")
                
        # Save final model
        
        print("\nSaving model to path: ", model_path)
        save_model(model= model, 
                   history= history_full, 
                   model_path=model_path)
        
        # Log training recipe

        print("\nLogging training recipe to path: ", log_path)
        log_recipe(recipe = training_recipe, log_path = log_path)

                
    else:

        # Load pre-trained model
        print("\nLoading model from path: ", model_path)
        model, history_full = load_model(model = model, 
                                         model_path = model_path)
                

    # --- Evaluation comparing GT ---

    if dynlayer == 'ode':

        print("Parameters learned: ")
        print(f"\n\tb1: {dyn_layer.get_b1().item():.4f}, b2: {dyn_layer.get_b2().item():.4f}")
        print(f"\tl1: {dyn_layer.get_l1().item():.4f}, l2: {dyn_layer.get_l2().item():.4f}")
        print(f"\tm1: {dyn_layer.get_m1().item():.4f}, m2: {dyn_layer.get_m2().item():.4f}")  

        print("Ground truth parameters: ")
        print(f"\n\tb1: {b1_gt:.4f}, b2: {b2_gt:.4f}")
        print(f"\tl1: {l1_gt:.4f}, l2: {l2_gt:.4f}")
        print(f"\tm1: {m1_gt:.4f}, m2: {m2_gt:.4f}\n")

    # --- Plot training history ---
    
    print("\nSaving training history plot to path: ", history_path)
    plot_training_history(
        history = history_full, 
        figsize=(10,6),
        save_path=history_path,
        show_plot=show_plots
    )

    # --- Evaluating an evolution with predicted parameters ---
    model.to(my_device)
    model.eval()

    for i in range(10):  # evaluate on 3 different sequences

        X_test_i_np = X_test[i:i+1,:,:]
        Y_test_i_np = Y_test[i,:,:]   

        X_test_i = torch.Tensor(X_test_i_np).to(my_device)
        Y_pred_i = model(X_test_i).detach().cpu().numpy()[0,:,:]

        evolution_i_path = os.path.join(plots_folder, f'evolution_comparison_{i}.png')
        
        print(f"\nSaving evolution comparison plot for sequence {i} to path: ", plots_folder)
        plot_evolutions_comparison(
            true_states = Y_test_i_np,
            predicted_states = Y_pred_i,
            save_path = evolution_i_path,
            show_plot = show_plots
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

    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--loss_type', type=str, default='mse')

    parser.add_argument('--curriculum_steps', type=int, nargs='+', 
                        default=[4, 8, 16, 32],
                        help='Lista degli step per l\'orizzonte temporale (es: 4 8 16 32)')
    parser.add_argument('--epochs_per_stage', type=int, nargs='+', 
                        default=[20, 20, 20, 20],
                        help='Lista delle epoche per ogni stage (es: 20 20 20 20)')
    parser.add_argument('--lrs', type=float, nargs='+', 
                        default=[1e-2, 1e-3, 1e-3, 1e-4],
                        help='Lista dei learning rate per ogni stage (es: 1e-2 1e-3 ...)')

    parser.add_argument('--integrator', type=str, default='rk4')
    
    parser.add_argument('--dynlayer', type=str, default='ude')
    parser.add_argument('--activation', type=str, default='relu')

    parser.add_argument('--fixed_l1', action='store_true', default=False)
    parser.add_argument('--fixed_l2', action='store_true', default=False)
    parser.add_argument('--fixed_m1', action='store_true', default=False)
    parser.add_argument('--fixed_m2', action='store_true', default=False)
    parser.add_argument('--fixed_b1', action='store_true', default=False)
    parser.add_argument('--fixed_b2', action='store_true', default=False)

    parser.add_argument('--l1_ref', type=float, default=1.0)
    parser.add_argument('--l2_ref', type=float, default=5.0)
    parser.add_argument('--m1_ref', type=float, default=1.0)
    parser.add_argument('--m2_ref', type=float, default=1.0)
    parser.add_argument('--b1_ref', type=float, default=0.1)
    parser.add_argument('--b2_ref', type=float, default=0.1)

    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--model_path', type=str, default='branch1/models/integrator/')
    parser.add_argument('--model_name', type=str, default='integrator_model.pth')

    parser.add_argument('--show_plots', action='store_true', default=False)
    
    args = parser.parse_args()

    len_steps = len(args.curriculum_steps)
    len_epochs = len(args.epochs_per_stage)
    len_lrs = len(args.lrs)

    if not (len_steps == len_epochs == len_lrs):
        print(" \n [ERROR] Mismatch in length of curriculum parameters:")
        print(f" - Curriculum Steps: {len_steps} elements {args.curriculum_steps}")
        print(f" - Epochs per Stage: {len_epochs} elements {args.epochs_per_stage}")
        print(f" - Learning Rates:   {len_lrs} elements {args.lrs}\n")
        exit()

    fixed_physical_params = {
        'fixed_l1': args.fixed_l1,
        'fixed_l2': args.fixed_l2,
        'fixed_m1': args.fixed_m1,
        'fixed_m2': args.fixed_m2,
        'fixed_b1': args.fixed_b1,
        'fixed_b2': args.fixed_b2
    }
    
    physical_params_ref = {
        'l1_ref': args.l1_ref,
        'l2_ref': args.l2_ref,
        'm1_ref': args.m1_ref,
        'm2_ref': args.m2_ref,
        'b1_ref': args.b1_ref,
        'b2_ref': args.b2_ref
    }

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

        curriculum_steps=args.curriculum_steps,
        epochs_per_stage=args.epochs_per_stage,
        lrs=args.lrs,

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

    
         
