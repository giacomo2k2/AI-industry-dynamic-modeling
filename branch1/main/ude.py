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

def main(train_path: str, 
         val_path: str,
         test_path: str,

         version: str,
         model_path: str,
         model_name: str,

         training: bool,
         patience: int,
         sin_cos_transform: bool,
         epochs: int,
         batch_size: int,
         lr: float,
         wd: float,
            optimizer_type: str,
            scheduler_type: str,
            loss_type: str,


         verbose: bool,
         show_plots: bool):
    
    # --- Device setup ---

    my_device = device_setup()

    # --- Sin Cos Transform ---
    if sin_cos_transform:
        modify_func = turn_to_cosine_sine
    else:
        modify_func = None

    # --- Train Data ---

    X_train, Y_train = pull_data_by_ids(data_path=
                                        train_path,
                                        data_splitter='id_pendulum',
                                        xcols = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step'],
                                        ycols = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2'],
                                        modify_func = modify_func)
    


    # --- Val Data ---
    
    X_val, Y_val = pull_data_by_ids(data_path=val_path,
                                    data_splitter='id_pendulum',
                                    xcols = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step'],
                                    ycols = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2'],
                                    modify_func = modify_func)
    


    # --- Test Data ---

    X_test, Y_test = pull_data_by_ids(data_path=test_path,
                                      data_splitter='id_pendulum',
                                      xcols = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step'],
                                      ycols = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2'],
                                      modify_func = modify_func)
    

    
    # --- Model ---

    if sin_cos_transform:
        input_dim = 7
    else:
        input_dim = 5

    hidden_dims = [32, 64, 64, 32, 16] # TODO meno layer alta dimensionalita
    output_dim = 10
    
    deepNN = DeepNN(input_dim=input_dim, 
                    output_dim=10, 
                    hidden_dims=hidden_dims) # TODO Softplus SILU

    # --- Loss function ---

    loss_fn = nn.MSELoss()

    # --- Training or Loading ---

    early_stopper = EarlyStopping(patience=patience)
        
    ds_train = KNGLDataset(X_train.astype(np.float32), Y_train.astype(np.float32), horizon=1)
    ds_val = KNGLDataset(X_val.astype(np.float32), Y_val.astype(np.float32), horizon=1)
    ds_test = KNGLDataset(X_test.astype(np.float32), Y_test.astype(np.float32), horizon=1)

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
                   model_path = model_path+version+'/'+model_name)
        
        training_recipe = {
            'train_path': train_path,
            'val_path': val_path,
            'test_path': test_path,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'sin_cos_transform': sin_cos_transform,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'wd': wd,
            'patience': patience,
            'model_name': model_name
        }

        log_recipe(recipe = training_recipe, log_path = model_path+version+'/training_recipe.txt')

    # Loading
    else:

        deepNN, hist = load_model(model=deepNN, model_path = model_path+version+'/'+model_name)

    
    # --- Evaluation ---

    test_loss, ypred_metrics, ytrue_metrics = evaluate_model(model=deepNN, ds_test=ds_test, loss_fn=loss_fn, device=my_device)
    print(f"\nTest loss is: {test_loss:.4f}")
    print(f"\nMetrics")
    print(f"\nAverage parameters difference: {avg_param_difference(true_params=ytrue_metrics, pred_params=ypred_metrics)}")
    print(f"\nMax parameters difference: {max_param_difference(true_params=ytrue_metrics, pred_params=ypred_metrics)}")
    print(f"\nAverage discrepancy percentage: {average_discrepancy_perc(true_params=ytrue_metrics, pred_params=ypred_metrics)}")
    
    # --- Plot training history ---

    if show_plots:
        
        plot_training_history(history = hist , 
                              figsize=(10,6), 
                              save_path = model_path+version+f"/plots/training_history")

    # --- Compare true vs predicted parameters of a certain pendulum ---
    
    if show_plots:

        X_test_0 = X_test[0,:,:]
        Y_test_0 = Y_test[0,:,:]
        
        Y_pred_0 = predict_model(model=deepNN,
                                X_test=X_test_0,
                                device=my_device)

        plot_matrices_predictions(y_true=Y_test_0, 
                                  y_pred=Y_pred_0,
                                  save_path = model_path+version+f"/plots/mat_pred_{0}")

    # --- Compare dynamics ---

    if show_plots:

        states_col = ['q1', 'q1_dot', 'q2', 'q2_dot']
        
        if sin_cos_transform:
            inputs_col = ['q1_sin', 'q1_cos', 'q1_dot', 'q2_sin', 'q2_cos', 'q2_dot', 'time_step']
        else:
            inputs_col = ['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']
        
        St_In_test2, Y_test2 = pull_data_by_ids(data_path=test_path,
                                      data_splitter='id_pendulum',
                                      xcols = states_col + inputs_col,
                                      ycols = ['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot'],
                                      modify_func = modify_func)
        
        Xstates_test2 = St_In_test2[:,:, :4]  
        Xinputs_test2 = St_In_test2[:,:, 4:]
        
        for i in range(5):

            plot_dynamics_predictions(states = Xstates_test2[i,:,:],
                                      inputs =  Xinputs_test2[i,:,:],
                                      true_states_dot = Y_test2[i,:,:], 
                                      model = deepNN,
                                      save_path = model_path+version+f"/plots/dyn_pred_{i}", 
                                      device=my_device)
    
if __name__ == "__main__":

    # extract data paths from main arguments
    parser = argparse.ArgumentParser()

    # Argomenti opzionali con flag
    parser.add_argument('--train_path', type=str, default='branch1/data_enhanced/training/train_lowfrict.csv')
    parser.add_argument('--val_path', type=str, default='branch1/data_enhanced/training/val_lowfrict.csv')
    parser.add_argument('--test_path', type=str, default='branch1/data_enhanced/training/test_lowfrict.csv')

    parser.add_argument('--training', action='store_true', default=True)
    parser.add_argument('--sin_cos_transform', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--loss_type', type=str, default='mse')

    parser.add_argument('--verbose', action='store_true', default=True)

    parser.add_argument('--version', type=str, default='v3')
    parser.add_argument('--model_path', type=str, default='branch1/models/ude/')
    parser.add_argument('--model_name', type=str, default='ude_model.pth')

    parser.add_argument('--show_plots', action='store_true', default=True)
    
    args = parser.parse_args()

    if args.training:
        print("Training mode")
    else:
        print("Evaluation mode")
    

    main(
        train_path=args.train_path,
         val_path=args.val_path,
         test_path=args.test_path,

         version=args.version,
         model_path=args.model_path,
         model_name=args.model_name,

         training=args.training,

         sin_cos_transform=args.sin_cos_transform,

         epochs=args.epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         wd=args.wd,
         optimizer_type=args.optimizer_type,
         scheduler_type=args.scheduler_type,
         loss_type=args.loss_type,

         patience=args.patience,
         verbose=args.verbose,
         show_plots=args.show_plots)

