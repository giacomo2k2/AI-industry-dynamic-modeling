import torch
import pandas as pd
import numpy as np
from branch1.training_utils import train_nn_model, plot_training_history, EarlyStopping, physics_loss, split_dataset, KNGLDataset, save_model, load_model, evaluate_model, predict_model
from branch1.architecture import DynLayer, EulerModel, RK4Model, DeepNN
from branch1.plots import plot_matrices_predictions, compare_results
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
         verbose: bool,
         model_path: str,
         show_plots: bool):


    # files = [
    #     'double_pendulum_lowfrict.csv',
    #     'double_pendulum_nofrict.csv',
    #     'double_pendulum_highfrict.csv']
    # 
    # for f in files:
    #     print(f"Splitting dataset for {f}...")
    #     df = pd.read_csv('branch1/data_enhanced/processed/' + f)
    #     sim_type = f.split('_')[2].split('.')[0]
    #     split_dataset(df, sim_type=sim_type, shuffle=False, splits_path='branch1/data_enhanced/training/')
    
    #exit()
    
    # --- Device setup ---

    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if my_device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_reserved(my_device)/1e9} GB")

    # --- Train Data ---

    # Load training data
    tr_data = pd.read_csv(train_path)

    # Build X_train and y_train

    tr_ids = tr_data['id_pendulum'].unique().tolist()
    tr_evolution_length = tr_data[tr_data['id_pendulum'] == tr_ids[0]].shape[0]


    X_train = np.zeros((len(tr_ids), tr_evolution_length, 5)) # dim -> (n_pendulums, evolution_length, 5)
    y_train = np.zeros((X_train.shape[0], X_train.shape[1], 10)) # dim -> (n_pendulums, evolution_length, matrices_params)

    for i, pid in enumerate(tr_ids):
        pend_data = tr_data[tr_data['id_pendulum'] == pid]
        X_train[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
        y_train[i, :, :] = pend_data[['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']].to_numpy()

    # --- Val Data ---
    
    # Load validation data
    val_data = pd.read_csv(val_path)

    # Build X_val and y_val
    val_ids = val_data['id_pendulum'].unique().tolist()
    val_evolution_length = val_data[val_data['id_pendulum'] == val_ids[0]].shape[0]


    X_val = np.zeros((len(val_ids), val_evolution_length, 5))
    y_val = np.zeros((X_val.shape[0], X_val.shape[1], 10))

    for i, pid in enumerate(val_ids):
        pend_data = val_data[val_data['id_pendulum'] == pid]
        X_val[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
        y_val[i, :, :] = pend_data[['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']].to_numpy()

    # --- Test Data ---

    # Load test data
    test_data = pd.read_csv(test_path)

    # Build X_test and y_test
    test_ids = test_data['id_pendulum'].unique().tolist()
    test_evolution_length = test_data[test_data['id_pendulum'] == test_ids[0]].shape[0]
    
    X_test = np.zeros((len(test_ids), test_evolution_length, 5))
    y_test = np.zeros((X_test.shape[0], X_test.shape[1], 10))

    for i, pid in enumerate(test_ids):
        pend_data = test_data[test_data['id_pendulum'] == pid]
        X_test[i, :, :] = pend_data[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy()
        y_test[i, :, :] = pend_data[['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']].to_numpy()
    
    # --- Model ---

    #building of deepNN
    deepNN = DeepNN(input_dim=5, 
                    output_dim=10, 
                    hidden_dims=[16, 32, 16])

    # --- Loss function ---

    loss_fn = nn.MSELoss()

    # --- Training or Loading ---

    
    early_stopper = EarlyStopping(patience=patience)
        
    ds_train = KNGLDataset(X_train.astype(np.float32), y_train.astype(np.float32), time_steps=1)
    ds_val = KNGLDataset(X_val.astype(np.float32), y_val.astype(np.float32), time_steps=1)
    ds_test = KNGLDataset(X_test.astype(np.float32), y_test.astype(np.float32), time_steps=1)

    # Training
    if training:

        hist = train_nn_model(
            model=deepNN,
            ds_train=ds_train,
            ds_val=ds_val,
            loss_fn=loss_fn,
            early_stopper=early_stopper,
            epochs=epochs,
            lr=lr,
            weight_decay=wd,
            batch_size=batch_size, 
            device=my_device,
            verbose=verbose, 
        )

        save_model(model= deepNN, 
                   history= hist, 
                   model_path=model_path)

    # Loading
    else:

        deepNN, hist = load_model(model=deepNN, model_path=model_path)

    
    # --- Evaluation ---

    test_loss = evaluate_model(model=deepNN, ds_test=ds_test, loss_fn=loss_fn, device=my_device)
    print(f"\nTest loss is: {test_loss:.4f}")
    
    # --- Plot training history ---

    if show_plots:
        
        plot_training_history(history = hist , figsize=(6,10))

    # --- Compare true vs predicted parameters of a certain pendulum ---
    
    if show_plots:

        pend_data_test_0 = test_data[test_data['id_pendulum'] == test_ids[0]]

        X_test_0 = pend_data_test_0[['q1', 'q1_dot', 'q2', 'q2_dot', 'time_step']].to_numpy().astype(np.float32)
        y_test_0 = pend_data_test_0[['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']].to_numpy().astype(np.float32)
        
        y_pred_0 = predict_model(model=deepNN,
                                X_test=X_test_0,
                                device=my_device)

        plot_matrices_predictions(y_true=y_test_0, y_pred=y_pred_0)

    # --- Compare dynamics ---

    if show_plots:
        Y_Ground_truth_test = pend_data_test_0[['q1_dot', 'q1_dot_dot', 'q2_dot', 'q2_dot_dot']].to_numpy().astype(np.float32)
        #print(X_test.shape)
        #print(Y_Ground_truth_test.shape)
        compare_results(X_test_0, Y_Ground_truth_test, model = deepNN, device=my_device)
    



if __name__ == "__main__":

    # extract data paths from main arguments
    parser = argparse.ArgumentParser()

    # Argomenti opzionali con flag
    parser.add_argument('--train_path', type=str, default='branch1/data_enhanced/training/train_lowfrict.csv')
    parser.add_argument('--val_path', type=str, default='branch1/data_enhanced/training/val_lowfrict.csv')
    parser.add_argument('--test_path', type=str, default='branch1/data_enhanced/training/test_lowfrict.csv')

    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--verbose', action='store_true', default=True)

    parser.add_argument('--model_path', type=str, default='branch1/models/deepNN/best_deepNN.pth')

    parser.add_argument('--show_plots', action='store_true', default=True)
    
    args = parser.parse_args()

    if args.training:
        print("Training mode")
    else:
        print("Evaluation mode")
    

    main(train_path=args.train_path,
         val_path=args.val_path,
         test_path=args.test_path,
         training=args.training,
         epochs=args.epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         wd=args.wd,
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
