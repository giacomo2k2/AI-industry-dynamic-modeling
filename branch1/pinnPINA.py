import torch
import pandas as pd
import numpy as np
from branch1.training_utils import train_nn_model, plot_training_history, EarlyStopping, KNGLDataset, save_model, load_model, evaluate_model, predict_model
from branch1.architecture import DeepNN, DoppioPendolo, NormalizedDoublePendulum
from branch1.plots import plot_matrices_predictions, compare_results, plot_results_PINA, check_physics_PINA
from torch import nn
import argparse
from pina import LabelTensor
from pina.model import FeedForward
from pina.solver import PINN 
from pina.trainer import Trainer
from sklearn.metrics import mean_squared_error

def main(train_path: str, 
         val_path: str,
         test_path: str,
         training: bool,
         epochs: int,
         batch_size: int,
         normalization: bool,
         lr: float,
         wd: float,
         patience: int,
         verbose: bool,
         model_path: str,
         show_plots: bool):

    # --- Train Data ---

    tr_data = pd.read_csv(train_path)

    # Parametri fisici (dal primo pendolo)
    m1 = tr_data['m1'].iloc[0]
    m2 = tr_data['m2'].iloc[0]
    l1 = tr_data['l1'].iloc[0]
    l2 = tr_data['l2'].iloc[0]
    b1 = tr_data['b1'].iloc[0]
    b2 = tr_data['b2'].iloc[0]

    tr_ids = tr_data['id_pendulum'].unique().tolist()
    tr_evolution_length = tr_data[tr_data['id_pendulum'] == tr_ids[0]].shape[0]

    # Preallochiamo i tensori
    X_all = np.zeros((len(tr_ids), tr_evolution_length, 1)) 
    y_all = np.zeros((len(tr_ids), tr_evolution_length, 4)) 

    for i, pid in enumerate(tr_ids):
        pend_data = tr_data[tr_data['id_pendulum'] == pid]
        X_all[i, :, :] = pend_data[['time_step']].to_numpy()
        
        # IMPORTANTE: Riordiniamo le colonne per matchare la classe DoppioPendolo
        # DoppioPendolo si aspetta: [theta1, theta2, omega1, omega2]
        # Il tuo CSV ha: q1, q1_dot, q2, q2_dot -> Quindi prendiamo: q1, q2, q1_dot, q2_dot
        y_all[i, :, :] = pend_data[['q1', 'q2', 'q1_dot', 'q2_dot']].to_numpy()

    # --- Select 1 pendulum ---

    t_train = X_all[0] 
    y_train = y_all[0] 

    # --- Create PINN Problem ---

    if normalization:

        t_scale = np.max(np.abs(t_train))
        y_scale = np.max(np.abs(y_train), axis=0) + 1e-6

        t_train_norm = t_train / t_scale
        y_train_norm = y_train / y_scale

        problem = NormalizedDoublePendulum(
            train_data_input=t_train_norm, 
            train_data_output=y_train_norm,
            scale_t=t_scale,
            scale_y=y_scale,
            m1=m1, m2=m2, l1=l1, l2=l2, b1=b1, b2=b2
        )
    
    else:

        problem = DoppioPendolo(
            train_data_input=t_train, 
            train_data_output=y_train,
            m1=m1, m2=m2, l1=l1, l2=l2, b1=b1, b2=b2
        )

    # --- Create PINN Model ---

    model = FeedForward(
        input_dimensions=1, 
        output_dimensions=4, 
        layers=[40, 40, 40, 40],
        func=torch.nn.Tanh 
    )

    solver = PINN(problem=problem, model=model)

    # --- Training or Loading ---

    if training:

        trainer = Trainer(solver=solver, max_epochs=epochs, batch_size=batch_size, accelerator='auto')

        trainer.train()

    else:

        pass

    # --- Evaluation ---
    
    model.eval()

    if normalization:
        
        input_eval = LabelTensor(torch.tensor(t_train_norm).float(), labels=['t'])
        with torch.no_grad():
            preds_norm = model(input_eval).numpy()

        preds = preds_norm * y_scale

    else: 

        input_eval = LabelTensor(torch.tensor(t_train).float(), labels=['t'])
        with torch.no_grad():
            preds = model(input_eval).numpy()
    
    mse = mean_squared_error(y_train, preds)
    print(f"MSE Totale (Ground Truth vs PINN): {mse:.6e}")
    
    # --- Plots ---

    if show_plots:

        plot_results_PINA(t_train, y_train, preds)
    
    if show_plots:
        
        check_physics_PINA(problem, model, input_eval, preds)
    



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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--normalization', action='store_true', default=True)

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
         normalization=args.normalization,
         lr=args.lr,
         wd=args.wd,
         patience=args.patience,
         verbose=args.verbose,
         model_path=args.model_path,
         show_plots=args.show_plots)

