import numpy as np
import torch

import matplotlib.pyplot as plt 


def plot_matrices_predictions(y_true, y_pred):
    
    plt.figure(figsize=(12, 8))
    param_names = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.plot(y_true[:, i], label='True')
        plt.plot(y_pred[:, i], label='Predicted', linestyle='--')
        plt.title(param_names[i])
        plt.legend()
    plt.tight_layout()
    plt.show()
    

import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_results(states, true_states_dot, model, u=None, device="cpu", dt=0.01):
    """
    states: (N, 4) -> [q1, q1_dot, q2, q2_dot]
    true_states_dot: (N, 4) -> [q1_dot, q1_ddot, q2_dot, q2_ddot] (Ground Truth)
    model: Rete neurale che predice i parametri fisici
    u: (N, 2) o None -> Input di controllo (coppia)
    """
    model.eval()
    
    pred_states_dot = []
    
    # Liste per il plotting (trasposizione per facilità)
    true_list = []
    pred_list = []

    with torch.no_grad():
        for i, state in enumerate(states):

            #print(f"state shape: {state.shape}")
            # --- 1. Preparazione Input ---
            # Input controllo u corrente (se None, è zero)
            u_curr = np.zeros((2, 1))
            if u is not None:
                u_curr = u[i].reshape(2, 1)

            # Tensore stato per il modello
            xb = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # --- 2. Predizione dei Parametri ---
            y_pred = model(xb).squeeze(0)
            
            vals = y_pred.detach().cpu().numpy().flatten()
            
            # Ora vals è un semplice array numpy di 10 float. 
            # Non serve più .item(), accedi direttamente agli indici.
            M11, M12 = vals[0], vals[1]
            M21, M22 = vals[2], vals[3]
            c1, c2   = vals[4], vals[5]
            D11, D22 = vals[6], vals[7]
            G1, G2   = vals[8], vals[9]

            # --- 3. Costruzione Matrici Stimate ---
            # Matrice di Inerzia M
            M_est = np.array([[M11, M12],
                              [M21, M22]])
            
            # Vettore Coriolis/Centrifuga C (nota: nel tuo codice dynamics era 'c')
            C_est = np.array([[c1], 
                              [c2]])
            
            # Matrice Damping D
            D_est = np.array([[D11, 0],
                              [0, D22]])
            
            # Vettore Gravità G
            G_est = np.array([[G1], 
                              [G2]])

            # --- 4. Calcolo della Dinamica (Forward Dynamics) ---
            # Estraiamo le velocità dallo stato attuale [q1, q1_dot, q2, q2_dot]
            q1_dot = state[1]
            q2_dot = state[3]
            
            q_dot_vec = np.array([[q1_dot], 
                                  [q2_dot]])
            
            # print(f"D_est shape: {D_est.shape}")
            # print(f"Q_dot_vec shape: {q_dot_vec.shape}")

            # Equazione del moto: M * q_ddot + C + D*q_dot + G = u
            # Quindi: M * q_ddot = u - C - D*q_dot - G
            # Termine noto b:
            b = u_curr - C_est - (D_est @ q_dot_vec) - G_est
            
            # Risolviamo per le accelerazioni (q_ddot)
            # Gestiamo eventuali matrici singolari con un try-except o pseudo-inversa
            try:
                q_ddot_est = np.linalg.solve(M_est, b)
            except np.linalg.LinAlgError:
                q_ddot_est = np.linalg.pinv(M_est) @ b

            q1_ddot_est = float(q_ddot_est[0, 0])
            q2_ddot_est = float(q_ddot_est[1, 0])

            # --- 5. Assemblaggio del vettore states_dot predetto ---
            # La struttura deve matchare true_states_dot: [q1_dot, q1_ddot, q2_dot, q2_ddot]
            # Nota: q1_dot e q2_dot sono noti dallo stato precedente, 
            # ma qui stiamo predicendo la derivata completa.
            current_pred_dot = np.array([q1_dot, q1_ddot_est, q2_dot, q2_ddot_est])
            
            pred_states_dot.append(current_pred_dot)
            
            # Salviamo per i plot
            true_list.append(true_states_dot[i])
            pred_list.append(current_pred_dot)

    # Conversione in numpy array per slicing
    true_arr = np.array(true_list)
    pred_arr = np.array(pred_list)
    base_states = np.array([[states[:,0], states[:,2]]])
    # --- 6. Plotting --
    #print(true_arr.shape)
    #print(pred_arr.shape)
    plot_comparison(true_arr, pred_arr, base_states)

    return np.array(pred_states_dot)



def plot_comparison(true_derivs, pred_derivs, states):
    """
    Plotta il confronto completo della dinamica su una griglia 2x3.
    
    Args:
        true_derivs: (N, 4) -> [q1_dot, q1_ddot, q2_dot, q2_ddot] (Ground Truth)
        pred_derivs: (N, 4) -> [q1_dot, q1_ddot, q2_dot, q2_ddot] (Predetto)
        states:      (N, 4) -> [q1, q1_dot, q2, q2_dot] (Input States - per le posizioni)
    """
    
    # Assicuriamoci che siano numpy array
    true_derivs = np.array(true_derivs)
    pred_derivs = np.array(pred_derivs)
    states = np.array(states)
    
    time_steps = np.arange(len(true_derivs))
    
    # Creiamo una griglia 2 righe (Joint 1, Joint 2) x 3 colonne (Pos, Vel, Acc)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dynamics Analysis: Position (Input), Velocity & Acceleration (Output)', fontsize=16)

    # --- Definizioni Variabili per facilitare il plotting ---
    # Struttura dati per iterare facilmente
    # Ogni tupla: (Titolo, dati_ground_truth, dati_predetti, is_input_state)
    
    # RIGA 1: JOINT 1
    # Posizione q1 (Input)
    d_q1 = states[:, 0].squeeze()
    # Velocità q1_dot (Confronto) - Indice 0 in true_derivs
    d_q1_dot_true = true_derivs[:, 0]
    d_q1_dot_pred = pred_derivs[:, 0]
    # Accelerazione q1_ddot (Confronto) - Indice 1 in true_derivs
    d_q1_ddot_true = true_derivs[:, 1]
    d_q1_ddot_pred = pred_derivs[:, 1]

    # RIGA 2: JOINT 2
    # Posizione q2 (Input)
    d_q2 = states[:, 1].squeeze()
    # Velocità q2_dot (Confronto) - Indice 2 in true_derivs
    d_q2_dot_true = true_derivs[:, 2]
    d_q2_dot_pred = pred_derivs[:, 2]
    # Accelerazione q2_ddot (Confronto) - Indice 3 in true_derivs
    d_q2_ddot_true = true_derivs[:, 3]
    d_q2_ddot_pred = pred_derivs[:, 3]

    # --- Plotting Joint 1 (Prima Riga) ---
    
    # 1. Posizione q1
    axs[0, 0].plot(time_steps, d_q1, color='blue', label='Input State')
    axs[0, 0].set_title(r'Joint 1 Position ($q_1$)')
    axs[0, 0].set_ylabel('Rad')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)

    # 2. Velocità q1_dot
    axs[0, 1].plot(time_steps, d_q1_dot_true, color='black', alpha=0.6, linewidth=2, label='True')
    axs[0, 1].plot(time_steps, d_q1_dot_pred, color='red', linestyle='--', label='Pred')
    axs[0, 1].set_title(r'Joint 1 Velocity ($\dot{q}_1$)')
    axs[0, 1].set_ylabel('Rad/s')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Accelerazione q1_ddot
    axs[0, 2].plot(time_steps, d_q1_ddot_true, color='black', alpha=0.6, linewidth=2, label='True')
    axs[0, 2].plot(time_steps, d_q1_ddot_pred, color='red', linestyle='--', label='Pred')
    axs[0, 2].set_title(r'Joint 1 Acceleration ($\ddot{q}_1$)')
    axs[0, 2].set_ylabel('Rad/s²')
    axs[0, 2].legend()
    axs[0, 2].grid(True, linestyle=':', alpha=0.6)

    # --- Plotting Joint 2 (Seconda Riga) ---

    # 1. Posizione q2
    axs[1, 0].plot(time_steps, d_q2, color='blue', label='Input State')
    axs[1, 0].set_title(r'Joint 2 Position ($q_2$)')
    axs[1, 0].set_ylabel('Rad')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)

    # 2. Velocità q2_dot
    axs[1, 1].plot(time_steps, d_q2_dot_true, color='black', alpha=0.6, linewidth=2, label='True')
    axs[1, 1].plot(time_steps, d_q2_dot_pred, color='red', linestyle='--', label='Pred')
    axs[1, 1].set_title(r'Joint 2 Velocity ($\dot{q}_2$)')
    axs[1, 1].set_ylabel('Rad/s')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Accelerazione q2_ddot
    axs[1, 2].plot(time_steps, d_q2_ddot_true, color='black', alpha=0.6, linewidth=2, label='True')
    axs[1, 2].plot(time_steps, d_q2_ddot_pred, color='red', linestyle='--', label='Pred')
    axs[1, 2].set_title(r'Joint 2 Acceleration ($\ddot{q}_2$)')
    axs[1, 2].set_ylabel('Rad/s²')
    axs[1, 2].set_xlabel('Time Step')
    axs[1, 2].legend()
    axs[1, 2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()