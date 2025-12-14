import numpy as np
import torch
import matplotlib.pyplot as plt 
from pina import LabelTensor
import os

def plot_matrices_predictions(
        y_true, 
        y_pred, 
        save_path=None,
        show_plot = False
    ):
    
    fig = plt.figure(figsize=(12, 8))
    param_names = ['M11', 'M12', 'M21', 'M22', 'c1', 'c2', 'D11', 'D22', 'G1', 'G2']
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.plot(y_true[:, i], label='True')
        plt.plot(y_pred[:, i], label='Predicted', linestyle='--')
        plt.title(param_names[i])
        plt.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()
        

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path+'.png')
    
def plot_dynamics_predictions(states,
                              inputs, 
                              true_states_dot, 
                              model,
                              save_path=None, 
                              u=None, 
                              device="cpu", 
                              dt=0.01):
    
    """
    states: (N, 4) -> [q1, q1_dot, q2, q2_dot]
    inputs: what to feed the network (N, ?)
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
        for i, (state, input) in enumerate(zip(states, inputs)):

            #print(f"state shape: {state.shape}")
            # --- 1. Preparazione Input ---
            # Input controllo u corrente (se None, è zero)
            u_curr = np.zeros((2, 1))
            if u is not None:
                u_curr = u[i].reshape(2, 1)

            # Tensore stato per il modello
            xb = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
            
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
    plot_comparison(true_arr, pred_arr, base_states, save_path=save_path)

    return np.array(pred_states_dot)



def plot_comparison(true_derivs, pred_derivs, states, save_path=None):
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

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path+'.png')

def analyze_pendulum_dynamics(model, 
                              X_test, 
                              y_test,
                              X_scale, # Array delle scale input
                              y_scale, # Array delle scale output
                              dt, 
                              device='cpu'):
    
    if hasattr(model, 'f'):
        core_net = model.f
    else:
        core_net = model

    core_net.eval()
    core_net.to(device)
    
    # --- PREPARAZIONE SCALE TENSORS ---
    # X_scale dal main ha 5 elementi (c'è il tempo), ma la rete ne vuole 4.
    # Prendiamo solo i primi 4 elementi (q1, q1_dot, q2, q2_dot)
    if len(X_scale) > 4:
        X_scale_4 = X_scale[:4]
    else:
        X_scale_4 = X_scale

    t_X_scale = torch.tensor(X_scale_4, dtype=torch.float32).to(device)
    t_y_scale = torch.tensor(y_scale, dtype=torch.float32).to(device)

    # --- 1. PREPARAZIONE DATI ---
    traj_idx = 0 
    
    # GT States (FISICI)
    gt_states_full = X_test[traj_idx] 
    gt_states = gt_states_full[:, :4] 
    
    # GT Derivs (FISICI)
    gt_derivs = y_test[traj_idx]

    time_steps = gt_states.shape[0]
    t_axis = np.arange(time_steps) * dt
    
    # --- 2. ROLLOUT ---
    pred_states = np.zeros_like(gt_states)
    
    # Stato iniziale (FISICO)
    current_state = torch.tensor(gt_states[0], dtype=torch.float32).to(device)
    pred_states[0] = current_state.cpu().detach().numpy()
    
    print(f"Eseguendo rollout su {time_steps} steps con dt={dt:.4f}...")
    
    with torch.no_grad():
        for i in range(1, time_steps):
            
            def get_derivs(state_tensor_phys):
                # state_tensor_phys contiene valori fisici (es. 3.14)
                
                if state_tensor_phys.dim() == 1: 
                    state_tensor_phys = state_tensor_phys.unsqueeze(0)
                
                input_norm = state_tensor_phys / t_X_scale
                
                # La rete predice derivate normalizzate
                derivs_norm = core_net(input_norm).squeeze()
                
                derivs_phys = derivs_norm * t_y_scale
                
                return derivs_phys

            # Step RK4 (Tutto avviene nel dominio FISICO grazie a get_derivs che gestisce la scala)
            k1 = get_derivs(current_state)
            k2 = get_derivs(current_state + 0.5 * dt * k1)
            k3 = get_derivs(current_state + 0.5 * dt * k2)
            k4 = get_derivs(current_state + dt * k3)
            
            current_state = current_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            pred_states[i] = current_state.cpu().numpy()

    # --- 3. PLOTTING ---
    plt.figure(figsize=(18, 12))
    
    # A & B (Posizioni e Velocità) rimangono uguali...
    plt.subplot(2, 3, 1)
    plt.plot(t_axis, gt_states[:, 0], 'k--', label='GT $q_1$')
    plt.plot(t_axis, pred_states[:, 0], 'r', label='Pred $q_1$')
    plt.plot(t_axis, gt_states[:, 2], 'b--', label='GT $q_2$')
    plt.plot(t_axis, pred_states[:, 2], 'g', label='Pred $q_2$')
    plt.title('Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(t_axis, gt_states[:, 1], 'k--', label='GT $\dot{q}_1$')
    plt.plot(t_axis, pred_states[:, 1], 'r', label='Pred $\dot{q}_1$')
    plt.plot(t_axis, gt_states[:, 3], 'b--', label='GT $\dot{q}_2$')
    plt.plot(t_axis, pred_states[:, 3], 'g', label='Pred $\dot{q}_2$')
    plt.title('Velocities')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # C. Accelerazione Istantanea
    
    inputs_phys = torch.tensor(gt_states, dtype=torch.float32).to(device)
    inputs_norm = inputs_phys / t_X_scale # <--- Normalizza Input
    
    with torch.no_grad():
        all_derivs_pred_norm = core_net(inputs_norm).cpu().numpy()
        
    all_derivs_pred = all_derivs_pred_norm * np.array(y_scale) # <--- Denormalizza Output

    acc_pred = all_derivs_pred[:, 2:] 
    acc_true = gt_derivs[:, 2:]

    plt.subplot(2, 3, 3)
    plt.scatter(acc_true[:, 0], acc_pred[:, 0], s=5, alpha=0.5, c='red')
    plt.scatter(acc_true[:, 1], acc_pred[:, 1], s=5, alpha=0.5, c='blue')
    min_val = min(acc_true.min(), acc_pred.min())
    max_val = max(acc_true.max(), acc_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.title('Instant Acceleration')
    plt.grid(True, alpha=0.3)

    # ... Resto dei plot (D, E, F) uguale a prima ...
    plt.subplot(2, 3, 4)
    plt.plot(gt_states[:, 0], gt_states[:, 1], 'k--')
    plt.plot(pred_states[:, 0], pred_states[:, 1], 'r')
    plt.title('Phase Link 1')
    
    plt.subplot(2, 3, 5)
    plt.plot(gt_states[:, 2], gt_states[:, 3], 'k--')
    plt.plot(pred_states[:, 2], pred_states[:, 3], 'g')
    plt.title('Phase Link 2')

    plt.subplot(2, 3, 6)
    error = np.abs(gt_states - pred_states)
    plt.plot(t_axis, np.mean(error, axis=1), 'k')
    plt.title('Rollout MAE')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_evolutions_comparison(true_states,
                               predicted_states,
                               dt=0.01,
                               save_path = None,
                               show_plot=False):
    
    """
    Confronta le traiettorie vere e predette (posizioni e velocità).
    true_states: (N, 4) -> [q1, q1_dot, q2, q2_dot]
    predicted_states: (N, 4) -> [q1, q1_dot, q2, q2_dot]
    """
    
    steps = len(true_states)
    time_array = np.arange(0, steps * dt, dt)
    
    # Crea una griglia 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Predicted vs True Trajectories', fontsize=16)
    
    # --- Grafico 1: Posizione q1 (Indice 0) ---
    axes[0, 0].plot(time_array, true_states[:, 0], label='True q1', color='orange', linestyle='--')
    axes[0, 0].plot(time_array, predicted_states[:, 0], label='Predicted q1', color='blue')
    axes[0, 0].set_ylabel('Angle q1 (rad)')
    axes[0, 0].set_title(r'Posizione $q_1$')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- Grafico 2: Velocità q1_dot (Indice 1) ---
    axes[0, 1].plot(time_array, true_states[:, 1], label='True q1_dot', color='purple', linestyle='--')
    axes[0, 1].plot(time_array, predicted_states[:, 1], label='Predicted q1_dot', color='cyan')
    axes[0, 1].set_ylabel('Velocity q1_dot (rad/s)')
    axes[0, 1].set_title(r'Velocità $\dot{q}_1$')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- Grafico 3: Posizione q2 (Indice 2) ---
    axes[1, 0].plot(time_array, true_states[:, 2], label='True q2', color='red', linestyle='--')
    axes[1, 0].plot(time_array, predicted_states[:, 2], label='Predicted q2', color='green')
    axes[1, 0].set_ylabel('Angle q2 (rad)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title(r'Posizione $q_2$')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # --- Grafico 4: Velocità q2_dot (Indice 3) ---
    axes[1, 1].plot(time_array, true_states[:, 3], label='True q2_dot', color='magenta', linestyle='--')
    axes[1, 1].plot(time_array, predicted_states[:, 3], label='Predicted q2_dot', color='lime')
    axes[1, 1].set_ylabel('Velocity q2_dot (rad/s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title(r'Velocità $\dot{q}_2$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show_plot:
        plt.show()
        plt.close()