# ----------------- 1 -----------------
# increasing learning rates slower

# # Dynamic layer for pendulum dynamics
#     dPend = DynLayer(
#         b1_ref=0.1,
#         fixed_b1=0.3, 
#         b2_ref=0.1,
#         fixed_b2 = 0.3,
#         l1_ref=1.0, 
#         fixed_l1 = 1.2,
#         l2_ref=10, 
#         fixed_l2=None,
#         m1_ref=1.0, 
#         fixed_m1 = 2.2,
#         m2_ref=1.0,
#         fixed_m2=1.8 
#         )

#     # Integration layer
#     integration = RK4Model(dPend)

#     # --- Loss function ---

#     #loss_fn = physics_loss 
#     loss_fn = nn.MSELoss()

#     # --- Training ---

#     curriculum_steps = [4, 8, 16, 32] # steps of horizon (dynamic during training)
#     epochs_per_stage = [10, 10, 10, 10] 
#     lrs = [1e-2, 2e-2, 4e-2, 8e-2]       

#     early_stopper = EarlyStopping(patience=10)

python -m branch1.main \
    --train_path data/training/train_lowfrict.csv \
    --val_path data/training/val_lowfrict.csv \
    

# ====================================
# ----------------- 2 -----------------
# keeping learning rates constant

# # Dynamic layer for pendulum dynamics
#     dPend = DynLayer(
#         b1_ref=0.1,
#         fixed_b1=0.3, 
#         b2_ref=0.1,
#         fixed_b2 = 0.3,
#         l1_ref=1.0, 
#         fixed_l1 = 1.2,
#         l2_ref=5, 
#         fixed_l2=None,
#         m1_ref=1.0, 
#         fixed_m1 = 2.2,
#         m2_ref=1.0,
#         fixed_m2=1.8 
#         )

#     # Integration layer
#     integration = RK4Model(dPend)

#     # --- Loss function ---

#     #loss_fn = physics_loss 
#     loss_fn = nn.MSELoss()

#     # --- Training ---

#     curriculum_steps = [4, 8, 16, 32] # steps of horizon (dynamic during training)
#     epochs_per_stage = [10, 10, 10, 10] 
#     lrs = [1e-2, 1e-2, 1e-2, 1e-2]       

#     early_stopper = EarlyStopping(patience=10)