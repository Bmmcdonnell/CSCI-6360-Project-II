import os
import contextlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from get_qof import get_qof
from Project_2_2L import NoHiddenLayerNN
from Project_2_3L import OneHiddenLayerNNLeakyRelu

def get_cv_qof(X, y, nn_type='2L', n_splits=5):
    """
    Performs k-fold cross-validation for either a 2-Layer or 3-Layer Neural Network model 
    and calculates Quality of Fit (QoF) metrics for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_stats = [[] for _ in range(15)]
    
    for train_idx, val_idx in kf.split(X):
        # --- Data Splitting ---
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # --- Data Scaling ---
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val) # Prevent data leakage

        # --- Tensor Conversion ---
        X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr.to_numpy(), dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
        
        # ==========================================
        # --- Model Training and Prediction ---
        # ==========================================
        input_features = X_tr.shape[1]
        output_classes = y_tr.shape[1] if len(y_tr.shape) > 1 else 1
        
        # Context manager to suppress epoch print spam
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            if nn_type == '2L':
                model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes)
                model.trainNN(X_tr_tensor, y_tr_tensor)
                predictions, _ = model.testNN(X_val_tensor, y_val_tensor)
            elif nn_type == '3L':
                model = OneHiddenLayerNNLeakyRelu(input_size=input_features, hidden_size=100, output_size=output_classes)
                model.trainLeakyRelu(X_tr_tensor, y_tr_tensor)
                predictions, _ = model.testLeakyRelu(X_val_tensor, y_val_tensor)
            else:
                raise ValueError(f"nn_type must be '2L' or '3L'. Received {nn_type}")
        
        # ==========================================
        # --- Model Evaluation ---
        # ==========================================
        y_val_numpy = y_val.to_numpy()
        y_pred_numpy = predictions.detach().cpu().numpy()
        k = X.shape[1] 
        
        temp_qof = get_qof(y_val_numpy, y_pred_numpy, k)
        
        for i in range(15):
            cv_stats[i].append(temp_qof[i])

    return cv_stats