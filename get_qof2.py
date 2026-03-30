import os
import contextlib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from get_qof import get_qof
from get_cv_qof import get_cv_qof
from Project_2_2L import NoHiddenLayerNN
from Project_2_3L import OneHiddenLayerNNLeakyRelu

def get_qof2(X, y, nn_type='2L', cv=False):
    """
    Fits a 2L or 3L Neural Network model and calculates Quality of Fit (QoF) metrics.
    Optionally performs k-fold cross-validation.
    """
    if cv:
        cv_stats = get_cv_qof(X, y, nn_type=nn_type)
    else:
        cv_stats = None
    
    # --- Data Scaling and Tensor Conversion ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
    
    # --- Model Training and Prediction ---
    input_features = X.shape[1]
    output_classes = y.shape[1] if len(y.shape) > 1 else 1
    
    # Context manager to suppress epoch print spam
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        if nn_type == '2L':
            model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes)
            model.trainNN(X_tensor, y_tensor)
            predictions, _ = model.testNN(X_tensor, y_tensor)
        elif nn_type == '3L':
            model = OneHiddenLayerNNLeakyRelu(input_size=input_features, hidden_size=100, output_size=output_classes)
            model.trainLeakyRelu(X_tensor, y_tensor)
            predictions, _ = model.testLeakyRelu(X_tensor, y_tensor)
        else:
            raise ValueError(f"nn_type must be '2L' or '3L'. Received {nn_type}")
    
    # --- Metrics Calculation & Return ---
    y_numpy = y.to_numpy()
    y_pred_numpy = predictions.detach().cpu().numpy()
    k = X.shape[1]  
    
    qof = get_qof(y_numpy, y_pred_numpy, k)

    return (qof, cv_stats)