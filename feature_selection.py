import pandas as pd
import numpy as np
from save_plots import save_rsq_plot, save_aic_bic_plot
from feature_selection_methods import forward_select_all, backward_eliminate_all, stepwise_selection

def feature_selection(key, nn_type, X, y, data_name, folder_name):
    """
    Performs feature selection (Forward, Backward, or Stepwise) for a specified PyTorch Neural Network model.
    
    Evaluates either a 2-Layer or 3-Layer NN model. Calculates and tracks Quality of Fit (QoF) 
    metrics at each stage of the selection process. Automatically generates and saves 
    high-resolution plots for R^2 vs. number of features, and AIC/BIC vs. number of features.

    Args:
        key (str): The feature selection method to use ('Forward', 'Backward', or 'Stepwise').
        nn_type (str): The network type to evaluate ('2L' or '3L').
        X (pd.DataFrame): The standard input feature matrix.
        y (pd.Series or pd.DataFrame): The target response vector.
        data_name (str): The name of the dataset (used for plot titles and logging).
        folder_name (str): The directory path where output plots will be saved.
        
    Returns:
        tuple: (features, qof_list, cv_stats_list) containing the ordered list of features
               and their associated metrics at each step.
    """
    
    # Configure model naming conventions for plots and logs
    if nn_type == '2L':
        model_name_t = '2 Layer Neural Network'
        model_name_s = '2L'
    elif nn_type == '3L':
        model_name_t = '3 Layer Neural Network'
        model_name_s = '3L'
    else:
        raise ValueError(f"nn_type must be '2L' or '3L'. Received {nn_type}")
        
    method_title = f"{key} Selection"
    print(f"Running {key} Selection for {model_name_t}...")
    
    # --- 1. Run Feature Selection & Print Feature Order ---
    if key == 'Forward':
        features, qof_list, cv_stats_list = forward_select_all(
            X, y, nn_type=nn_type, start_cols=None, metric=0
        )
        print(f"Python {model_name_s} Forward Selection Order: {features}")
        
    elif key == 'Backward':
        features, qof_list, cv_stats_list = backward_eliminate_all(
            X, y, nn_type=nn_type, start_cols=None, metric=0
        )
        print(f"Python {model_name_s} Backward Elimination Reversed Order: {features}")
        
    elif key == 'Stepwise':
        features, qof_list, cv_stats_list = stepwise_selection(
            X, y, nn_type=nn_type, start_cols=None, metric=1
        )
        print(f"Python {model_name_s} Stepwise Selection Order: {features}")
        
    else:
        raise ValueError(f"key must be 'Forward', 'Backward', or 'Stepwise'. Received {key}")

    # --- 2. Extract Tracking Metrics ---
    # Initialize lists for tracking
    x_axis = list(range(len(features)))
    r_sq = []
    adj_r_sq = []
    smape = []
    r_sq_cv = []
    aic = []
    bic = []
    
    for i in x_axis:
        # Metrics mappings based on `get_qof.py` index locations:
        # 0=R^2, 1=Adj R^2, 8=sMAPE, 13=AIC, 14=BIC
        r_sq.append(100 * qof_list[i][0])
        adj_r_sq.append(100 * qof_list[i][1])
        smape.append(qof_list[i][8])
        
        # Cross-validation stats matrix has metric lists as its rows
        cv_metric = np.mean(cv_stats_list[i][0]) if cv_stats_list[i] is not None else 0
        r_sq_cv.append(100 * cv_metric)
        
        aic.append(qof_list[i][13])
        bic.append(qof_list[i][14])

    # --- 3. Generate and Save Plots ---
    save_rsq_plot(
        key=key, 
        x=x_axis, 
        r_sq=r_sq, 
        adj_r_sq=adj_r_sq, 
        smape=smape, 
        r_sq_cv=r_sq_cv, 
        method=method_title, 
        data_name=data_name, 
        folder_name=folder_name, 
        model_name_t=model_name_t, 
        model_name_s=model_name_s
    )
    
    save_aic_bic_plot(
        key=key, 
        x=x_axis, 
        aic=aic, 
        bic=bic, 
        method=method_title, 
        data_name=data_name, 
        folder_name=folder_name, 
        model_name_t=model_name_t, 
        model_name_s=model_name_s
    )
    
    print(f"Finished {model_name_t} {key} Selection. Plots saved to {folder_name}/\n")
    
    return features, qof_list, cv_stats_list

def p2_auto_mpg_feature_selection():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_auto_mpg_with_intercept.csv")
    ox = oxy.drop('mpg', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['mpg']]

    features_2L_fwd, _, _ = feature_selection('Forward', '2L', X, y, 'Auto MPG', 'Auto_MPG_P2_Feature_Selection')
    features_2L_bwd, _, _ = feature_selection('Backward', '2L', X, y, 'Auto MPG', 'Auto_MPG_P2_Feature_Selection')
    features_2L_stp, _, _ = feature_selection('Stepwise', '2L', X, y, 'Auto MPG', 'Auto_MPG_P2_Feature_Selection')

    features_3L_fwd, _, _ = feature_selection('Forward', '3L', X, y, 'Auto MPG', 'Auto_MPG_P2_Feature_Selection')
    features_3L_bwd, _, _ = feature_selection('Backward', '3L', X, y, 'Auto MPG', 'Auto_MPG_P2_Feature_Selection')
    features_3L_stp, _, _ = feature_selection('Stepwise', '3L', X, y, 'Auto MPG', 'Auto_MPG_P2_Feature_Selection')

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------2 Layer Neural Network--------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print(f"PyTorch Forward Selection Order: {features_2L_fwd}")
    print(f"PyTorch Backward Elimination Reversed Order: {features_2L_bwd}")
    print(f"PyTorch Stepwise Selection Order: {features_2L_stp}")

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------3 Layer Neural Network--------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print(f"PyTorch Forward Selection Order: {features_3L_fwd}")
    print(f"PyTorch Backward Elimination Reversed Order: {features_3L_bwd}")
    print(f"PyTorch Stepwise Selection Order: {features_3L_stp}")





def p2_housing_feature_selection():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_housing_with_intercept.csv")
    ox = oxy.drop('median_house_value', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['median_house_value']]

    features_2L_fwd, _, _ = feature_selection('Forward', '2L', X, y, 'California House Prices', 'Housing_P2_Feature_Selection')
    features_2L_bwd, _, _ = feature_selection('Backward', '2L', X, y, 'California House Prices', 'Housing_P2_Feature_Selection')
    features_2L_stp, _, _ = feature_selection('Stepwise', '2L', X, y, 'California House Prices', 'Housing_P2_Feature_Selection')

    features_3L_fwd, _, _ = feature_selection('Forward', '3L', X, y, 'California House Prices', 'Housing_P2_Feature_Selection')
    features_3L_bwd, _, _ = feature_selection('Backward', '3L', X, y, 'California House Prices', 'Housing_P2_Feature_Selection')
    features_3L_stp, _, _ = feature_selection('Stepwise', '3L', X, y, 'California House Prices', 'Housing_P2_Feature_Selection')

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------2 Layer Neural Network--------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print(f"PyTorch Forward Selection Order: {features_2L_fwd}")
    print(f"PyTorch Backward Elimination Reversed Order: {features_2L_bwd}")
    print(f"PyTorch Stepwise Selection Order: {features_2L_stp}")

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------3 Layer Neural Network--------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print(f"PyTorch Forward Selection Order: {features_3L_fwd}")
    print(f"PyTorch Backward Elimination Reversed Order: {features_3L_bwd}")
    print(f"PyTorch Stepwise Selection Order: {features_3L_stp}")






def p2_insurance_feature_selection():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_insurance_with_intercept.csv")
    ox = oxy.drop('charges', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['charges']]

    features_2L_fwd, _, _ = feature_selection('Forward', '2L', X, y, 'Insurance Charges', 'Insurance_P2_Feature_Selection')
    features_2L_bwd, _, _ = feature_selection('Backward', '2L', X, y, 'Insurance Charges', 'Insurance_P2_Feature_Selection')
    features_2L_stp, _, _ = feature_selection('Stepwise', '2L', X, y, 'Insurance Charges', 'Insurance_P2_Feature_Selection')

    features_3L_fwd, _, _ = feature_selection('Forward', '3L', X, y, 'Insurance Charges', 'Insurance_P2_Feature_Selection')
    features_3L_bwd, _, _ = feature_selection('Backward', '3L', X, y, 'Insurance Charges', 'Insurance_P2_Feature_Selection')
    features_3L_stp, _, _ = feature_selection('Stepwise', '3L', X, y, 'Insurance Charges', 'Insurance_P2_Feature_Selection')

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------2 Layer Neural Network--------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print(f"PyTorch Forward Selection Order: {features_2L_fwd}")
    print(f"PyTorch Backward Elimination Reversed Order: {features_2L_bwd}")
    print(f"PyTorch Stepwise Selection Order: {features_2L_stp}")

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------3 Layer Neural Network--------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print(f"PyTorch Forward Selection Order: {features_3L_fwd}")
    print(f"PyTorch Backward Elimination Reversed Order: {features_3L_bwd}")
    print(f"PyTorch Stepwise Selection Order: {features_3L_stp}")



# p2_auto_mpg_feature_selection()
# p2_housing_feature_selection()
p2_insurance_feature_selection()