import numpy as np

def is_oos_comparison(is_qof, oos_qof, data_name, model_name):
    """
    Generates a LaTeX formatted table comparing In-Sample and Out-of-Sample 
    (80-20 Split) Quality of Fit (QoF) metrics for a single model.

    Args:
        is_qof (list): A list of 15 QoF metrics evaluated on the In-Sample dataset.
        oos_qof (list): A list of 15 QoF metrics evaluated on the Out-of-Sample dataset.
        data_name (str): The name of the dataset.
        model_name (str): The name of the regression model.

    Returns:
        None: Outputs the LaTeX table string directly to standard output.
    """
    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    print("\\begin{table}[H]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} {model_name}}}")
    print(f"\\label{{tab:Statsmodels - {data_name} {model_name}}}")
    print("\\begin{tabular}{|c|c|c|}\\hline")
    print("Metric & In-Sample & 80-20 Split \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    print(f"rSq & {is_qof[0]:.4f} & {oos_qof[0]:.4f} \\\\ \\hline")
    print(f"rSqBar & {is_qof[1]:.4f} & {oos_qof[1]:.4f} \\\\ \\hline")
    print(f"sst & {is_qof[2]:.4f} & {oos_qof[2]:.4f} \\\\ \\hline")
    print(f"sse & {is_qof[3]:.4f} & {oos_qof[3]:.4f} \\\\ \\hline")
    print(f"sde & {is_qof[4]:.4f} & {oos_qof[4]:.4f} \\\\ \\hline")
    print(f"mse0 & {is_qof[5]:.4f} & {oos_qof[5]:.4f} \\\\ \\hline")
    print(f"rmse & {is_qof[6]:.4f} & {oos_qof[6]:.4f} \\\\ \\hline")
    print(f"mae & {is_qof[7]:.4f} & {oos_qof[7]:.4f} \\\\ \\hline")
    print(f"smape & {is_qof[8]:.4f} & {oos_qof[8]:.4f} \\\\ \\hline")
    print(f"m & {is_qof[9]:.4f} & {oos_qof[9]:.4f} \\\\ \\hline")
    print(f"dfr & {is_qof[10]:.4f} & {oos_qof[10]:.4f} \\\\ \\hline")
    print(f"df & {is_qof[11]:.4f} & {oos_qof[11]:.4f} \\\\ \\hline")
    print(f"fStat & {is_qof[12]:.4f} & {oos_qof[12]:.4f} \\\\ \\hline")
    print(f"aic & {is_qof[13]:.4f} & {oos_qof[13]:.4f} \\\\ \\hline")
    print(f"bic & {is_qof[14]:.4f} & {oos_qof[14]:.4f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")


def model_comparison(reg_qof, ridge_qof, lasso_qof, sqrt_qof, log1p_qof, boxcox_qof, order2reg_qof, data_name, validate):
    """
    Generates a LaTeX formatted table comparing the Quality of Fit (QoF) metrics 
    across all evaluated regression models for a specific data split.

    Args:
        reg_qof (list): QoF metrics for Linear Regression.
        ridge_qof (list): QoF metrics for Ridge Regression.
        lasso_qof (list): QoF metrics for Lasso Regression.
        sqrt_qof (list): QoF metrics for Square Root Transformed Regression.
        log1p_qof (list): QoF metrics for Log1p Transformed Regression.
        boxcox_qof (list): QoF metrics for Box-Cox Transformed Regression.
        order2reg_qof (list): QoF metrics for Order-2 Polynomial Regression.
        data_name (str): The name of the dataset.
        validate (bool): Flag indicating if the metrics are Out-of-Sample (True) 
                         or In-Sample (False).

    Returns:
        None: Outputs the LaTeX table string directly to standard output.
    """
    method = "Out-of-Sample" if validate else "In-Sample"

    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    print("\\begin{table}[H]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} {method} QoF Comparison}}")
    print(f"\\label{{tab:Statsmodels - {data_name} {method} QoF Comparison}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\\hline")
    print("Metric & Regression & Ridge & Lasso & Sqrt & Log1p & Box-Cox & Order 2 Polynomial \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    print(f"rSq & {reg_qof[0]:.4f} & {ridge_qof[0]:.4f} & {lasso_qof[0]:.4f} & {sqrt_qof[0]:.4f} & {log1p_qof[0]:.4f} & {boxcox_qof[0]:.4f} & {order2reg_qof[0]:.4f} \\\\ \\hline")
    print(f"rSqBar & {reg_qof[1]:.4f} & {ridge_qof[1]:.4f} & {lasso_qof[1]:.4f} & {sqrt_qof[1]:.4f} & {log1p_qof[1]:.4f} & {boxcox_qof[1]:.4f} & {order2reg_qof[1]:.4f} \\\\ \\hline")
    print(f"sst & {reg_qof[2]:.4f} & {ridge_qof[2]:.4f} & {lasso_qof[2]:.4f} & {sqrt_qof[2]:.4f} & {log1p_qof[2]:.4f} & {boxcox_qof[2]:.4f} & {order2reg_qof[2]:.4f} \\\\ \\hline")
    print(f"sse & {reg_qof[3]:.4f} & {ridge_qof[3]:.4f} & {lasso_qof[3]:.4f} & {sqrt_qof[3]:.4f} & {log1p_qof[3]:.4f} & {boxcox_qof[3]:.4f} & {order2reg_qof[3]:.4f} \\\\ \\hline")
    print(f"sde & {reg_qof[4]:.4f} & {ridge_qof[4]:.4f} & {lasso_qof[4]:.4f} & {sqrt_qof[4]:.4f} & {log1p_qof[4]:.4f} & {boxcox_qof[4]:.4f} & {order2reg_qof[4]:.4f} \\\\ \\hline")
    print(f"mse0 & {reg_qof[5]:.4f} & {ridge_qof[5]:.4f} & {lasso_qof[5]:.4f} & {sqrt_qof[5]:.4f} & {log1p_qof[5]:.4f} & {boxcox_qof[5]:.4f} & {order2reg_qof[5]:.4f} \\\\ \\hline")
    print(f"rmse & {reg_qof[6]:.4f} & {ridge_qof[6]:.4f} & {lasso_qof[6]:.4f} & {sqrt_qof[6]:.4f} & {log1p_qof[6]:.4f} & {boxcox_qof[6]:.4f} & {order2reg_qof[6]:.4f} \\\\ \\hline")
    print(f"mae & {reg_qof[7]:.4f} & {ridge_qof[7]:.4f} & {lasso_qof[7]:.4f} & {sqrt_qof[7]:.4f} & {log1p_qof[7]:.4f} & {boxcox_qof[7]:.4f} & {order2reg_qof[7]:.4f} \\\\ \\hline")
    print(f"smape & {reg_qof[8]:.4f} & {ridge_qof[8]:.4f} & {lasso_qof[8]:.4f} & {sqrt_qof[8]:.4f} & {log1p_qof[8]:.4f} & {boxcox_qof[8]:.4f} & {order2reg_qof[8]:.4f} \\\\ \\hline")
    print(f"m & {reg_qof[9]:.4f} & {ridge_qof[9]:.4f} & {lasso_qof[9]:.4f} & {sqrt_qof[9]:.4f} & {log1p_qof[9]:.4f} & {boxcox_qof[9]:.4f} & {order2reg_qof[9]:.4f} \\\\ \\hline")
    print(f"dfr & {reg_qof[10]:.4f} & {ridge_qof[10]:.4f} & {lasso_qof[10]:.4f} & {sqrt_qof[10]:.4f} & {log1p_qof[10]:.4f} & {boxcox_qof[10]:.4f} & {order2reg_qof[10]:.4f} \\\\ \\hline")
    print(f"df & {reg_qof[11]:.4f} & {ridge_qof[11]:.4f} & {lasso_qof[11]:.4f} & {sqrt_qof[11]:.4f} & {log1p_qof[11]:.4f} & {boxcox_qof[11]:.4f} & {order2reg_qof[11]:.4f} \\\\ \\hline")
    print(f"fStat & {reg_qof[12]:.4f} & {ridge_qof[12]:.4f} & {lasso_qof[12]:.4f} & {sqrt_qof[12]:.4f} & {log1p_qof[12]:.4f} & {boxcox_qof[12]:.4f} & {order2reg_qof[12]:.4f} \\\\ \\hline")
    print(f"aic & {reg_qof[13]:.4f} & {ridge_qof[13]:.4f} & {lasso_qof[13]:.4f} & {sqrt_qof[13]:.4f} & {log1p_qof[13]:.4f} & {boxcox_qof[13]:.4f} & {order2reg_qof[13]:.4f} \\\\ \\hline")
    print(f"bic & {reg_qof[14]:.4f} & {ridge_qof[14]:.4f} & {lasso_qof[14]:.4f} & {sqrt_qof[14]:.4f} & {log1p_qof[14]:.4f} & {boxcox_qof[14]:.4f} & {order2reg_qof[14]:.4f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")


def cv_table(cv_stats, data_name, model_name):
    """
    Generates a LaTeX formatted table summarizing Cross-Validation (CV) statistics.
    
    Displays the minimum, maximum, mean, and standard deviation for each of the 
    15 Quality of Fit (QoF) metrics across all evaluated CV folds.

    Args:
        cv_stats (list): A list of 15 lists containing metric values for each CV fold.
        data_name (str): The name of the dataset.
        model_name (str): The name of the regression model.

    Returns:
        None: Outputs the LaTeX table string directly to standard output.
    """
    # ==========================================
    # --- Table Header Configuration ---
    # ==========================================
    print("\\begin{table}[H]")
    print("\\centering")
    print(f"\\caption{{Statsmodels - {data_name} {model_name} CV}}")
    print(f"\\label{{tab:Statsmodels - {data_name} {model_name} CV}}")
    print("\\begin{tabular}{|c|c|c|c|c|c|}\\hline")
    print("Metric & num folds & min & max & mean & stdev \\\\ \\hline \\hline")
    
    # ==========================================
    # --- Table Data Rows ---
    # ==========================================
    print(f"rSq & {len(cv_stats[0])} & {min(cv_stats[0]):.4f} & {max(cv_stats[0]):.4f} & {np.mean(cv_stats[0]):.4f} & {np.std(cv_stats[0]):.4f} \\\\ \\hline")
    print(f"rSqBar & {len(cv_stats[1])} & {min(cv_stats[1]):.4f} & {max(cv_stats[1]):.4f} & {np.mean(cv_stats[1]):.4f} & {np.std(cv_stats[1]):.4f} \\\\ \\hline")
    print(f"sst & {len(cv_stats[2])} & {min(cv_stats[2]):.4f} & {max(cv_stats[2]):.4f} & {np.mean(cv_stats[2]):.4f} & {np.std(cv_stats[2]):.4f} \\\\ \\hline")
    print(f"sse & {len(cv_stats[3])} & {min(cv_stats[3]):.4f} & {max(cv_stats[3]):.4f} & {np.mean(cv_stats[3]):.4f} & {np.std(cv_stats[3]):.4f} \\\\ \\hline")
    print(f"sde & {len(cv_stats[4])} & {min(cv_stats[4]):.4f} & {max(cv_stats[4]):.4f} & {np.mean(cv_stats[4]):.4f} & {np.std(cv_stats[4]):.4f} \\\\ \\hline")
    print(f"mse0 & {len(cv_stats[5])} & {min(cv_stats[5]):.4f} & {max(cv_stats[5]):.4f} & {np.mean(cv_stats[5]):.4f} & {np.std(cv_stats[5]):.4f} \\\\ \\hline")
    print(f"rmse & {len(cv_stats[6])} & {min(cv_stats[6]):.4f} & {max(cv_stats[6]):.4f} & {np.mean(cv_stats[6]):.4f} & {np.std(cv_stats[6]):.4f} \\\\ \\hline")
    print(f"mae & {len(cv_stats[7])} & {min(cv_stats[7]):.4f} & {max(cv_stats[7]):.4f} & {np.mean(cv_stats[7]):.4f} & {np.std(cv_stats[7]):.4f} \\\\ \\hline")
    print(f"smape & {len(cv_stats[8])} & {min(cv_stats[8]):.4f} & {max(cv_stats[8]):.4f} & {np.mean(cv_stats[8]):.4f} & {np.std(cv_stats[8]):.4f} \\\\ \\hline")
    print(f"m & {len(cv_stats[9])} & {min(cv_stats[9]):.4f} & {max(cv_stats[9]):.4f} & {np.mean(cv_stats[9]):.4f} & {np.std(cv_stats[9]):.4f} \\\\ \\hline")
    print(f"dfr & {len(cv_stats[10])} & {min(cv_stats[10]):.4f} & {max(cv_stats[10]):.4f} & {np.mean(cv_stats[10]):.4f} & {np.std(cv_stats[10]):.4f} \\\\ \\hline")
    print(f"df & {len(cv_stats[11])} & {min(cv_stats[11]):.4f} & {max(cv_stats[11]):.4f} & {np.mean(cv_stats[11]):.4f} & {np.std(cv_stats[11]):.4f} \\\\ \\hline")
    print(f"fStat & {len(cv_stats[12])} & {min(cv_stats[12]):.4f} & {max(cv_stats[12]):.4f} & {np.mean(cv_stats[12]):.4f} & {np.std(cv_stats[12]):.4f} \\\\ \\hline")
    print(f"aic & {len(cv_stats[13])} & {min(cv_stats[13]):.4f} & {max(cv_stats[13]):.4f} & {np.mean(cv_stats[13]):.4f} & {np.std(cv_stats[13]):.4f} \\\\ \\hline")
    print(f"bic & {len(cv_stats[14])} & {min(cv_stats[14]):.4f} & {max(cv_stats[14]):.4f} & {np.mean(cv_stats[14]):.4f} & {np.std(cv_stats[14]):.4f} \\\\ \\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")