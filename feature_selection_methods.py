from get_qof2 import get_qof2

def select_single_feature(X, y, in_cols, out_cols, nn_type='2L', metric=0):
    if metric in [0, 1, 12]:
        best_metric = -float('inf')
    elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
        best_metric = float('inf')
    else:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    feature_to_add = out_cols[0]
    
    for col in out_cols:
        new_cols = in_cols + [col]
        temp_X = X[new_cols].copy()

        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=False)
        cur_metric = temp_qof[metric]

        if metric in [0, 1, 12]:
            if cur_metric > best_metric:
                best_metric = cur_metric
                feature_to_add = col
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            if cur_metric < best_metric:
                best_metric = cur_metric
                feature_to_add = col

    new_in_cols = in_cols + [feature_to_add]
    new_out_cols = [col for col in out_cols if col != feature_to_add]
    temp_X = X[new_in_cols].copy()

    (best_qof, best_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
    return (new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats)


def forward_select_all(X, y, nn_type='2L', start_cols=None, metric=0):
    if start_cols is None:
        if 'intercept' in X.columns:
            start_cols_copy = ['intercept']
            in_cols = ['intercept']
            for_sel_features = ['intercept']
        else:
            start_cols_copy = []
            in_cols = []
            for_sel_features = []
    else:
        start_cols_copy = start_cols.copy()
        in_cols = start_cols.copy()
        for_sel_features = [start_cols]

    if len(in_cols) == 0:
        qof_list = []
        cv_stats_list = []
        if 'intercept' not in X.columns:
            temp_X = X.copy()
            temp_X['intercept'] = 1
            X_int = temp_X[['intercept']].copy()
            (int_qof, int_cv_stats) = get_qof2(X_int, y, nn_type=nn_type, cv=True)
            for_sel_features.append('Null')
            qof_list.append(int_qof.copy())
            cv_stats_list.append(int_cv_stats.copy())
    else:
        temp_X = X[in_cols].copy()
        (qof, cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
        qof_list = [qof]
        cv_stats_list = [cv_stats]

    out_cols = [col for col in X.columns if col not in start_cols_copy]

    while True:
        (new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats) = select_single_feature(
            X, y, in_cols, out_cols, nn_type=nn_type, metric=metric
        )
        
        in_cols = new_in_cols.copy()
        out_cols = new_out_cols.copy()
        for_sel_features.append(feature_to_add)
        qof_list.append(best_qof.copy())
        cv_stats_list.append(best_cv_stats.copy())

        if len(out_cols) == 0:
            break

    return (for_sel_features, qof_list, cv_stats_list)


def eliminate_single_feature(X, y, in_cols, nn_type='2L', metric=0):
    if metric in [0, 1, 12]:
        best_metric = -float('inf')
    elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
        best_metric = float('inf')
    else:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    feature_to_remove = in_cols[0]
    
    if 'intercept' in in_cols:
        in_cols_copy = [col for col in in_cols if col != 'intercept']
    else:
        in_cols_copy = in_cols.copy()

    for col in in_cols_copy:
        if 'intercept' in X.columns:
            new_cols = [col2 for col2 in in_cols_copy if col2 != col] + ['intercept']
        else:
            new_cols = [col2 for col2 in in_cols_copy if col2 != col]
        
        temp_X = X[new_cols].copy()
        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=False)

        cur_metric = temp_qof[metric]

        if metric in [0, 1, 12]:
            if cur_metric > best_metric:
                best_metric = cur_metric
                feature_to_remove = col
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            if cur_metric < best_metric:
                best_metric = cur_metric
                feature_to_remove = col

    new_in_cols = [col2 for col2 in in_cols if col2 != feature_to_remove]
    temp_X = X[new_in_cols].copy()

    (best_qof, best_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
    return (new_in_cols, feature_to_remove, best_qof, best_cv_stats)


def backward_eliminate_all(X, y, nn_type='2L', start_cols=None, metric=0):
    if start_cols is None:
        in_cols = X.columns.tolist().copy()
    elif len(start_cols) == 0:
        raise ValueError(f"start_cols must be non-empty")
    else:
        in_cols = start_cols.copy()
    
    temp_X = X[in_cols].copy()
    (qof, cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)

    bac_eli_features = []
    qof_list = [qof]
    cv_stats_list = [cv_stats]

    while True:
        (new_in_cols, feature_to_remove, best_qof, best_cv_stats) = eliminate_single_feature(
            X, y, in_cols, nn_type=nn_type, metric=metric
        )
        
        in_cols = new_in_cols.copy()
        bac_eli_features.append(feature_to_remove)
        qof_list.append(best_qof.copy())
        cv_stats_list.append(best_cv_stats.copy())

        if len(in_cols) == 1:
            break
    
    bac_eli_features.append(in_cols[0])

    if in_cols[0] != 'intercept':
        temp_X = X.copy()
        temp_X['intercept'] = 1
        X_int = temp_X[['intercept']].copy()
        (int_qof, int_cv_stats) = get_qof2(X_int, y, nn_type=nn_type, cv=True)
        bac_eli_features.append('Null')
        qof_list.append(int_qof.copy())
        cv_stats_list.append(int_cv_stats.copy())

    bac_eli_features.reverse()
    qof_list.reverse()
    cv_stats_list.reverse()

    return (bac_eli_features, qof_list, cv_stats_list)


def stepwise_selection(X, y, nn_type='2L', start_cols=None, metric=1):
    if metric not in [0, 1, 3, 4, 5, 6, 7, 8, 12, 13, 14]:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")
    
    qof_dict = {}
    cv_stats_dict = {}

    if start_cols is None:
        if 'intercept' in X.columns:
            start_cols_copy = ['intercept']
            in_cols = ['intercept']
            step_sel_features = ['intercept']

            temp_X = X[in_cols].copy()
            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
    
            qof_dict['intercept'] = temp_qof.copy()
            cv_stats_dict['intercept'] = temp_cv_stats.copy()
            cur_metric = temp_qof[metric]
        else:
            start_cols_copy = []
            in_cols = []
            step_sel_features = []
    else:
        start_cols_copy = start_cols.copy()
        in_cols = start_cols.copy()
        step_sel_features = start_cols.copy()
        
        for i in range(len(in_cols)):
            temp_in_cols = in_cols[:i+1]
            temp_X = X[temp_in_cols].copy()

            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)

            qof_dict[in_cols[i]] = temp_qof.copy()
            cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
        cur_metric = temp_qof[metric]
    
    if len(in_cols) == 0:
        if 'intercept' not in X.columns:
            temp_X = X.copy()
            temp_X['intercept'] = 1
            X_int = temp_X[['intercept']].copy()
            (int_qof, int_cv_stats) = get_qof2(X_int, y, nn_type=nn_type, cv=True)
            step_sel_features.append('Null')
            qof_dict['Null'] = int_qof.copy()
            cv_stats_dict['Null'] = int_cv_stats.copy()
            cur_metric = int_qof[metric]
        elif metric in [0, 1, 12]:
            cur_metric = -float('inf')
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            cur_metric = float('inf')
        else:
            raise ValueError(f"cur_metric was not assigned a value: {cur_metric}")

    out_cols = [col for col in X.columns if col not in start_cols_copy]
    num_cols = X.shape[1]
 
    while True:
        if len(in_cols) <= 1:
            (sel_new_in_cols, sel_new_out_cols, feature_to_add, sel_best_qof, sel_best_cv_stats) = select_single_feature(
                X, y, in_cols, out_cols, nn_type=nn_type, metric=metric
            )

            if metric in [0, 1, 12] and (sel_best_qof[metric] >= cur_metric):
                in_cols = sel_new_in_cols.copy()
                out_cols = sel_new_out_cols.copy()
                step_sel_features.append(feature_to_add)
                qof_dict[feature_to_add] = sel_best_qof.copy()
                cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                cur_metric = sel_best_qof[metric]

            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and (sel_best_qof[metric] <= cur_metric):
                in_cols = sel_new_in_cols.copy()
                out_cols = sel_new_out_cols.copy()
                step_sel_features.append(feature_to_add)
                qof_dict[feature_to_add] = sel_best_qof.copy()
                cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                cur_metric = sel_best_qof[metric]

            else:
                break

        elif len(in_cols) == num_cols:
            (bac_new_in_cols, feature_to_remove, bac_best_qof, bac_best_cv_stats) = eliminate_single_feature(
                X, y, in_cols, nn_type=nn_type, metric=metric
            )

            if metric in [0, 1, 12] and (bac_best_qof[metric] >= cur_metric):
                old_in_cols = in_cols.copy()
                in_cols = bac_new_in_cols.copy()
                temp_in_cols = bac_new_in_cols.copy()
                out_cols = [col for col in X.columns if col not in temp_in_cols]
                step_sel_features_copy = step_sel_features.copy()
                step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                flag = False
                for i in range(len(old_in_cols)):
                    if old_in_cols[i] == feature_to_remove:
                        flag = True
                    if flag and (i < len(in_cols)):
                        temp_in_cols = in_cols[:i]
                        temp_X = X[temp_in_cols].copy()
                        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
                        qof_dict[in_cols[i]] = temp_qof.copy()
                        cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                
                cur_metric = bac_best_qof[metric]

            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and (bac_best_qof[metric] <= cur_metric):
                old_in_cols = in_cols.copy()
                in_cols = bac_new_in_cols.copy()
                temp_in_cols = bac_new_in_cols.copy()
                out_cols = [col for col in X.columns if col not in temp_in_cols]
                step_sel_features_copy = step_sel_features.copy()
                step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                flag = False
                for i in range(len(old_in_cols)):
                    if old_in_cols[i] == feature_to_remove:
                        flag = True
                    if flag and (i < len(in_cols)):
                        temp_in_cols = in_cols[:i+1]
                        temp_X = X[temp_in_cols].copy()
                        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
                        qof_dict[in_cols[i]] = temp_qof.copy()
                        cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                
                cur_metric = bac_best_qof[metric]

            else:
                break

        else:
            (sel_new_in_cols, sel_new_out_cols, feature_to_add, sel_best_qof, sel_best_cv_stats) = select_single_feature(
                X, y, in_cols, out_cols, nn_type=nn_type, metric=metric
            )

            (bac_new_in_cols, feature_to_remove, bac_best_qof, bac_best_cv_stats) = eliminate_single_feature(
                X, y, in_cols, nn_type=nn_type, metric=metric
            )

            if metric in [0, 1, 12] and ((sel_best_qof[metric] >= cur_metric) or (bac_best_qof[metric] >= cur_metric)):
                if sel_best_qof[metric] >= bac_best_qof[metric]:
                    in_cols = sel_new_in_cols.copy()
                    out_cols = sel_new_out_cols.copy()
                    step_sel_features.append(feature_to_add)
                    qof_dict[feature_to_add] = sel_best_qof.copy()
                    cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                    cur_metric = sel_best_qof[metric]

                elif sel_best_qof[metric] < bac_best_qof[metric]:
                    old_in_cols = in_cols.copy()
                    in_cols = bac_new_in_cols.copy()
                    temp_in_cols = bac_new_in_cols.copy()
                    out_cols = [col for col in X.columns if col not in temp_in_cols]
                    step_sel_features_copy = step_sel_features.copy()
                    step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                    flag = False
                    for i in range(len(old_in_cols)):
                        if old_in_cols[i] == feature_to_remove:
                            flag = True
                        if flag and (i < len(in_cols)):
                            temp_in_cols = in_cols[:i]
                            temp_X = X[temp_in_cols].copy()
                            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
                            qof_dict[in_cols[i]] = temp_qof.copy()
                            cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                    
                    cur_metric = bac_best_qof[metric]

            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and ((sel_best_qof[metric] <= cur_metric) or (bac_best_qof[metric] <= cur_metric)):
                if sel_best_qof[metric] <= bac_best_qof[metric]:
                    in_cols = sel_new_in_cols.copy()
                    out_cols = sel_new_out_cols.copy()
                    step_sel_features.append(feature_to_add)
                    qof_dict[feature_to_add] = sel_best_qof.copy()
                    cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                    cur_metric = sel_best_qof[metric]

                elif sel_best_qof[metric] > bac_best_qof[metric]:
                    old_in_cols = in_cols.copy()
                    in_cols = bac_new_in_cols.copy()
                    temp_in_cols = bac_new_in_cols.copy()
                    out_cols = [col for col in X.columns if col not in temp_in_cols]
                    step_sel_features_copy = step_sel_features.copy()
                    step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                    flag = False
                    for i in range(len(old_in_cols)):
                        if old_in_cols[i] == feature_to_remove:
                            flag = True
                        if flag and (i < len(in_cols)):
                            temp_in_cols = in_cols[:i+1]
                            temp_X = X[temp_in_cols].copy()
                            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, nn_type=nn_type, cv=True)
                            qof_dict[in_cols[i]] = temp_qof.copy()
                            cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                    
                    cur_metric = bac_best_qof[metric]

            else:
                break

    qof_list = []
    cv_stats_list = []

    for col in step_sel_features:
        qof_list.append(qof_dict[col].copy())
        cv_stats_list.append(cv_stats_dict[col].copy())

    return (step_sel_features, qof_list, cv_stats_list)