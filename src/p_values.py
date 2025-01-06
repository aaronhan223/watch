import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn import preprocessing
from utils import *


## Ofline density ratio estimation
def logistic_regression_weight_est(X, class_labels):
    clf = LogisticRegression(random_state=0).fit(X, class_labels)
    lr_probs = clf.predict_proba(X)
    return lr_probs[:,1] / lr_probs[:,0]

def random_forest_weight_est(X, class_labels, ntree=100):
    rf = RandomForestClassifier(n_estimators=ntree,criterion='entropy', min_weight_fraction_leaf=0.1).fit(X, class_labels)
    rf_probs = rf.predict_proba(X)
    return rf_probs[:,1] / rf_probs[:,0]


def online_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only, classifier='LR'):
    
    n_cal = len(X_cal)
    init_phase = len(X_test_w_est)
    n_test = len(X_test_0_only)
    
    W_i = np.zeros((n_test, n_cal + n_test))
    
    if (classifier=='LR'):
        lik_ratio_model = LogisticRegression(warm_start=True)
        
    elif (classifier=='RF'):
        lik_ratio_model = RandomForestClassifier(n_estimators=ntree,criterion='entropy', min_weight_fraction_leaf=0.1,\
                                                 warm_start=True)
    
    ## Scale data
    X_all = np.concatenate((X_cal, X_test_w_est, X_test_0_only), axis=0)
    scaler = preprocessing.StandardScaler().fit(X_all)
    X_all_scaled = scaler.transform(X_all)
    X_cal_test_scaled = np.concatenate((X_all_scaled[0:n_cal], X_all_scaled[-n_test:]), axis=0)
    
    class_labels_all = np.concatenate((np.zeros(n_cal), np.ones(init_phase + n_test)), axis=0)
    idx_include = np.concatenate((np.repeat(True, n_cal + init_phase), np.repeat(False, n_test)), axis=0)
        
        
    ## t=0 : Offline initialization phase (but set warm_start=True)
    ## t>0 : Online adaptation phase
    for t in range(0, n_test):
        
        lik_ratio_model.fit(X_all_scaled[idx_include], class_labels_all[idx_include])
        
        est_probs = lik_ratio_model.predict_proba(X_cal_test_scaled)
        W_i[t] = est_probs[:,1] / est_probs[:,0]
#         print(f'W_i[{t}][-10:] : ', W_i[t][-10:])
        
        idx_include[n_cal + init_phase + t] = True
    
    
    return W_i



def offline_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only, classifier='LR'):
    n_cal = len(X_cal)
    init_phase = len(X_test_w_est)
    n_test = len(X_test_0_only)
    
    class_labels = np.concatenate((np.zeros(n_cal), np.ones(init_phase)), axis=0)
    
    ## Scale data
    X_all = np.concatenate((X_cal, X_test_w_est, X_test_0_only), axis=0)
    scaler = preprocessing.StandardScaler().fit(X_all)
    X_all_scaled = scaler.transform(X_all)
    
    ## Fit lik_ratio_model using cal + init_phase data
    if (classifier=='LR'):
        lik_ratio_model = LogisticRegression(random_state=0).fit(X_all_scaled[0:(n_cal + init_phase)], class_labels)
        
    elif (classifier=='RF'):
        lik_ratio_model = RandomForestClassifier(n_estimators=ntree,criterion='entropy',\
                                                 min_weight_fraction_leaf=0.1).fit(X_all_scaled[0:(n_cal + init_phase)], class_labels)
        
    X_cal_test_scaled = np.concatenate((X_all_scaled[0:n_cal], X_all_scaled[-n_test:]), axis=0)
    
    est_probs = lik_ratio_model.predict_proba(X_cal_test_scaled)
    return est_probs[:,1] / est_probs[:,0]


def calculate_p_values(conformity_scores):
    """
    Calculate the conformal p-values from conformity scores.
    """
    
    n = len(conformity_scores)
    p_values = np.array([(np.sum(conformity_scores[:i] < conformity_scores[i]) + 
                         np.random.uniform() * np.sum(conformity_scores[:i] == conformity_scores[i])) / (i + 1)
                         for i in range(n)])
    return p_values


## Note: This is for calculating the weighted p-values once the normalized weights have already been calculated
def calculate_weighted_p_values(conformity_scores, W_i, n_cal, method='fixed_cal_oracle', depth=1):
    """
    Calculate the weighted conformal p-values from conformity scores and given normalized weights 
    (i.e., enforce np.sum(normalized_weights) = 1).
    
    W_i : List of likelihood ratio weight est. arrays, each t-th array is length (n_cal+t)
    """
    init_phase = 0
    wp_values = np.zeros(len(conformity_scores))
    
    
    ## p-values for original calibration set calculated as before (REVISIT THIS LINE)
    wp_values[0:(n_cal+init_phase)] = calculate_p_values(conformity_scores[0:(n_cal+init_phase)]) 
        
    if method in ['fixed_cal', 'fixed_cal_oracle', 'sliding_window', 'fixed_cal_offline']:

        if method == 'fixed_cal':
            assert depth == 1, "Estimation depth must be 1."
        elif method == 'sliding_window':
            assert depth > 1, "Estimation depth must be greater than 1."

        T = len(conformity_scores) - n_cal ## Number of total test observations
        idx_include = np.concatenate((np.repeat(True, n_cal), np.repeat(False, T)), axis=0) ## indicies to include
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## idx_include implements indices for 'fixed cal' ie, comparing to [0:n_cal] \cup n_cal + t_ 
            idx_include[n_cal+t_] = True ## Move to curr test point
            if (t_ > depth - 1):
                idx_include[n_cal+t_-depth] = False ## Exclude most recent "depth" number of test point again (except at start, is a cal point)
            
            ## Subset conformity scores and weights based on idx_include
            conformity_scores_t = conformity_scores[idx_include]
            
            if (method in ['fixed_cal_oracle', 'fixed_cal_offline']):
                W_i_t = W_i[idx_include]
            else:
                W_i_t = W_i[t_][idx_include]
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
            ## Calculate weighted p-values
#             test_pt_weight = np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
            wp_values[n_cal+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])

    elif (method in ['one_step_oracle', 'one_step_est', 'batch_oracle', 'multistep_oracle']):
        
        T = len(conformity_scores) - n_cal ## Number of total test observations
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
            ## Note: Previously this loop started at init_phase
        for t_ in range(0, T):
            
            ## Subset conformity scores and weights based on idx_include
            conformity_scores_t = conformity_scores[:(n_cal+t_+1)]
            
            if (method in ['one_step_est', 'batch_oracle']):
                W_i_t = W_i[t_][:(n_cal+t_+1)]
                
            elif (method == 'one_step_oracle'):
                W_i_t = W_i[:(n_cal+t_+1)]
                
            elif (method == 'multistep_oracle'):
                w_mat = np.matrix(W_i[-(t_+2):,:(n_cal+t_+1)])
                
                W_i_t = compute_w_ptest_split_active_replacement(w_mat, depth_max=2)
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
                        
            ## Calculate weighted p-values
            wp_values[n_cal+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
           
                
    
    else:
        raise Exception("Not implemented")
        
    return wp_values





def subsample_batch_weights(w_array_all, n_cal, max_num_samples=20):
    ## w_array_all: length n_cal + T
    ## out: W, list with weight arrays
    
    W_list = []
    for t in range(1, len(w_array_all) - n_cal + 1):
#         print(t)
        W_list.append(subsample_batch_weights_helper_t(w_array_all[:(n_cal + t)], t, max_num_samples=max_num_samples))
    return W_list


def subsample_batch_weights_helper_t(w_array, t, max_num_samples=20):
    ## w_array: length n_cal + t
    
#     max_num_samples = np.min([math.comb(len(w_array), t), max_num_samples])
    
    idx_not_i = np.repeat(False, len(w_array))
    w_sums = np.zeros(len(w_array))
        
    for i in range(0, len(w_array)):
#         print(i)
        idx_not_i[i] = True
        idx_not_i[i-1] = False
        w_array_not_i = w_array[idx_not_i]
        
        ## Subsampling
        s = 0
        while (s < max_num_samples):
            w_sums[i] += np.prod(np.random.choice(w_array_not_i, size=t))
            s += 1
    
    return w_array * w_sums