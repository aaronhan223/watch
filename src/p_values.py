import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn import preprocessing
from utils import *

import torch.optim as optim
import torch.nn.functional as F
import random
from main_mnist_cifar import set_seed, MLP, fit, eval_loss_prob, evaluate, train_one_epoch


## Ofline density ratio estimation
def logistic_regression_weight_est(X, class_labels):
    clf = LogisticRegression(random_state=0).fit(X, class_labels)
    lr_probs = clf.predict_proba(X)
    return lr_probs[:,1] / lr_probs[:,0]


def random_forest_weight_est(X, class_labels, ntree=100):
    rf = RandomForestClassifier(n_estimators=ntree,criterion='entropy', min_weight_fraction_leaf=0.1).fit(X, class_labels)
    rf_probs = rf.predict_proba(X)
    return rf_probs[:,1] / rf_probs[:,0]


def offline_lik_ratio_estimates_images(cal_test_w_est_loader, test_loader, dataset0_name = 'mnist', \
                                       classifier='MLP', device=None, setting='', epochs=60, lr=1e-3):

     # Train smaller MLP model to estimate source/target probabilities
    if dataset0_name == 'mnist':
        model = MLP(input_size=784, hidden_size=32, num_classes=2).to(device)
    elif dataset0_name == 'cifar10':
        model = MLP(input_size=3*32*32, hidden_size=32, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ## Fit prob classifier offline
    fit(model, epochs, cal_test_w_est_loader, optimizer, setting, device)
    ## Evaluate probability estimiates
    cal_test_prob_est, _ = eval_loss_prob(model, device, setting, cal_test_w_est_loader, test_loader, binary_classifier_probs = True)

    return cal_test_prob_est / (1 - cal_test_prob_est)


def online_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only, adapt_start=None, classifier='LR'):
    
    n_cal = len(X_cal)
    init_phase = len(X_test_w_est)
    n_test = len(X_test_0_only)
    T = len(X_test_0_only) + n_cal - adapt_start
    
    W_i = np.zeros((T, adapt_start + T))
    
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
    
    if (adapt_start is None):
        ## Begin adaptive at deployment time, ie first test point after calibration set
        class_labels_all = np.concatenate((np.zeros(n_cal), np.ones(init_phase + n_test)), axis=0)
    else:
        ## Begin adapting at 'adapt_start' (>n_cal), an estimated X-changepoint
        
        class_labels_all = np.concatenate((np.zeros(adapt_start), np.ones(init_phase + T)), axis=0)
        
    idx_include = np.concatenate((np.repeat(True, adapt_start + init_phase), np.repeat(False, T)), axis=0)
        
    ## t=0 : Offline initialization phase (but set warm_start=True)
    ## t>0 : Online adaptation phase
    for t in range(0, T):
        
        lik_ratio_model.fit(X_all_scaled[idx_include], class_labels_all[idx_include])
        
        est_probs = lik_ratio_model.predict_proba(X_cal_test_scaled)
        W_i[t] = est_probs[:,1] / est_probs[:,0]
#         print(f'W_i[{t}][-10:] : ', W_i[t][-10:])
        
        idx_include[adapt_start + init_phase + t] = True
    
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


def calculate_p_values_and_quantiles(conformity_scores, alpha, cs_type='abs'):
    """
    Calculate the conformal p-values from conformity scores.
    """
    n = len(conformity_scores)
    q_lower = np.zeros(n)
    q_upper = np.zeros(n)
    
    ## Quantiles: Each ith quantile is computed before labels observed, with conservative validity
    if (cs_type != 'signed'):
        conformity_scores_inf = np.concatenate((conformity_scores, [np.inf])) ## inf in place of test pt cs for conservativeness
        idx_include_inf = np.concatenate((np.repeat(False, n), [True]), axis=0) ## indicies to include
        
        for i in range(n):
            ## For each i, q_upper[i] := quantile(conformity_scores[:i] \cup inf, 1-alpha)
                        ## q_lower[i] := -q_upper[i]
            q_upper[i] = np.quantile(conformity_scores_inf[idx_include_inf], 1-alpha)
            idx_include_inf[i] = True
        q_lower = - q_upper
        
    else:
        ## Intervals for signed scores computed by (alpha/2) lower q, (1-alpha/2) upper q
        conformity_scores_inf = np.concatenate((conformity_scores, [np.inf]))
        conformity_scores_neg_inf = np.concatenate((conformity_scores, [-np.inf]))
        idx_include_inf = np.concatenate((np.repeat(False, n), [True]), axis=0) ## indicies to include
        
        for i in range(n):
            ## For each i, q_upper[i] := quantile(conformity_scores[:i] \cup inf, 1-alpha/2)
                        ## q_lower[i] := quantile(conformity_scores[:i] \cup -inf, alpha/2)
            q_upper[i] = np.quantile(conformity_scores_inf[idx_include_inf], 1-alpha/2)
            q_lower[i] = np.quantile(conformity_scores_neg_inf[idx_include_inf], alpha/2)
            idx_include_inf[i] = True

    
    ## P-values: Each ith p-value is computed after labels observed, breaking ties randomly for exact validity
    p_values = calculate_p_values(conformity_scores)
    
    ## Replace nan with appropriate inf values
    np.nan_to_num(q_lower, copy=False, nan=-np.inf, neginf=-np.inf)
    np.nan_to_num(q_upper, copy=False, nan=np.inf, posinf=np.inf)
        
    return p_values , q_lower, q_upper


## Note: This is for calculating the weighted p-values once the normalized weights have already been calculated
def calculate_weighted_p_values_and_quantiles(args, conformity_scores, W_i, adapt_start, method, depth=1):
    """
    Calculate the weighted conformal p-values from conformity scores and given normalized weights 
    (i.e., enforce np.sum(normalized_weights) = 1).
    
    W_i : List of likelihood ratio weight est. arrays, each t-th array is length (adapt_start+t)
    adapt_start : Index of first point that is assumed part of test distribution rather than cal. If method != 'fixed_cal_dyn', 
                  then  adapt_start==n_cal
    """
    n = len(conformity_scores)
    wp_values = np.zeros(n) ## p-values calculated with weighted conformity scores
    wq_lower = np.zeros(n) ## lower weighted quantiles
    wq_upper = np.zeros(n) ## upper weighted quantiles

    ## For 0:adapt_start, compute as standard p-values and quantiles
    wp_values[0:adapt_start], wq_lower[0:adapt_start], wq_upper[0:adapt_start] = \
                                        calculate_p_values_and_quantiles(conformity_scores[0:adapt_start], args.alpha, args.cs_type) 
    
    ## For computing (conservative) weighted quantiles, append infinity (which takes place of test pt score)
    conformity_scores_inf = np.concatenate((conformity_scores, [np.inf]))
    if (args.cs_type == 'signed'):
        conformity_scores_neg_inf = np.concatenate((conformity_scores, [-np.inf])) 

    if method in ['fixed_cal', 'fixed_cal_oracle', 'sliding_window', 'fixed_cal_offline', 'fixed_cal_dyn']:

        if method == 'fixed_cal':
            assert depth == 1, "Estimation depth must be 1."
        elif method == 'sliding_window':
            assert depth > 1, "Estimation depth must be greater than 1."

        T = len(conformity_scores) - adapt_start ## Number of total test observations
        ## indices to include in computing weighted p-values
        idx_include = np.concatenate((np.repeat(True, adapt_start), np.repeat(False, T)), axis=0) 
        
        ## indices to include in computing weighted quantiles, where last entry includes np.inf
        idx_include_inf = np.concatenate((np.repeat(True, adapt_start), np.repeat(False, T), [True]), axis=0) 
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## idx_include implements indices for 'fixed cal' ie, comparing to [0:adapt_start] \cup adapt_start + t_ 
            idx_include[adapt_start+t_] = True ## Move to curr test point
            if (t_ > depth - 1):
                idx_include[adapt_start+t_-depth] = False ## Exclude most recent "depth" number of test point again (except at start, is a cal point)
            
            ## Subset conformity scores and weights based on idx_include
            conformity_scores_t = conformity_scores[idx_include]
            
            if (method in ['fixed_cal_oracle', 'fixed_cal_offline']):
                ## 
                W_i_t = W_i[idx_include]
            else:
                W_i_t = W_i[t_][idx_include]
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
            ## Calculate weighted quantiles
            if (args.cs_type != 'signed'):
                wq_upper[adapt_start+t_] = weighted_quantile(conformity_scores_inf[idx_include_inf], normalized_weights_t, 1-args.alpha)
                wq_lower[adapt_start+t_] = - wq_upper[adapt_start+t_]
            else:
                ## Intervals for signed scores computed by (alpha/2) lower q, (1-alpha/2) upper q
                wq_upper[adapt_start+t_] = weighted_quantile(conformity_scores_inf[idx_include_inf], normalized_weights_t, 1-args.alpha/2)
                wq_lower[adapt_start+t_] = weighted_quantile(conformity_scores_neg_inf[idx_include_inf], normalized_weights_t, args.alpha/2)
#             idx_include_inf[adapt_start+t_] = True
            
            ## Calculate weighted p-values
#             test_pt_weight = np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
#             conservative_p=True
            
#             if (conservative_p):
#                 ## Exact p-values with uniform-randomization to break ties
#                 wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t <= conformity_scores_t[-1]])
            
#             else:
#                 ## Exact p-values with uniform-randomization to break ties
    
            ## 
            if (np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]]) < args.alpha):
                ## If no more than (relative) weight 'alpha' put on test point score, compute exact (randomized) p-values:
                wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
            else:
                ## Else: over (relative) weight 'alpha' put on test pt score, compute conservative (and deterministic) p-values:
                print("Using conservative p-values : ", t_)
                wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]])

    elif (method in ['one_step_oracle', 'one_step_est', 'batch_oracle', 'multistep_oracle']):
        
        T = len(conformity_scores) - adapt_start ## Number of total test observations
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
            ## Note: Previously this loop started at init_phase
        for t_ in range(0, T):
            
            ## Subset conformity scores and weights based on idx_include
            conformity_scores_t = conformity_scores[:(adapt_start+t_+1)]
            
            if (method in ['one_step_est', 'batch_oracle']):
                W_i_t = W_i[t_][:(adapt_start+t_+1)]
                
            elif (method == 'one_step_oracle'):
                W_i_t = W_i[:(adapt_start+t_+1)]
                
            elif (method == 'multistep_oracle'):
                w_mat = np.matrix(W_i[-(t_+2):,:(adapt_start+t_+1)])
                
                W_i_t = compute_w_ptest_split_active_replacement(w_mat, depth_max=2)
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)

            ## Calculate weighted p-values
            wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
           
    else:
        raise Exception("Not implemented")
        
    ## Replace nan with appropriate inf values
    np.nan_to_num(wq_lower, copy=False, nan=-np.inf, neginf=-np.inf)
    np.nan_to_num(wq_upper, copy=False, nan=np.inf, posinf=np.inf)
        
    return wp_values, wq_lower, wq_upper


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
