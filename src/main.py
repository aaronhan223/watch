import numpy as np
import pandas as pd
from plot import plot_martingale_paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import pdb

from utils import *
from martingales import *
from p_values import *
import argparse
import os
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math
from podkopaev_ramdas.baseline_alg import podkopaev_ramdas_algorithm1, podkopaev_ramdas_changepoint_detection
import time


def train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name='RF', seed=0, cs_type='signed',\
                       methods=['fixed_cal_oracle', 'none'], dataset0_name='white_wine', cov_shift_bias=0, init_phase=500,\
                      x_ctm_thresh=None, x_sched_thresh=None):
    fold_results = []
    cs_0 = []
    errors_0 = [] ## Prediction errors (absolute value residuals); recorded regardless of score function used
    cs_1 = []
    W_dict = {}
    
    for method in methods:
        W_dict[method] = []
        
#     W = [] ## Will contain estimated likelihood ratio weights for each fold
    adapt_starts = [] ## Index of test point where adaptation should begin. If method != fixed_cal_dyn, then this is equal to num cal points in each fold
    n_cals = []
    
    y_name = dataset0_test_0.columns[-1] ## Outcome must be last column
    
    
    ## Allocate some test points for density-ratio estimation:
    print("init_phase : ", init_phase)
    dataset0_test_w_est = dataset0_test_0.iloc[:init_phase]
    X_test_w_est = dataset0_test_w_est.drop(y_name, axis=1).to_numpy()
    ## Test points used in eval are all those not used for density-ratio estimation
    dataset0_test_0 = dataset0_test_0.iloc[init_phase:]
    
    for i, (train_index, cal_index) in enumerate(folds):
        print("fold : ", i)
            
        if i == 2:  # Adjust the last fold to have 1099 in training
            train_index, cal_index = train_index[:-1], cal_index
        X_train, X_cal = X[train_index], X[cal_index]
        y_train, y_cal = y[train_index], y[cal_index]
        
        n_cal = len(X_cal)
        n_cals.append(n_cal)
        
        # Train the model on the training set proper
        if (muh_fun_name == 'RF'):
            model = Pipeline([
                ('scaler', StandardScaler()),  # Normalize the data
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=seed))
            ])
        elif (muh_fun_name == 'NN'):
            model = Pipeline([
                ('scaler', StandardScaler()),  # Normalize the data
                ('regressor', MLPRegressor(solver='lbfgs',activation='logistic', random_state=seed))
            ])
        print("fitting model")
        model.fit(X_train, y_train)
        print("model fitted")
        
        ## Save test set alone
        X_test_0_only = dataset0_test_0.drop(y_name, axis=1).to_numpy()
        y_test_0_only = dataset0_test_0[y_name].to_numpy()
        
        # Evaluate using the calibration set + test set 0 
        X_cal_test_0 = np.concatenate((X_cal, X_test_0_only), axis=0)
        y_cal_test_0 = np.concatenate((y_cal, y_test_0_only), axis=0)
        y_pred_0 = model.predict(X_cal_test_0)
        
        ## Dynamically determine when to start weight estimation using strictly X-dependent CTM
        if ('fixed_cal_dyn' in methods):
            assert (x_ctm_thresh is not None and x_sched_thresh is not None)
            
            
#             print("x_ctm_thresh : ", x_ctm_thresh)
            ## NN distance conformity score
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_train)
            distances, _ = nbrs.kneighbors(X_cal_test_0)
            
            ## Distance from centroid conformity score:
#             centroid_train = np.mean(X_train, axis=0)
#             distances = (X_cal_test_0 - centroid_train).sum(axis=1)**2
            X_conformity_scores_0 = distances.flatten()
            p_values = calculate_p_values(X_conformity_scores_0)
            
            ## Run martingale on test pt p-values, ie on p_values[n_cal:]
            _, martingale_value_test = composite_jumper_martingale(p_values[(n_cal):]) ## simulate that first 500 points are IID, then shift at 501st test point
            _, sigma_test = shiryaev_roberts_procedure(martingale_value_test, 100)
            
            ## Test point index where X-CTM first exceeds x_ctm_thresh*sigma[n_cal-1]
            x_alarm_idx = n_cal + np.argmax(np.bitwise_or(sigma_test>=x_sched_thresh, martingale_value_test>=x_ctm_thresh)) if (np.max(sigma_test)>=x_sched_thresh or np.max(martingale_value_test)>=x_ctm_thresh) else len(X_cal_test_0-1) 
            

            adapt_starts.append(x_alarm_idx) ## Update where to start adaptation
            
                        
        else:
            ## Default is to begin adaption immediately after calibration set, ie n_cal
            adapt_starts.append(n_cal)
         
                   

        # Evaluate using the calibration set + test set 1 (Scenario 1)
        if (dataset1 is not None):
            X_test_1 = np.concatenate((X_cal, dataset1.drop(y_name, axis=1)), axis=0)
            y_test_1 = np.concatenate((y_cal, dataset1[y_name]), axis=0)
            y_pred_1 = model.predict(X_test_1)
        
        np.set_printoptions(threshold=np.inf)
        
        if (cs_type == 'signed'):
            conformity_scores_0 = y_cal_test_0 - y_pred_0
            
        elif (cs_type == 'abs'):
            conformity_scores_0 = np.abs(y_cal_test_0 - y_pred_0)
            print("conformity_scores_0 shape : ", np.shape(conformity_scores_0))
        elif (cs_type == 'nn_dist'):
            ## CS is nearest neighbor distance
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_train)
            distances, _ = nbrs.kneighbors(X_cal_test_0)
            conformity_scores_0 = distances.flatten()
            
        cs_0.append(conformity_scores_0)
        errors_0.append(np.abs(y_cal_test_0 - y_pred_0))
        
        if (dataset1 is not None):
            if (cs_type == 'signed'):
                conformity_scores_1 = y_test_1 - y_pred_1
            elif (cs_type == 'abs'):
                conformity_scores_1 = np.abs(y_test_1 - y_pred_1)
                
            cs_1.append(conformity_scores_1)

            # Store results for each fold
            fold_results.append({
                'fold': i + 1,
                'scenario_0_predictions': y_pred_0,
                'scenario_1_predictions': y_pred_1
            })
                        
        else:
            # Store results for each fold
            fold_results.append({
                'fold': i + 1,
                'scenario_0_predictions': y_pred_0,
            })
            
            
            
        #### Computing (unnormalized) weights
        for method in methods:
            
            ## Online logistic regression for weight estimation
            W_i = [] ## List of weight est. arrays, each t-th array is length (n+t)
            

            if (method in ['fixed_cal', 'one_step_est']):
                ## Estimating likelihood ratios for each cal, test point
                ## np.shape(W_i) = (T, n_cal + T)
                W_i = online_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only, adapt_start=n_cal)
            
            elif (method in ['fixed_cal_dyn']):
                ## fixed_cal except with dynamically/automatically determined start to adaptation
                W_i = online_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only, adapt_start=adapt_starts[i])
            

            elif (method in ['fixed_cal_offline']):
#                 W_i = offline_lik_ratio_estimates(X_cal, X_test_w_est, X_cal_test_0)
                W_i = offline_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only)

            elif (method in ['fixed_cal_oracle','one_step_oracle', 'batch_oracle', 'multistep_oracle']):
    #             print("getting oracle lik ratios")
                ## Oracle one-step likelihood ratios
                ## np.shape(W_i) = (n_cal + T, )
                X_full = np.concatenate((X_train, X_cal_test_0), axis = 0)
                
                W_i = get_w(x_pca=X_train, x=X_cal_test_0, dataset=dataset0_name, bias=cov_shift_bias) 
                print("obtained weights")

                if (method == 'batch_oracle'):
                    W_i = (W_i - min(W_i)) / (max(W_i) - min(W_i))
                    W_i = subsample_batch_weights(W_i, n_cal, max_num_samples=100)


                if (method == 'multistep_oracle'):
                    W_i = np.tile(W_i, (len(X_cal_test_0) - n_cal, 1))
                    
            else:
                ## Else: Unweighted / uniform-weighted CTM
                W_i = np.ones(len(X_cal_test_0))

            W_dict[method].append(W_i)
            
                
    return cs_0, cs_1, W_dict, adapt_starts, n_cals, errors_0


def retrain_count(conformity_score, training_schedule, sr_threshold, cu_confidence, W_i, adapt_start, n_cal, alpha=0.1,\
                  cs_type='abs',verbose=False, method='fixed_cal_oracle', depth=1, init_ctm_on_cal_set=True):
    
    if (method in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle', 'fixed_cal_offline', 'fixed_cal_dyn']):
        p_values, q_lower, q_upper = calculate_weighted_p_values_and_quantiles(conformity_score, W_i, adapt_start, alpha, cs_type, method)
        
    else:
        p_values, q_lower, q_upper = calculate_p_values_and_quantiles(conformity_score, alpha, cs_type)
        
    print("init_ctm_on_cal_set : ", init_ctm_on_cal_set)
        
    if (init_ctm_on_cal_set):
        ## Initialize CTM on calibration set, as in Vovk et al. 2021
        retrain_m, martingale_value = composite_jumper_martingale(p_values, verbose=verbose)
    else:
        ## Initialize CTM at deployment time (ie, not including calibration set) to facilitate comparison to other methods
#         p_values = p_values[n_cal:]
#         q_lower  = q_lower[n_cal:]
#         q_upper  = q_upper[n_cal:]
        
        retrain_m, martingale_value = composite_jumper_martingale(p_values[n_cal:], verbose=verbose)
        
    
    
    if training_schedule == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
        
    elif (training_schedule == 'basic'):
#         print("martingale_value shape :", np.shape(martingale_value))
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
#         print("SR shape :", np.shape(sigma))
        sigma = martingale_value
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence, verbose)
        

    return retrain_m, retrain_s, martingale_value, sigma, p_values, q_lower, q_upper





def training_function(dataset0, dataset0_name, dataset1=None, training_schedule='variable', \
                      sr_threshold=1e6, cu_confidence=0.99, muh_fun_name='RF', test0_size=1599/4898, \
                      dataset0_shift_type='none', cov_shift_bias=1.0, plot_errors=False, seed=0, cs_type='signed', \
                      label_uptick=1, verbose=False, noise_mu=0, noise_sigma=0, methods=['fixed_cal_oracle', 'none'],\
                      depth=1,init_phase=500, num_folds=3, x_ctm_thresh=None, x_sched_thresh=None, alpha=0.1,\
                      num_test_unshifted=500, run_PR_ST=True, run_PR_CD=False, pr_source_conc_type='betting', \
                      pr_target_conc_type='betting', pr_eps_tol=0.05, pr_source_delta=0.025, \
                      pr_target_delta = 0.025,pr_st_stop_criterion='first_alarm',pr_cd_stop_criterion='first_alarm',\
                      init_ctm_on_cal_set=True):
    
    
    
    dataset0_train, dataset0_test_0 = split_and_shift_dataset0(dataset0, dataset0_name, test0_size=test0_size, \
                                                               dataset0_shift_type=dataset0_shift_type, \
                                                               cov_shift_bias=cov_shift_bias, seed=seed, \
                                                               label_uptick=label_uptick, noise_mu=noise_mu,\
                                                                noise_sigma=noise_sigma, num_test_unshifted=num_test_unshifted)
    
    X, y, folds = split_into_folds(dataset0_train, num_folds=num_folds, seed=seed)
        
#     ## Add simulated measurement noise with OLS
#     ols = LinearRegression(fit_intercept=False)  # featurization from walsh_hadamard_from_seqs has intercept
#     ols.fit(X, y)
#     y_pred = ols.predict(X)
#     resid = np.abs(y - y_pred)
#     y = y + np.random.normal(0, 0.1*resid)

    ## Add simulated measurement noise with Kernel Ridge
#     kernel_ridge = KernelRidge(alpha=0.1)  # featurization from walsh_hadamard_from_seqs has intercept
#     kernel_ridge.fit(X, y)
#     y_pred = kernel_ridge.predict(X)
#     resid = np.abs(y - y_pred)
#     y = y + np.random.normal(0, resid)

   


    cs_0, cs_1, W_dict, adapt_starts, n_cals, errors_0 = train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name, seed=seed, cs_type=cs_type, methods=methods, dataset0_name=dataset0_name, cov_shift_bias=cov_shift_bias, init_phase=init_phase, x_ctm_thresh=x_ctm_thresh, x_sched_thresh=x_sched_thresh)
    
    
    
    martingales_0_dict, martingales_1_dict = {}, {}
    sigmas_0_dict, sigmas_1_dict = {}, {}
    retrain_m_count_0_dict, retrain_s_count_0_dict = {}, {}
    retrain_m_count_1_dict, retrain_s_count_1_dict = {}, {}
    p_values_0_dict = {}
    coverage_0_dict = {}
    widths_0_dict = {}
    
    if run_PR_ST:
        ## Podkopaev Ramdas sequential testing method
        PR_ST_alarm_0_dict, PR_ST_alarm_1_dict = {}, {}
        PR_ST_source_UCB_tols_0_dict, PR_ST_source_UCB_tols_1_dict = {}, {}
        PR_ST_target_LCBs_0_dict, PR_ST_target_LCBs_1_dict = {}, {}
        
    if run_PR_CD:
        ## Podkopaev Ramdas changepoint detection method
        PR_CD_alarm_0_dict, PR_CD_alarm_1_dict = {}, {}
        PR_CD_source_UCB_tols_0_dict, PR_CD_source_UCB_tols_1_dict = {}, {}
        PR_CD_target_LCBs_0_dict, PR_CD_target_LCBs_1_dict = {}, {}
        
        
    
    for method in methods:
        martingales_0_dict[method], martingales_1_dict[method] = [], []
        sigmas_0_dict[method], sigmas_1_dict[method] = [], []
        retrain_m_count_0_dict[method], retrain_s_count_0_dict[method] = [], []
        retrain_m_count_1_dict[method], retrain_s_count_1_dict[method] = [], []
        p_values_0_dict[method] = []
        coverage_0_dict[method] = []
        widths_0_dict[method] = []
        
        if run_PR_ST:
            PR_ST_alarm_0_dict['PR_ST_cp_'+method], PR_ST_alarm_1_dict['PR_ST_cp_'+method] = [], []
            PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method], PR_ST_source_UCB_tols_1_dict['PR_ST_cp_'+method] = [], []
            PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method], PR_ST_target_LCBs_1_dict['PR_ST_cp_'+method] = [], []
            
        if run_PR_CD:
            PR_CD_alarm_0_dict['PR_CD_cp_'+method], PR_CD_alarm_1_dict['PR_CD_cp_'+method] = [], []
            PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method], PR_CD_source_UCB_tols_1_dict['PR_CD_cp_'+method] = [], []
            PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method], PR_CD_target_LCBs_1_dict['PR_CD_cp_'+method] = [], []
            

#     fold_martingales_0, fold_martingales_1 = [], []
#     sigmas_0, sigmas_1 = [], []
#     retrain_m_count_0, retrain_s_count_0 = 0, 0
#     retrain_m_count_1, retrain_s_count_1 = 0, 0
#     p_values_0 = []
#     coverage_0 = []
        
    for i, score_0 in enumerate(cs_0):
        n_cal = n_cals[i]
            
        for method in methods:
            
            if (method in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle', 'fixed_cal_offline', 'fixed_cal_dyn']):
                m_0, s_0, martingale_value_0, sigma_0, p_vals, q_lower, q_upper = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence, W_dict[method][i], adapt_starts[i], n_cal, alpha, cs_type, verbose, method, depth, init_ctm_on_cal_set=init_ctm_on_cal_set)
                
            else:
                ## Run baseline with uniform weights
                m_0, s_0, martingale_value_0, sigma_0, p_vals, q_lower, q_upper = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence, None, adapt_starts[i], n_cal, alpha, cs_type, verbose, method, depth, init_ctm_on_cal_set=init_ctm_on_cal_set)
                
                
            

            if m_0:
                retrain_m_count_0_dict[method] += 1
            if s_0:
                retrain_s_count_0_dict[method] += 1
                
            martingales_0_dict[method].append(martingale_value_0)
            sigmas_0_dict[method].append(sigma_0)

            ## Storing p-values
            p_values_0_dict[method].append(p_vals)
            coverage_0_dict[method].append(((q_lower <= score_0)&(q_upper >= score_0)))
            widths_0_dict[method].append(q_upper - q_lower)
            

#             coverage_0_dict[method].append(p_vals <= 0.9)

            if (not init_ctm_on_cal_set):
                p_vals = p_vals[n_cal:]
                coverage_vals = coverage_vals[n_cal:]
                width_vals = width_vals[n_cal:]

            p_values_0_dict[method].append(p_vals)
            coverage_0_dict[method].append(coverage_vals)
            widths_0_dict[method].append(width_vals)
                
         
        if (not init_ctm_on_cal_set):
            cs_0[i] = cs_0[i][n_cal:]
            errors_0[i] = errors_0[i][n_cal:] 
    
    ## Note: 
    for i, score_1 in enumerate(cs_1):
        for method in methods:
            m_1, s_1, martingale_value_1, sigma_1 = retrain_count(score_1, training_schedule, sr_threshold, cu_confidence, W[i], adapt_starts[i], alpha, cs_type, verbose, method, depth)

            if m_1:
                retrain_m_count_1_dict[method] += 1
            if s_1:
                retrain_s_count_1_dict[method] += 1
            martingales_1_dict[method].append(martingale_value_1)
            sigmas_1_dict[method].append(sigma_1)
            
            
            if run_PR_ST:
                raise ("Error: have not implemented PodRamdas baseline for dataset1. Need to also compute CP quantiles and coverage values.")
              
            
   
    ### Don't need this part for plotting purposes ###
    # Decide to retrain based on two out of three martingales exceeding the threshold
    # if retrain_m_count_0 >= 2 or retrain_s_count_0 >= 2:
    #     retrain_decision_0 = True
    # else:
    #     retrain_decision_0 = False

    # if retrain_m_count_1 >= 2 or retrain_s_count_1 >= 2:
    #     retrain_decision_1 = True
    # else:
    #     retrain_decision_1 = False

    # if retrain_decision_0:
    #     print("Retraining the model for normal white wine...")
    # else:
    #     print("No retraining needed for normal white wine.")

    # if retrain_decision_1:
    #     print("Retraining the model for red wine...")
    # else:
    #     print("No retraining needed for red wine.")
    ### Don't need this part for plotting purposes ###
        
    ## min_len : Smallest fold length, for clipping longer ones to all same length
    min_len = np.min([len(sigmas_0_dict[method][i]) for i in range(0, len(sigmas_0_dict[method]))])
    
    paths_dict = {}
    PR_ST_paths_dict = {}
    PR_ST_paths=None
    
    PR_CD_paths_dict = {}
    PR_CD_paths=None
    
    for method in methods:
    
        paths = pd.DataFrame(np.c_[np.repeat(seed, min_len), np.arange(0, min_len)], columns = ['itrial', 'obs_idx'])
        ## For each fold:
        
        ## (W)CTM methods:
        sigmas_0 = sigmas_0_dict[method]
        sigmas_1 = sigmas_1_dict[method]
        print("min_len : ", min_len)
        for k in range(0, len(sigmas_0_dict[method])):
            paths['sigmas_0_'+str(k)] = sigmas_0_dict[method][k][0:min_len]
            paths['martingales_0_'+str(k)] = martingales_0_dict[method][k][0:min_len]
#             paths['cs_0_'+str(k)] = cs_0[k][0:min_len]
            paths['errors_0_'+str(k)] = errors_0[k][0:min_len]
            paths['pvals_0_'+str(k)] = p_values_0_dict[method][k][0:min_len]
            paths['coverage_0_'+str(k)] = coverage_0_dict[method][k][0:min_len]
            paths['widths_0_'+str(k)] = widths_0_dict[method][k][0:min_len]


        for k in range(0, len(sigmas_1)):
            paths['sigmas_1_'+str(k)] = sigmas_1[k][0:min_len]
            paths['martingales_1_'+str(k)] = martingales_1_dict[method][k][0:min_len]
#             paths['cs_1_'+str(k)] = cs_1[k][0:min_len]

        paths_dict[method] = paths
    
        
        
        if run_PR_ST:
            ## PR_ST_cp baseline method:
            PR_ST_min_len = len(PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method][0])
            PR_ST_paths = pd.DataFrame(np.c_[np.repeat(seed, PR_ST_min_len), np.arange(0, PR_ST_min_len)], columns = ['itrial', 'obs_idx'])
            for k in range(0, len(PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method])):
                PR_ST_paths['PR_ST_alarm_0_'+str(k)] = PR_ST_alarm_0_dict['PR_ST_cp_'+method][k]
                PR_ST_paths['PR_ST_UCBtol_0_'+str(k)] = PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method][k]
                PR_ST_paths['PR_ST_LCB_0_'+str(k)] = PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method][k][0:PR_ST_min_len]
                
            PR_ST_paths_dict['PR_ST_cp_'+method] = PR_ST_paths
            
        if run_PR_CD:
            ## PR_CD_cp baseline method:
            PR_CD_min_len = len(PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method][0])
            PR_CD_paths = pd.DataFrame(np.c_[np.repeat(seed, PR_CD_min_len), np.arange(0, PR_CD_min_len)], columns = ['itrial', 'obs_idx'])
            for k in range(0, len(PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method])):
                PR_CD_paths['PR_CD_alarm_0_'+str(k)] = PR_CD_alarm_0_dict['PR_CD_cp_'+method][k]
                PR_CD_paths['PR_CD_UCBtol_0_'+str(k)] = PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method][k]
                PR_CD_paths['PR_CD_LCB_0_'+str(k)] = PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method][k][0:PR_CD_min_len]
                
            PR_CD_paths_dict['PR_CD_cp_'+method] = PR_CD_paths
            
            
    
    return paths_dict, PR_ST_paths_dict, PR_CD_paths_dict

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run WTR experiments.')
    
    parser.add_argument('--dataset0', type=str, default='white_wine', \
                        help='Training/cal dataset for expts; Shifted split of dataset0 used for test set 0.')
    parser.add_argument('--dataset1', type=str, default=None, \
                        help='(Optional) Dataset for test set 1; Test dataset which may differ from dataset0.')
    parser.add_argument('--muh_fun_name', type=str, default='RF', help='Mu (mean) function predictor. RF or NN.')
    parser.add_argument('--test0_size', type=float, default=1599/4898, \
                        help='value in (0,1); Proportion of dataset0 used for testing')
    parser.add_argument('--verbose', action='store_true', help="Whether to print out alarm raising info.")
    parser.add_argument('--d0_shift_type', type=str, default='none', help='Shift type to induce in dataset0.')
    parser.add_argument('--depth', type=int, default=1, help="Estimation depth for sliding window approach.")
    parser.add_argument('--bias', type=float, default=0.0, help='Scalar bias magnitude parameter lmbda for exponential tilting covariate shift.')
    parser.add_argument('--plot_errors', type=bool, default=False, help='Whether to also plot absolute errors.')
    parser.add_argument('--schedule', type=str, default='variable', help='Training schedule: variable or fixed.')
#     parser.add_argument('--schedule', type='+', default=['variable', 'basic'], help='Training schedules or statistics to plot: variable=SR, fixed=CUSUM, basic=wealth.')

    parser.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to run experiments on.')
    parser.add_argument('--errs_window', type=int, default=50, help='Num observations to average for plotting errors.')
    parser.add_argument('--cs_type', type=str, default='signed', help="Nonconformity score type: 'abs' or 'signed' ")
#     parser.add_argument('--weights_to_compute', type=str, default='fixed_cal', help='Type of weight computation to do.')
    parser.add_argument('--methods', nargs='+', help='Names of methods to try (weight types)', required = True)
    parser.add_argument('--label_shift', type=float, default=0, help="Label shift value.")
    parser.add_argument('--noise_mu', type=float, default=0.0, help="x-dependent noise mean, wine data")
    parser.add_argument('--noise_sigma', type=float, default=0.0, help="x-dependent noise variance, wine data")
    parser.add_argument('--init_phase', type=int, default=500, help="Num test pts that pre-trained density-ratio estimator has access to")
    parser.add_argument('--num_folds', type=int, default=3, help="Num folds for CTMs and WCTMs")
    
    ## Params only used for fixed_cal_dyn method:
    parser.add_argument('--x_ctm_thresh', type=float, default=3.1623, help="Threshold X-test martingale value that triggers adaptation for fixed_cal_dyn when X-CTM value exceeds it. Default: sqrt(10)")
    parser.add_argument('--x_sched_thresh', type=int, default=1000, help="Threshold ratio that triggers adaptation for fixed_cal_dyn when X-CTM value exceeds it.")
    parser.add_argument('--num_test_unshifted', type=int, default=1000, help="Number of test points that are not shifted; ie, num test points prior to the changepoint occuring")
    parser.add_argument('--run_PR_ST', dest='run_PR_ST', action='store_true', help="Whether to run PodkopaevRamdas sequential testing (their algorithm 1) baseline.")
    parser.add_argument('--run_PR_CD', dest='run_PR_CD', action='store_true', help="Whether to run PodkopaevRamdas changepoint detection baseline (runs algorithm 1 many times, can be slow).")
    parser.add_argument('--pr_source_conc_type', type=str, default='betting', help="PodRam Concentration type used for source data.")
    parser.add_argument('--pr_target_conc_type', type=str, default='betting', help="PodRam Concentration type used for target data.")
    parser.add_argument('--pr_eps_tol', type=float, default=0.0, help="PodRam epsilon tolerance.")
    parser.add_argument('--pr_source_delta', type=float, default=1/200, help="PodRam source delta.")
    parser.add_argument('--pr_target_delta', type=float, default=1/200, help="PodRam target delta.")
    parser.add_argument('--pr_st_stop_criterion', type=str, default='fixed_length', help="Stopping criterion for PodRam Algorithm 1 baseline.")
    parser.add_argument('--pr_cd_stop_criterion', type=str, default='fixed_length', help="Stopping criterion for PodRam changepoint detection baseline.")
#     parser.add_argument('--init_ctm_on_cal_set', type=bool, default=True, help="Whether to initialize conformal martingales on the calibration set (as in Vovk et al); false := initialize at deployment time instead for comparison with Ramdas")
    parser.add_argument('--init_on_cal', dest='init_ctm_on_cal_set', action='store_true',
                    help='Set the init_ctm_on_cal_set flag value to True.')
    parser.add_argument('--init_on_test', dest='init_ctm_on_cal_set', action='store_false',
                    help='Set the init_ctm_on_cal_set flag value to False.')
    parser.set_defaults(init_ctm_on_cal_set=True, run_PR_ST=False, run_PR_CD=False)
   

    ## python main.py dataset muh_fun_name bias
    ## python main.py --dataset0 white_wine --dataset1 red_wine --muh_fun_name NN --d0_shift_type covariate --bias 0.53
    
    args = parser.parse_args()
    dataset0_name = args.dataset0
    dataset1_name = args.dataset1
    muh_fun_name = args.muh_fun_name
    test0_size = args.test0_size
    dataset0_shift_type = args.d0_shift_type
    cov_shift_bias = args.bias
    plot_errors = args.plot_errors
    training_schedule = args.schedule
    n_seeds = args.n_seeds
    errs_window = args.errs_window
    cs_type = args.cs_type
#     weights_to_compute = args.weights_to_compute
    methods = args.methods
    init_phase = args.init_phase
    label_shift = args.label_shift  
    num_folds = args.num_folds
    x_ctm_thresh = args.x_ctm_thresh
    x_sched_thresh = args.x_sched_thresh
    num_test_unshifted = args.num_test_unshifted
    init_ctm_on_cal_set = args.init_ctm_on_cal_set
    run_PR_ST = args.run_PR_ST
    run_PR_CD = args.run_PR_CD
    
    print("init_ctm_on_cal_set from args : ", init_ctm_on_cal_set)
    
    pr_source_conc_type=args.pr_source_conc_type
    pr_target_conc_type=args.pr_target_conc_type
    pr_eps_tol=args.pr_eps_tol
    pr_source_delta=args.pr_source_delta
    pr_target_delta=args.pr_target_delta
    pr_st_stop_criterion=args.pr_st_stop_criterion
    pr_cd_stop_criterion=args.pr_cd_stop_criterion
    
    print("run_PR_ST : ", run_PR_ST)
    print("run_PR_CD : ", run_PR_CD)
    
    ## Load datasets into dataframes
    dataset0 = eval(f'get_{dataset0_name}_data()')
    if (dataset1_name is not None):
        dataset1 = eval(f'get_{dataset1_name}_data()')
    else:
        dataset1 = None
    
    paths_dict_all = {}
    PR_ST_paths_dict_all = {}
    PR_CD_paths_dict_all = {}
    
    for method in methods:
        paths_dict_all[method] = pd.DataFrame()
        
        if run_PR_ST:
            PR_ST_paths_dict_all['PR_ST_cp_'+method] = pd.DataFrame()
        if run_PR_CD:
            PR_CD_paths_dict_all['PR_CD_cp_'+method] = pd.DataFrame()
#     paths_all = pd.DataFrame()
    
    methods_all = "_".join(methods)
    setting = '{}-{}-{}-shift_bias{}-label_shift{}-err_win{}-cs_type{}-nseeds{}-W{}-numTestUnshifted{}-test0Size{}-initCTMcal{}'.format(
        dataset0_name,
        muh_fun_name,
        dataset0_shift_type,
        cov_shift_bias,
        label_shift,
        errs_window,
        cs_type,
        n_seeds,
        methods_all,
        num_test_unshifted,
        test0_size,
        init_ctm_on_cal_set
    )
    
    if run_PR_ST:
        pod_ram_setting = 'sConc{}-tConc{}-eTol{}-sDelta{}-tDelta{}-ST{}-CD{}'.format(
            pr_source_conc_type, 
            pr_target_conc_type,
            pr_eps_tol, 
            pr_source_delta, 
            pr_target_delta,
            pr_st_stop_criterion,
            pr_cd_stop_criterion
        )
    
    print(f'Running with setting: {setting}...\n')
    
    for seed in tqdm(range(0, n_seeds)):
        # training_schedule = ['variable', 'fix']
        paths_dict_curr, PR_ST_paths_dict_curr, PR_CD_paths_dict_curr = training_function(
            dataset0, 
            dataset0_name, 
            dataset1, 
            training_schedule=training_schedule, 
            muh_fun_name=muh_fun_name, 
            test0_size = test0_size, 
            dataset0_shift_type=dataset0_shift_type, 
            cov_shift_bias=cov_shift_bias, 
            plot_errors=plot_errors, 
            seed=seed, 
            cs_type=cs_type, 
            label_uptick=label_shift,
            verbose=args.verbose,
            noise_mu=args.noise_mu,
            noise_sigma=args.noise_sigma,
            methods=methods,
            depth=args.depth,
            init_phase=init_phase,
            num_folds=num_folds,
            x_ctm_thresh=x_ctm_thresh,
            x_sched_thresh=x_sched_thresh,
            num_test_unshifted=num_test_unshifted,
            run_PR_ST=run_PR_ST,
            run_PR_CD=run_PR_CD,
            pr_source_conc_type=pr_source_conc_type,
            pr_target_conc_type=pr_target_conc_type,
            pr_eps_tol=pr_eps_tol,
            pr_source_delta=pr_source_delta,
            pr_target_delta=pr_target_delta,
            pr_st_stop_criterion=pr_st_stop_criterion,
            pr_cd_stop_criterion=pr_cd_stop_criterion,
            init_ctm_on_cal_set=init_ctm_on_cal_set
        )
        for method in methods:
            paths_dict_all[method] = pd.concat([paths_dict_all[method], paths_dict_curr[method]], ignore_index=True)
            
            if run_PR_ST:
                PR_ST_paths_dict_all['PR_ST_cp_'+method] = pd.concat([PR_ST_paths_dict_all['PR_ST_cp_'+method], \
                                                                      PR_ST_paths_dict_curr['PR_ST_cp_'+method]],\
                                                                     ignore_index=True)
            if run_PR_CD:
                PR_CD_paths_dict_all['PR_CD_cp_'+method] = pd.concat([PR_CD_paths_dict_all['PR_CD_cp_'+method], \
                                                                      PR_CD_paths_dict_curr['PR_CD_cp_'+method]],\
                                                                     ignore_index=True)
        
    ## Save all results together
    results_all = paths_dict_all[methods[0]]
    results_all['method'] = methods[0]
    
    if run_PR_ST:
        PR_ST_results_all = PR_ST_paths_dict_all['PR_ST_cp_'+methods[0]]
        PR_ST_results_all['method'] = 'PR_ST_cp_'+methods[0]
    if run_PR_CD:
        PR_CD_results_all = PR_CD_paths_dict_all['PR_CD_cp_'+methods[0]]
        PR_CD_results_all['method'] = 'PR_CD_cp_'+methods[0]
    
    for method in methods[1:]:
        paths_dict_all[method]['method'] = method
        results_all = pd.concat([results_all, paths_dict_all[method]], ignore_index=True)
        
        if run_PR_ST:
            PR_ST_paths_dict_all['PR_ST_cp_'+method]['method'] = 'PR_ST_cp_'+method
            PR_ST_results_all = pd.concat([PR_ST_results_all, PR_ST_paths_dict_all['PR_ST_cp_'+method]], ignore_index=True)
        if run_PR_CD:
            PR_CD_paths_dict_all['PR_CD_cp_'+method]['method'] = 'PR_CD_cp_'+method
            PR_CD_results_all = pd.concat([PR_CD_results_all, PR_CD_paths_dict_all['PR_CD_cp_'+method]], ignore_index=True)
        
        
    results_all.to_csv(f'../results/{setting}.csv')
    
    if run_PR_ST:
        PR_ST_results_all.to_csv(f'../results/{setting}_PR_ST-{pod_ram_setting}.csv')
    if run_PR_CD:
        PR_CD_results_all.to_csv(f'../results/{setting}_PR_CD-{pod_ram_setting}.csv')
    
    
    ## Preparation for plotting
    sigmas_0_means_dict, sigmas_1_means_dict = {}, {}
    sigmas_0_stderr_dict, sigmas_1_stderr_dict = {}, {}
    martingales_0_means_dict, martingales_1_means_dict = {}, {}
    martingales_0_stderr_dict, martingales_1_stderr_dict = {}, {}
    errors_0_means_dict, errors_1_means_dict = {}, {}
    errors_0_stderr_dict, errors_1_stderr_dict = {}, {}
    coverage_0_means_dict = {}
    coverage_0_stderr_dict = {}
    widths_0_medians_dict = {}
    widths_0_lower_q_dict = {}
    widths_0_upper_q_dict = {}
    pvals_0_means_dict = {}
    pvals_0_stderr_dict = {}
    p_vals_pre_change_dict = {}
    p_vals_post_change_dict = {}
    
    
    if (init_ctm_on_cal_set):
        ## Changepoint index is calibration set size + num_test_unshifted
        change_point_index = len(dataset0)*(1-test0_size)/max(2,num_folds)+num_test_unshifted
    else:
        change_point_index = num_test_unshifted
    
    for method in methods:
    
        ## Compute average and stderr values for plotting
        paths_all = paths_dict_all[method]
        num_obs = paths_all['obs_idx'].max() + 1

        sigmas_0_means, sigmas_1_means = [], []
        sigmas_0_stderr, sigmas_1_stderr = [], []
        martingales_0_means, martingales_1_means = [], []
        martingales_0_stderr, martingales_1_stderr = [], []
        errors_0_means, errors_1_means = [], []
        errors_0_stderr, errors_1_stderr = [], []
        coverage_0_means = []
        coverage_0_stderr = []
        widths_0_medians = []
        widths_0_lower_q = []
        widths_0_upper_q = []
        pvals_0_means = []
        pvals_0_stderr = []

        ## For each fold/separate martingale path
        for i in range(0, num_folds):
            ## Compute average martingale values over trials
            sigmas_0_means.append(paths_all[['sigmas_0_'+str(i), 'obs_idx']].groupby('obs_idx').mean())
            sigmas_0_stderr.append(paths_all[['sigmas_0_'+str(i), 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))
            
            martingales_0_means.append(paths_all[['martingales_0_'+str(i), 'obs_idx']].groupby('obs_idx').mean())
            martingales_0_stderr.append(paths_all[['martingales_0_'+str(i), 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))


            ## Compute average and stderr absolute score (residual) values over window, trials
            errors_0_means_fold = []
            errors_0_stderr_fold = []
            coverage_0_means_fold = []
            coverage_0_stderr_fold = []
            widths_0_medians_fold = []
            widths_0_lower_q_fold = []
            widths_0_upper_q_fold = []
            pvals_0_means_fold = []
            pvals_0_stderr_fold = []
            for j in range(0, int(num_obs / errs_window)):
                ## Subset dataframe by window
                paths_all_sub = paths_all[paths_all['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]

                ## Averages and stderrs for that window
                errors_0_means_fold.append(paths_all_sub['errors_0_'+str(i)].mean())
                errors_0_stderr_fold.append(paths_all_sub['errors_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))

                ## Coverages for window
                coverage_0_means_fold.append(paths_all_sub['coverage_0_'+str(i)].mean())
                coverage_0_stderr_fold.append(paths_all_sub['coverage_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))
                
                ## Widths for window
                wid_med = paths_all_sub['widths_0_'+str(i)].median()
                widths_0_medians_fold.append(wid_med)
                widths_0_lower_q_fold.append(paths_all_sub['widths_0_'+str(i)].quantile(0.25))
                widths_0_upper_q_fold.append(paths_all_sub['widths_0_'+str(i)].quantile(0.75))

                ## P values for window
                pvals_0_means_fold.append(paths_all_sub['pvals_0_'+str(i)].mean())
                pvals_0_stderr_fold.append(paths_all_sub['pvals_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))


            ## Averages and stderrs for that fold
            errors_0_means.append(errors_0_means_fold)
            errors_0_stderr.append(errors_0_stderr_fold)

            ## Average coverages for fold
            coverage_0_means.append(coverage_0_means_fold)
            coverage_0_stderr.append(coverage_0_stderr_fold)
            
            ## Median widths for fold
            widths_0_medians.append(widths_0_medians_fold)
            widths_0_lower_q.append(widths_0_lower_q_fold)
            widths_0_upper_q.append(widths_0_upper_q_fold)

            ## Average pvals for fold
            pvals_0_means.append(pvals_0_means_fold)
            pvals_0_stderr.append(pvals_0_stderr_fold)

            
        
            if (dataset1 is not None):
                ## Compute average martingale values over trials
                sigmas_1_means.append(paths_all_abs[['sigmas_1_'+str(i), 'obs_idx']].groupby('obs_idx').mean())
                sigmas_1_stderr.append(paths_all[['sigmas_1_'+str(i), 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))
                
                martingales_1_means.append(paths_all[['martingales_1_'+str(i), 'obs_idx']].groupby('obs_idx').mean())
                martingales_1_stderr.append(paths_all[['martingales_1_'+str(i), 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))

                ## Compute average and stderr absolute score (residual) values over window, trials
                errors_1_means_fold = []
                errors_1_stderr_fold = []
                for j in range(0, int(num_obs/errs_window)):
                    ## Subset dataframe by window
                    paths_all_sub = paths_all_abs[paths_all_abs['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]

                    ## Averages and stderrs for that window
                    errors_1_means_fold.append(paths_all_sub['errors_1_'+str(i)].mean())
                    errors_1_stderr_fold.append(paths_all_sub['errors_1_'+str(i)].std()/ np.sqrt(n_seeds*errs_window))

                ## Averages and stderrs for that fold
                errors_1_means.append(errors_1_means_fold)
                errors_1_stderr.append(errors_1_stderr_fold)
                
                
        sigmas_0_means_dict[method], sigmas_1_means_dict[method] = sigmas_0_means, sigmas_1_means
        sigmas_0_stderr_dict[method], sigmas_1_stderr_dict[method] = sigmas_0_stderr, sigmas_1_stderr
        martingales_0_means_dict[method], martingales_1_means_dict[method] = martingales_0_means, martingales_1_means
        martingales_0_stderr_dict[method], martingales_1_stderr_dict[method] = martingales_0_stderr, martingales_1_stderr
        errors_0_means_dict[method], errors_1_means_dict[method] = errors_0_means, errors_1_means
        errors_0_stderr_dict[method], errors_1_stderr_dict[method] = errors_0_stderr, errors_1_stderr
        coverage_0_means_dict[method] = coverage_0_means
        coverage_0_stderr_dict[method] = coverage_0_stderr
        pvals_0_means_dict[method] = pvals_0_means
        pvals_0_stderr_dict[method] = pvals_0_stderr
        widths_0_medians_dict[method] = widths_0_medians
        widths_0_lower_q_dict[method] = widths_0_lower_q
        widths_0_upper_q_dict[method] = widths_0_upper_q
        
        
        
        ## Saving p-values together for histograms
        paths_pre_change = paths_all[paths_all['obs_idx'] < change_point_index]
        paths_post_change = paths_all[paths_all['obs_idx'] >= change_point_index]
        
        p_vals_pre_change = paths_pre_change['pvals_0_0']
        p_vals_post_change = paths_post_change['pvals_0_0']
        
        for i in range(1, num_folds):
            p_vals_pre_change = np.concatenate((p_vals_pre_change, paths_pre_change[f'pvals_0_{i}']))
            p_vals_post_change = np.concatenate((p_vals_post_change, paths_post_change[f'pvals_0_{i}']))
            
        p_vals_pre_change_dict[method] = p_vals_pre_change
        p_vals_post_change_dict[method] = p_vals_post_change
    
        
    plot_martingale_paths(
        dataset0_paths_dict=sigmas_0_means_dict,
        dataset0_paths_stderr_dict=sigmas_0_stderr_dict,
        dataset0_name=dataset0_name,
        martingales_0_dict=martingales_0_means_dict,
        martingales_0_stderr_dict=martingales_0_stderr_dict,
        dataset1_paths_dict=sigmas_1_means_dict, 
        dataset1_paths_stderr_dict=sigmas_1_stderr_dict,
        dataset1_name=dataset1_name,
        martingales_1_dict=martingales_1_means_dict,
        martingales_1_stderr_dict=martingales_1_stderr_dict,
        errors_0_means_dict=errors_0_means_dict,
        errors_1_means_dict=errors_1_means_dict,
        errors_0_stderr_dict=errors_0_stderr_dict,
        errors_1_stderr_dict=errors_1_stderr_dict,
        p_vals_pre_change_dict=p_vals_pre_change_dict,
        p_vals_post_change_dict=p_vals_post_change_dict,
        errs_window=errs_window,
        change_point_index=change_point_index,
        title="Average paths of Shiryaev-Roberts Procedure",
        ylabel="Shiryaev-Roberts Statistics",
        martingale=["Shiryaev-Roberts", "martingale"],
        dataset0_shift_type=dataset0_shift_type,
        cov_shift_bias=cov_shift_bias,
        label_shift_bias=label_shift,
        noise_mu=args.noise_mu,
        noise_sigma=args.noise_sigma,
        plot_errors=plot_errors,
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting,
        coverage_0_means_dict=coverage_0_means_dict,
        coverage_0_stderr_dict=coverage_0_stderr_dict,
        pvals_0_means_dict=pvals_0_means_dict,
        pvals_0_stderr_dict=pvals_0_stderr_dict,
        widths_0_medians_dict=widths_0_medians_dict,
        widths_0_lower_q_dict=widths_0_lower_q_dict,
        widths_0_upper_q_dict=widths_0_upper_q_dict,
        methods=methods,
        schedule=training_schedule,
        num_test_unshifted=num_test_unshifted
    )
    print('\nProgram done!')