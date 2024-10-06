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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


def train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name='RF', seed=0, cs_type='signed',\
                       weights_to_compute='fixed_cal', dataset0_name='white_wine', cov_shift_bias=0):
    fold_results = []
    cs_0 = []
    cs_1 = []
    W = [] ## Will contain estimated likelihood ratio weights for each fold
    n_cals = [] ## Num cal points in each fold
    
    y_name = dataset0_test_0.columns[-1] ## Outcome must be last column
        
    for i, (train_index, cal_index) in enumerate(folds):
        if i == 2:  # Adjust the last fold to have 1099 in training
            train_index, cal_index = train_index[:-1], cal_index
        X_train, X_cal = X[train_index], X[cal_index]
        y_train, y_cal = y[train_index], y[cal_index]
        
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
            
        model.fit(X_train, y_train)
        
        # Evaluate using the calibration set + test set 0 (Scenario 0)
        X_test_0 = np.concatenate((X_cal, dataset0_test_0.drop(y_name, axis=1).to_numpy()), axis=0)
        y_test_0 = np.concatenate((y_cal, dataset0_test_0[y_name].to_numpy()), axis=0)
        y_pred_0 = model.predict(X_test_0)
        
        #### Computing likelihood ratios (estimated or oracle)
        
        ## Online logistic regression for weight estimation
        W_i = [] ## List of weight est. arrays, each t-th array is length (n+t)
        n_cal = len(X_cal)
        n_cals.append(n_cal)
        
        if (weights_to_compute in ['fixed_cal', 'one_step_est']):
            ## Estimating likelihood ratios for each cal, test point
            ## np.shape(W_i) = (T, n_cal + T)
            W_i = online_lik_ratio_estimates(X_test_0, n_cal, init_phase = 50)
            W.append(W_i)
        
        elif (weights_to_compute == 'one_step_oracle'):
            ## Oracle one-step likelihood ratios
            ## np.shape(W_i) = (n_cal + T, )
            W_i = get_w(x_pca=None, x=X_test_0, dataset=dataset0_name, bias=cov_shift_bias)
            W.append(W_i)
           
        # Evaluate using the calibration set + test set 1 (Scenario 1)
        if (dataset1 is not None):
            X_test_1 = np.concatenate((X_cal, dataset1.drop(y_name, axis=1)), axis=0)
            y_test_1 = np.concatenate((y_cal, dataset1[y_name]), axis=0)
            y_pred_1 = model.predict(X_test_1)
        
        np.set_printoptions(threshold=np.inf)
        
        if (cs_type == 'signed'):
            conformity_scores_0 = y_test_0 - y_pred_0
        elif (cs_type == 'abs'):
            conformity_scores_0 = np.abs(y_test_0 - y_pred_0)
            
        cs_0.append(conformity_scores_0)
        
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
                
    return cs_0, cs_1, W, n_cals


def retrain_count(conformity_score, training_schedule, sr_threshold, cu_confidence, W_i, n_cal, verbose=False, weights_to_compute='fixed_cal'):
    p_values = calculate_p_values(conformity_score)
    
    
    if (weights_to_compute == 'fixed_cal'):
        p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
    
    elif (weights_to_compute == 'one_step_est'):
        ## One step weights, ie depth d=1 weights
        p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
        
    elif (weights_to_compute == 'one_step_oracle'):
        ## One step weights, ie depth d=1 weights
        p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
    
    retrain_m, martingale_value = simple_jumper_martingale(p_values, verbose=verbose)

    if training_schedule == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence, verbose)
    
    return retrain_m, retrain_s, martingale_value, sigma


def training_function(dataset0, dataset0_name, dataset1=None, training_schedule='variable', \
                      sr_threshold=1e6, cu_confidence=0.99, muh_fun_name='RF', test0_size=1599/4898, \
                      dataset0_shift_type='none', cov_shift_bias=1.0, plot_errors=False, seed=0, cs_type='signed', \
                        label_uptick=1, verbose=False, noise_mu=0, noise_sigma=0, weights_to_compute='fixed_cal'):
    
    
    
    dataset0_train, dataset0_test_0 = split_and_shift_dataset0(dataset0, dataset0_name, test0_size=test0_size, \
                                                               dataset0_shift_type=dataset0_shift_type, \
                                                               cov_shift_bias=cov_shift_bias, seed=seed, \
                                                               label_uptick=label_uptick, noise_mu=noise_mu,\
                                                                noise_sigma=noise_sigma)
    X, y, folds = split_into_folds(dataset0_train, seed=seed)

    cs_0, cs_1, W, n_cals = train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name, seed=seed, cs_type=cs_type, weights_to_compute=weights_to_compute, dataset0_name=dataset0_name, cov_shift_bias=cov_shift_bias)

    fold_martingales_0, fold_martingales_1 = [], []
    sigmas_0, sigmas_1 = [], []
    retrain_m_count_0, retrain_s_count_0 = 0, 0
    retrain_m_count_1, retrain_s_count_1 = 0, 0
    
    
    for i, score_0 in enumerate(cs_0):
        if (weights_to_compute in ['fixed_cal', 'one_step_est', 'one_step_oracle']):
            m_0, s_0, martingale_value_0, sigma_0 = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence, W[i], n_cals[i], verbose, weights_to_compute)
        else:
            m_0, s_0, martingale_value_0, sigma_0 = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence, None, None, verbose, weights_to_compute)

        if m_0:
            retrain_m_count_0 += 1
        if s_0:
            retrain_s_count_0 += 1
        fold_martingales_0.append(martingale_value_0)
        sigmas_0.append(sigma_0)
        
        
    for i, score_1 in enumerate(cs_1):
        m_1, s_1, martingale_value_1, sigma_1 = retrain_count(score_1, training_schedule, sr_threshold, cu_confidence, W[i], n_cals[i], verbose, weights_to_compute)

        if m_1:
            retrain_m_count_1 += 1
        if s_1:
            retrain_s_count_1 += 1
        fold_martingales_1.append(martingale_value_1)
        sigmas_1.append(sigma_1)
    
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
        
    min_len = np.min([len(sigmas_0[i]) for i in range(0, len(sigmas_0))])
    
    paths = pd.DataFrame(np.c_[np.repeat(seed, min_len), np.arange(0, min_len)], columns = ['itrial', 'obs_idx'])
    for k in range(0, len(sigmas_0)):
        paths['sigmas_0_'+str(k)] = sigmas_0[k][0:min_len]
        paths['cs_0_'+str(k)] = cs_0[k][0:min_len]
    for k in range(0, len(sigmas_1)):
        paths['sigmas_1_'+str(k)] = sigmas_1[k][0:min_len]
        paths['cs_1_'+str(k)] = cs_1[k][0:min_len]
    
    return paths

        
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
    parser.add_argument('--bias', type=float, default=0.0, help='Scalar bias magnitude parameter lmbda for exponential tilting covariate shift.')
    parser.add_argument('--plot_errors', type=bool, default=False, help='Whether to also plot absolute errors.')
    parser.add_argument('--schedule', type=str, default='variable', help='Training schedule: variable or fixed.')
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to run experiments on.')
    parser.add_argument('--errs_window', type=int, default=50, help='Num observations to average for plotting errors.')
    parser.add_argument('--cs_type', type=str, default='signed', help="Nonconformity score type: 'abs' or 'signed' ")
    parser.add_argument('--weights_to_compute', type=str, default='fixed_cal', help='Type of weight computation to do.')
    parser.add_argument('--label_shift', type=float, default=1, help="Label shift value.")
    parser.add_argument('--noise_mu', type=float, default=0.2, help="x-dependent noise mean, wine data")
    parser.add_argument('--noise_sigma', type=float, default=0.05, help="x-dependent noise variance, wine data")
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
    weights_to_compute = args.weights_to_compute
    print("cov_shift_bias: ", cov_shift_bias)
    
    label_shift = args.label_shift    
    
    ## Load datasets into dataframes
    dataset0 = eval(f'get_{dataset0_name}_data()')
    if (dataset1_name is not None):
        dataset1 = eval(f'get_{dataset1_name}_data()')
    else:
        dataset1 = None
    
    paths_all = pd.DataFrame()
    print(f'Running {n_seeds} random experiments...\n')
    for seed in tqdm(range(0, n_seeds)):
        # training_schedule = ['variable', 'fix']

        paths_curr = training_function(
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
            weights_to_compute=weights_to_compute
        )
        paths_all = pd.concat([paths_all, paths_curr], ignore_index=True)
        
    setting = '{}-{}-{}-shift_bias{}-label_shift{}-err_win{}-cs_type{}-nseeds{}-W{}'.format(
        dataset0_name,
        muh_fun_name,
        dataset0_shift_type,
        cov_shift_bias,
        label_shift,
        errs_window,
        cs_type,
        n_seeds,
        weights_to_compute
    )
    paths_all.to_csv(f'../results/' + setting + '.csv')
    
    ## Compute average and stderr values for plotting
    paths_all_abs = paths_all.abs()
    num_obs = paths_all_abs['obs_idx'].max() + 1
    
    sigmas_0_means = []
    sigmas_1_means = []
    sigmas_0_stderr = []
    sigmas_1_stderr = []
    cs_abs_0_means = []
    cs_abs_1_means = []
    cs_abs_0_stderr = []
    cs_abs_1_stderr = []
    
    ## For each fold/separate martingale path
    for i in range(0, 3):
        ## Compute average martingale values over trials
        sigmas_0_means.append(paths_all_abs[['sigmas_0_'+str(i), 'obs_idx']].groupby('obs_idx').mean())
        
        ## Compute average and stderr absolute score (residual) values over window, trials
        cs_abs_0_means_fold = []
        cs_abs_0_stderr_fold = []
        for j in range(0, int(num_obs / errs_window)):
            ## Subset dataframe by window
            paths_all_abs_sub = paths_all_abs[paths_all_abs['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]
            
            ## Averages and stderrs for that window
            cs_abs_0_means_fold.append(paths_all_abs_sub['cs_0_'+str(i)].mean())
            cs_abs_0_stderr_fold.append(paths_all_abs_sub['cs_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))
        
        ## Averages and stderrs for that fold
        cs_abs_0_means.append(cs_abs_0_means_fold)
        cs_abs_0_stderr.append(cs_abs_0_stderr_fold)
        
    if (dataset1 is not None):
        for i in range(0, 3):
            ## Compute average martingale values over trials
            sigmas_1_means.append(paths_all_abs[['sigmas_1_'+str(i), 'obs_idx']].groupby('obs_idx').mean())

            ## Compute average and stderr absolute score (residual) values over window, trials
            cs_abs_1_means_fold = []
            cs_abs_1_stderr_fold = []
            for j in range(0, int(num_obs/errs_window)):
                ## Subset dataframe by window
                paths_all_abs_sub = paths_all_abs[paths_all_abs['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]

                ## Averages and stderrs for that window
                cs_abs_1_means_fold.append(paths_all_abs_sub['cs_1_'+str(i)].mean())
                cs_abs_1_stderr_fold.append(paths_all_abs_sub['cs_1_'+str(i)].std()/ np.sqrt(n_seeds*errs_window))

            ## Averages and stderrs for that fold
            cs_abs_1_means.append(cs_abs_1_means_fold)
            cs_abs_1_stderr.append(cs_abs_1_stderr_fold)
        
    plot_martingale_paths(
        dataset0_paths=sigmas_0_means,
        dataset0_name=dataset0_name,
        dataset1_paths=sigmas_1_means, 
        dataset1_name=dataset1_name,
        cs_abs_0_means=cs_abs_0_means,
        cs_abs_1_means=cs_abs_1_means,
        cs_abs_0_stderr=cs_abs_0_stderr,
        cs_abs_1_stderr=cs_abs_1_stderr,
        errs_window=errs_window,
        change_point_index=len(dataset0)*(1-test0_size)/3,
        title="Average paths of Shiryaev-Roberts Procedure",
        ylabel="Shiryaev-Roberts Statistics",
        martingale="shiryaev_roberts",
        dataset0_shift_type=dataset0_shift_type,
        cov_shift_bias=cov_shift_bias,
        label_shift_bias=label_shift,
        noise_mu=args.noise_mu,
        noise_sigma=args.noise_sigma,
        plot_errors=plot_errors,
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting
    )
    print('\nProgram done!')