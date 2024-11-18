import numpy as np
import pandas as pd
from plot import plot_martingale_paths
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import pdb

## Drew added
from utils import *
import argparse
import os
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import math


def get_white_wine_data():
    white_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
    return white_wine

def get_white_wine_pca_data():
    white_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
    return white_wine

def get_red_wine_data():
    red_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    return red_wine

def get_airfoil_data():
    airfoil = pd.read_csv(os.getcwd() + '/../datasets/airfoil/airfoil.txt', sep = '\t', header=None)
    airfoil.columns = ["Frequency","Angle","Chord","Velocity","Suction","Sound"]
    airfoil.iloc[:,0] = np.log(airfoil.iloc[:,0])
    airfoil.iloc[:,4] = np.log(airfoil.iloc[:,4])
    return airfoil

def get_airfoil_pca_data():
    return get_airfoil_data()


# def get_1dim_synthetic_data(size=5000):
#     X = np.random.uniform(low=1, high=10, size=size)
#     epsilon1 = np.random.normal(size=size)
#     epsilon2 = np.random.normal(size=size)
#     U = np.random.uniform(size=size)
#     indicator_U = U < 0.01
#     Y = np.zeros(size)
#     for i in range(0, size):
#         Y[i] = np.random.normal(np.sin(X[i]), np.cos(5*X[i])+5) 
#     return pd.DataFrame(np.c_[X, Y])

# def get_1dim_synthetic_data(size=5000):
#     high=10
#     X = np.random.uniform(low=1, high=high, size=size)
#     epsilon1 = np.random.normal(size=size)
#     epsilon2 = np.random.normal(size=size)
#     U = np.random.uniform(size=size)
#     indicator_U = U < 0.1
# #     indicator_X = X < 1
#     Y = np.zeros(size)
#     for i in range(0, size):
# #         Y[i] = np.random.normal(2*X[i]*(np.sin(X[i])+1), X[i]+0.5) + (indicator_X[i] * indicator_U[i]) * 25
# #         Y[i] = np.random.normal(2*X[i]*(np.sin(X[i])+1), X[i]+0.5) + indicator_U[i] * 10
#         Y[i] = np.random.normal(2*X[i]*(np.sin(X[i])+1), X[i]+0.5)
# #         Y[i] = np.random.normal(X[i]*(np.sin(X[i])+1), 2*np.sin(X[i]/3-np.pi/2)+2.5)
# #         Y[i] = np.random.normal(X[i]*(np.sin(X[i])+1), 2*np.sin(X[i]/2-9/8*np.pi)+2.5)
# #         Y[i] = np.random.normal(X[i]*(np.sin(X[i])+1), 2*np.sin(X[i]/2+np.pi/2)+2.5)
# #         Y[i] = np.random.normal(X[i]*(3*np.sin(X[i]**2 / 10)+1), 2*np.sin(X[i]/2+np.pi/2)+2.5)
#     return pd.DataFrame(np.c_[X, Y])


# def get_1dim_linear_synthetic_data(size=5000):
#     high=2*np.pi
#     X = np.random.uniform(low=0, high=high, size=size)
#     Y = np.zeros(size)
#     for i in range(0, size):
#         Y[i] = np.random.normal(2, X[i]+0.5)

#     return pd.DataFrame(np.c_[X, Y])

def get_1dim_synthetic_data(size=10000):
    high=2*np.pi
    X = np.random.uniform(low=0, high=high, size=size)
    Y = np.zeros(size)
    for i in range(0, size):
        Y[i] = np.random.normal(np.sin(X[i]), (X[i]+1)/10)

    return pd.DataFrame(np.c_[X, Y])



def get_1dim_synthetic_v2_data(size=10000):
    high=2*np.pi
    X = np.random.uniform(low=-np.pi/2, high=high, size=size)
    Y = np.zeros(size)
    for i in range(0, size):
        if (X[i] >= 0):
            Y[i] = np.random.normal(np.sin(X[i]), np.abs(X[i]+1)/10)
        else:
            Y[i] = np.random.normal(-3*np.sin(X[i]**3), np.abs(X[i])/10)

    return pd.DataFrame(np.c_[X, Y])


def split_and_shift_dataset0(
    dataset0, 
    dataset0_name, 
    test0_size, 
    dataset0_shift_type='none', 
    cov_shift_bias=1.0, 
    label_uptick=1,
    seed=0,
    noise_mu=0,
    noise_sigma=0
):
    
    dataset0_train, dataset0_test_0 = train_test_split(dataset0, test_size=test0_size, shuffle=True, random_state=seed)

    if (dataset0_shift_type == 'none'):
        ## No shift within dataset0    
        return dataset0_train, dataset0_test_0
    
    elif (dataset0_shift_type == 'covariate'):
        ## Covariate shift within dataset0
        dataset0_test_0 = dataset0_test_0.reset_index(drop=True)
        
        dataset0_train_copy = dataset0_train.copy()
        
        X_train = dataset0_train_copy.iloc[:, :-1].values
        print("np.shape(X_train) : ", np.shape(X_train))
        dataset0_test_0_copy = dataset0_test_0.copy()
        X_test_0 = dataset0_test_0_copy.iloc[:, :-1].values
        
        dataset0_test_0_biased_idx = exponential_tilting_indices(x_pca=X_train, x=X_test_0, dataset=dataset0_name, bias=cov_shift_bias)
#         print(dataset0_test_0_biased_idx)
#         print("unique proportion :", len(np.unique(dataset0_test_0_biased_idx)) / len(dataset0_test_0_biased_idx))
        
        return dataset0_train, dataset0_test_0.iloc[dataset0_test_0_biased_idx]
    
    elif (dataset0_shift_type == 'label'):
        ## Label shift within dataset0

        if 'wine' in dataset0_name:
            # Define a threshold for 'alcohol' to identify high alcohol content wines
            alcohol_threshold = dataset0_test_0['alcohol'].median()
            # Increase the quality score by a number for wines with alcohol above the threshold
            dataset0_test_0.loc[dataset0_test_0['alcohol'] > alcohol_threshold, 'quality'] += label_uptick
            dataset0_test_0['quality'] = dataset0_test_0['quality'].clip(lower=0, upper=10)

        return dataset0_train, dataset0_test_0

    elif (dataset0_shift_type == 'noise'):
        ## X-dependent noise within dataset0
        data_before_shift = dataset0_test_0.copy()
        if 'wine' in dataset0_name:
            dataset0_test_0['sulphates'] += np.where(
            dataset0_test_0['quality'] >= dataset0_test_0['quality'].median(),
            # Add positive noise for higher quality wines
            np.random.normal(loc=noise_mu, scale=noise_sigma, size=len(dataset0_test_0)),
            # Subtract noise for lower quality wines
            np.random.normal(loc=-noise_mu, scale=noise_sigma, size=len(dataset0_test_0)))
            # Ensure 'sulphates' remains within valid range
            dataset0_test_0['sulphates'] = dataset0_test_0['sulphates'].clip(lower=data_before_shift['sulphates'].min(), upper=data_before_shift['sulphates'].max())
        return dataset0_train, dataset0_test_0
        

def split_into_folds(dataset0_train, seed=0):
    y_name = dataset0_train.columns[-1] ## Outcome column must be last in dataframe
    X = dataset0_train.drop(y_name, axis=1).to_numpy()
    y = dataset0_train[y_name].to_numpy()
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(kf.split(X, y))
    return X, y, folds


def online_lik_ratio_estimates(X_test_0, n_cal, init_phase = 50):
    
    T = len(X_test_0) - n_cal ## Number of test points (points after true changepoint)
    
    W_i = np.zeros((T, (n_cal + T)))
    
    ## Initialize density ratio estimation with 'init_phase' number of first test points
    X_test_0_curr = X_test_0[0:(n_cal+init_phase)] 
    class_labels = np.concatenate((np.zeros(n_cal), np.ones(init_phase)), axis=0)
    clf = SGDClassifier(random_state=0, loss='log_loss', alpha=0.1, max_iter=1000)
    clf.fit(X_test_0_curr, class_labels)

    for t in range(1, T+1):
        
        if (t <= init_phase):
            ## In the initial phase, assigning uniform weight to all points
            W_i[t-1] = np.ones(n_cal+T)

        else:
            ## 
            X_test_0_curr = X_test_0[0:(n_cal+t)] ## X_{1:(n+t)} cal and test points
            class_labels = np.concatenate((np.zeros(n_cal), np.ones(t)), axis=0)
            clf.partial_fit(X_test_0_curr[-1].reshape(1, -1), np.array([class_labels[-1]]))
            lr_probs = clf.predict_proba(X_test_0)
            W_i[t-1] = lr_probs[:,1] / lr_probs[:,0] ## p_test / p_train

    return W_i





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
        
        elif (weights_to_compute in ['fixed_cal_oracle','one_step_oracle', 'batch_oracle', 'multistep_oracle']):
#             print("getting oracle lik ratios")
            ## Oracle one-step likelihood ratios
            ## np.shape(W_i) = (n_cal + T, )
            W_i = get_w(x_pca=X_train, x=X_test_0, dataset=dataset0_name, bias=cov_shift_bias) 
#             print(np.mean(W_i[n_cal:]) / np.mean(W_i[:n_cal]))
#             print(W_i)
            
            if (weights_to_compute == 'batch_oracle'):
                W_i = (W_i - min(W_i)) / (max(W_i) - min(W_i))
                W_i = subsample_batch_weights(W_i, n_cal, max_num_samples=100)
                
                
            if (weights_to_compute == 'multistep_oracle'):
                W_i = np.tile(W_i, (len(X_test_0) - n_cal, 1))
            
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


def calculate_p_values(conformity_scores):
    """
    Calculate the conformal p-values from conformity scores.
    """
    
    n = len(conformity_scores)
    p_values = np.array([(np.sum(conformity_scores[:i] < conformity_scores[i]) + 
                         np.random.uniform() * np.sum(conformity_scores[:i] == conformity_scores[i])) / (i + 1)
                         for i in range(n)])
    return p_values


## Drew added
## Note: This is for calculating the weighted p-values once the normalized weights have already been calculated
def calculate_weighted_p_values(conformity_scores, W_i, n_cal, weights_to_compute='fixed_cal'):
    """
    Calculate the weighted conformal p-values from conformity scores and given normalized weights 
    (i.e., enforce np.sum(normalized_weights) = 1).
    
    W_i : List of likelihood ratio weight est. arrays, each t-th array is length (n_cal+t)
    """
    init_phase = 100
    wp_values = np.zeros(len(conformity_scores))
    
    
    ## p-values for original calibration set calculated as before
    wp_values[0:(n_cal+init_phase)] = calculate_p_values(conformity_scores[0:(n_cal+init_phase)]) 
        
    if (weights_to_compute in ['fixed_cal', 'fixed_cal_oracle']):
        
        
        T = len(conformity_scores) - n_cal ## Number of total test observations
        idx_include = np.concatenate((np.repeat(True, n_cal), np.repeat(False, T)), axis=0) ## indicies to include
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## idx_include implements indices for 'fixed cal' ie, comparing to [0:n_cal] \cup n_cal + t_ 
            idx_include[n_cal+t_] = True ## Move to curr test point
            if (t_ > 0):
                idx_include[n_cal+t_-1] = False ## Exclude most recent test point again (except at start, is a cal point)
            
            ## Subset conformity scores and weights based on idx_include
            conformity_scores_t = conformity_scores[idx_include]
            if (weights_to_compute == 'fixed_cal'):
                W_i_t = W_i[t_][idx_include]
            else:
#                 print("computing oracle weighted p vals")
                W_i_t = W_i[idx_include]
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
            ## Calculate weighted p-values
            wp_values[n_cal+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
            

    
    elif (weights_to_compute in ['one_step_oracle', 'one_step_est', 'batch_oracle', 'multistep_oracle']):
        
        print("computing multistep weights")
        T = len(conformity_scores) - n_cal ## Number of total test observations
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## Subset conformity scores and weights based on idx_include
#             print("len conformity_scores : ", len(conformity_scores))
            conformity_scores_t = conformity_scores[:(n_cal+t_+1)]
#             print("conformity_scores_t len :", len(conformity_scores_t))
#             print("n_cal+t_ ", n_cal+t_)
            
            if (weights_to_compute in ['one_step_est', 'batch_oracle']):
                W_i_t = W_i[t_][:(n_cal+t_+1)]
                
            elif (weights_to_compute == 'one_step_oracle'):
                W_i_t = W_i[:(n_cal+t_+1)]
                
            elif (weights_to_compute == 'multistep_oracle'):
#                 print("computing multistep weights")
#                 print("W_i shape : ", np.shape(W_i))
#                 print("n_cal+t_ ", n_cal+t_)
#                 print("W_i[-(t_+2):] shape ", np.shape(W_i[-(t_+2):]))
                w_mat = np.matrix(W_i[-(t_+2):,:(n_cal+t_+1)])
#                 print("w_mat shape : ", np.shape(w_mat))
#                 print("one-step weights : ", w_mat[0] / np.sum(w_mat[0]))
                
                W_i_t = compute_w_ptest_split_active_replacement(w_mat, depth_max=2)
#                 print("two-step weights : ", W_i_t / np.sum(W_i_t))
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
#             print("conformity_scores_t len :", len(conformity_scores_t))
#             print("normalized_weights_t len :", len(normalized_weights_t))
                        
            ## Calculate weighted p-values
            wp_values[n_cal+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
           
                
    
    else:
        raise Exception("Note implemented")
        
    return wp_values




def ville_procedure(p_values, threshold=100, verbose=False):
    """
    Implements the Ville procedure. Raises an alarm when the martingale exceeds the threshold.
    """
    martingale = 1.0  # Start with initial capital of 1
    for i, p in enumerate(p_values):
        # This implies that the martingale grows if the p-value is small (indicating that the observation is unlikely under the null hypothesis)
        # and shrinks if the p-value is large.
        martingale *= (1 / p)
        if martingale >= threshold and verbose:
            print(f"Alarm raised at observation {i + 1} with martingale value = {martingale}")
            # break
    return martingale

def cusum_procedure(S, alpha, verbose=False):
    """
    Implements the CUSUM statistic.
    """
    gamma = np.zeros(len(S))
    threshold = np.percentile(S, 100 * alpha)
    for n in range(1, len(S)):
        gamma[n] = max(S[n] / S[i] for i in range(n))
        if gamma[n] >= threshold and verbose:
            print(f"Alarm raised at observation {n} with gamma={gamma[n]}")
            # return True, gamma
    return False, gamma

def shiryaev_roberts_procedure(S, c, verbose=False):
    """
    Implements the Shiryaev-Roberts statistic.
    """
    sigma = np.zeros(len(S))
    for n in range(1, len(S)):
        sigma[n] = sum(S[n] / S[i] for i in range(n))
        if sigma[n] >= c and verbose:
            print(f"Alarm raised at observation {n} with sigma={sigma[n]}")
            # return True, sigma
    return False, sigma

def simple_jumper_martingale(p_values, J=0.01, threshold=100, verbose=False):
    """
    Implements the Simple Jumper martingale betting strategy.
    """
    C_minus1, C_0, C_1 = 1/3, 1/3, 1/3
    C = 1
    martingale_values = []

    for i, p in enumerate(p_values):
        C_minus1 = (1 - J) * C_minus1 + (J / 3) * C
        C_0 = (1 - J) * C_0 + (J / 3) * C
        C_1 = (1 - J) * C_1 + (J / 3) * C

        C_minus1 *= (1 + (p - 0.5) * -1)
        C_0 *= (1 + (p - 0.5) * 0)
        C_1 *= (1 + (p - 0.5) * 1)

        C = C_minus1 + C_0 + C_1
        martingale_values.append(C)

        if C >= threshold and verbose:
            print(f"Alarm raised at observation {i} with martingale value={C}")
            # return True, np.array(martingale_values)
    
    return False, np.array(martingale_values)


def retrain_count(conformity_score, training_schedule, sr_threshold, cu_confidence, W_i, n_cal, verbose=False, weights_to_compute='fixed_cal'):
    p_values = calculate_p_values(conformity_score)
    
    
    if (weights_to_compute in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle']):
        p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
    
#     elif (weights_to_compute == 'one_step_est'):
#         ## One step weights, ie depth d=1 weights
#         p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
        
#     elif (weights_to_compute == 'one_step_oracle'):
#         ## One step weights, ie depth d=1 weights
#         p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
        
#     elif (weights_to_compute == 'batch_oracle'):
#         p_values = calculate_weighted_p_values(conformity_score, W_i, n_cal, weights_to_compute)
    
    
#     print("len(p_values) ", len(p_values))
#     print("n_cal ", n_cal)

#     print("p_values[:n_cal]   : ", p_values[0:n_cal])
#     print("average cal p-val  : ", np.mean(p_values[0:n_cal]))
    
# #     print("p_values[n_cal:]   : ", p_values[n_cal:])
#     print("average test p-val : ", np.mean(p_values[n_cal:]))
#     fig, ax = plt.subplots(1, 2)
#     ax[0].hist(p_values[0:n_cal]) #row=0, col=0
#     ax[0].set_title('cal p-values')
#     ax[1].hist(p_values[n_cal:]) #row=1, col=0
#     ax[1].set_title('test p-values')
#     fig.savefig(os.getcwd() + '/../figs/p_vals_hist.pdf')
#     print("\n")
    
    retrain_m, martingale_value = simple_jumper_martingale(p_values, verbose=verbose)

    if training_schedule == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
        
    elif (training_schedule == 'basic'):
        print("plotting martingale (wealth) values directly")
#         print("martingale_value shape :", np.shape(martingale_value))
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
#         print("SR shape :", np.shape(sigma))
        sigma = martingale_value
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence, verbose)
        
    
    return retrain_m, retrain_s, martingale_value, sigma, p_values



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
    
    ## Add simulated measurement noise
#     ols = LinearRegression(fit_intercept=False)  # featurization from walsh_hadamard_from_seqs has intercept
#     ols.fit(X, y)
#     y_pred = ols.predict(X)
#     resid = np.abs(y - y_pred)
#     y = y + np.random.normal(0, resid)
#     kernel_ridge = KernelRidge(alpha=0.1)  # featurization from walsh_hadamard_from_seqs has intercept
#     kernel_ridge.fit(X, y)
#     y_pred = kernel_ridge.predict(X)
#     resid = np.abs(y - y_pred)
#     y = y + np.random.normal(0, resid)

    cs_0, cs_1, W, n_cals = train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name, seed=seed, cs_type=cs_type, weights_to_compute=weights_to_compute, dataset0_name=dataset0_name, cov_shift_bias=cov_shift_bias)

    fold_martingales_0, fold_martingales_1 = [], []
    sigmas_0, sigmas_1 = [], []
    retrain_m_count_0, retrain_s_count_0 = 0, 0
    retrain_m_count_1, retrain_s_count_1 = 0, 0
    p_values_0 = []
    coverage_0 = []
        
    for i, score_0 in enumerate(cs_0):
        if (weights_to_compute in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle']):
            m_0, s_0, martingale_value_0, sigma_0, p_vals = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence, W[i], n_cals[i], verbose, weights_to_compute)
        else:
            m_0, s_0, martingale_value_0, sigma_0, p_vals = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence, None, n_cals[i], verbose, weights_to_compute)

        if m_0:
            retrain_m_count_0 += 1
        if s_0:
            retrain_s_count_0 += 1
        fold_martingales_0.append(martingale_value_0)
        sigmas_0.append(sigma_0)
        
        ## Storing p-values
        p_values_0.append(p_vals)
        coverage_0.append(p_vals <= 0.9)
#         p_values_0_test.append(p_vals[n_cals[i]:])
        
        
    
    
    for i, score_1 in enumerate(cs_1):
        m_1, s_1, martingale_value_1, sigma_1 = retrain_count(score_1, training_schedule, sr_threshold, cu_confidence, W[i], n_cals[i], verbose, weights_to_compute)

        if m_1:
            retrain_m_count_1 += 1
        if s_1:
            retrain_s_count_1 += 1
        fold_martingales_1.append(martingale_value_1)
        sigmas_1.append(sigma_1)
    
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
        
    min_len = np.min([len(sigmas_0[i]) for i in range(0, len(sigmas_0))])
    
#     print("p_values_0[k]", len(p_values_0[0]))
#     print(min_len)
    
    paths = pd.DataFrame(np.c_[np.repeat(seed, min_len), np.arange(0, min_len)], columns = ['itrial', 'obs_idx'])
    for k in range(0, len(sigmas_0)):
        paths['sigmas_0_'+str(k)] = sigmas_0[k][0:min_len]
        paths['cs_0_'+str(k)] = cs_0[k][0:min_len]
        paths['pvals_0_'+str(k)] = p_values_0[k][0:min_len]
        paths['coverage_0_'+str(k)] = coverage_0[k][0:min_len]
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
    parser.add_argument('--label_shift', type=int, default=1, help="Label shift value, for wine data it is an integer for label uptick.")
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
    coverage_0_means = []
    coverage_0_stderr = []
    pvals_0_means = []
    pvals_0_stderr = []
    
    ## For each fold/separate martingale path
    for i in range(0, 3):
        ## Compute average martingale values over trials
        sigmas_0_means.append(paths_all_abs[['sigmas_0_'+str(i), 'obs_idx']].groupby('obs_idx').mean())
        
        ## Compute average and stderr absolute score (residual) values over window, trials
        cs_abs_0_means_fold = []
        cs_abs_0_stderr_fold = []
        coverage_0_means_fold = []
        coverage_0_stderr_fold = []
        pvals_0_means_fold = []
        pvals_0_stderr_fold = []
        for j in range(0, int(num_obs / errs_window)):
            ## Subset dataframe by window
            paths_all_abs_sub = paths_all_abs[paths_all_abs['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]
            
            ## Averages and stderrs for that window
            cs_abs_0_means_fold.append(paths_all_abs_sub['cs_0_'+str(i)].mean())
            cs_abs_0_stderr_fold.append(paths_all_abs_sub['cs_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))
            
            ## Coverages for window
            coverage_0_means_fold.append(paths_all_abs_sub['coverage_0_'+str(i)].mean())
            coverage_0_stderr_fold.append(paths_all_abs_sub['coverage_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))
            
            ## P values for window
            pvals_0_means_fold.append(paths_all_abs_sub['pvals_0_'+str(i)].mean())
            pvals_0_stderr_fold.append(paths_all_abs_sub['pvals_0_'+str(i)].std() / np.sqrt(n_seeds*errs_window))
            
        
        ## Averages and stderrs for that fold
        cs_abs_0_means.append(cs_abs_0_means_fold)
        cs_abs_0_stderr.append(cs_abs_0_stderr_fold)
        
        ## Average coverages for fold
        coverage_0_means.append(coverage_0_means_fold)
        coverage_0_stderr.append(coverage_0_stderr_fold)
        
        ## Average pvals for fold
        pvals_0_means.append(pvals_0_means_fold)
        pvals_0_stderr.append(pvals_0_stderr_fold)
        
        
        
    print(np.shape(coverage_0_means[0]))
    print(np.shape(coverage_0_means[0]))
        
    ## Plotting p-values for debugging
    fig, ax = plt.subplots(1, 2)
    changepoint_index = len(dataset0)*(1-test0_size)/3
    paths_cal = paths_all[paths_all['obs_idx'] < changepoint_index]
    paths_test = paths_all[paths_all['obs_idx'] >= changepoint_index]
    
    p_vals_cal = np.concatenate((paths_cal['pvals_0_0'], paths_cal['pvals_0_1'], paths_cal['pvals_0_2']))
    p_vals_test = np.concatenate((paths_test['pvals_0_0'], paths_test['pvals_0_1'], paths_test['pvals_0_2']))

            
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
        p_vals_cal=p_vals_cal,
        p_vals_test=p_vals_test,
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
        setting=setting,
        coverage_0_means=coverage_0_means,
        coverage_0_stderr=coverage_0_stderr,
        weights_to_compute=weights_to_compute
    )
