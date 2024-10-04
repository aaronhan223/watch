import numpy as np
import pandas as pd
from plot import plot_martingale_paths
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import pdb

## Drew added
from utils import *
import argparse
import os


def get_white_wine_data():
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

## Drew edited
def split_and_shift_dataset0(dataset0, dataset0_name, test0_size, dataset0_shift_type='none', cov_shift_bias = 1.0, seed=0):
    
    dataset0_train, dataset0_test_0 = train_test_split(dataset0, test_size=test0_size, shuffle=True, random_state=seed)

    if (dataset0_shift_type == 'none'):
        ## No shift within dataset0    
        return dataset0_train, dataset0_test_0
    
    elif (dataset0_shift_type == 'covariate'):
        ## Covariate shift within dataset0
        dataset0_test_0 = dataset0_test_0.reset_index(drop=True)
        
        dataset0_train_copy = dataset0_train.copy()
        
        X_train = dataset0_train_copy.iloc[:, :-1].values
        dataset0_test_0_copy = dataset0_test_0.copy()
        X_test_0 = dataset0_test_0_copy.iloc[:, :-1].values
        
        
        dataset0_test_0_biased_idx = exponential_tilting_indices(x_pca=X_train, x=X_test_0, dataset=dataset0_name, bias=cov_shift_bias)
        
        return dataset0_train, dataset0_test_0.iloc[dataset0_test_0_biased_idx]
    
    elif (dataset0_shift_type == 'label'):
        ## Label shift within dataset0

        # Define a threshold for 'alcohol' to identify high alcohol content wines
        alcohol_threshold = dataset0_test_0['alcohol'].median()
        # Increase the quality score by 1 for wines with alcohol above the threshold
        dataset0_test_0.loc[dataset0_test_0['alcohol'] > alcohol_threshold, 'quality'] += 1
        dataset0_test_0['quality'] = dataset0_test_0['quality'].clip(lower=0, upper=10)

        return dataset0_train, dataset0_test_0


def split_into_folds(dataset0_train, seed=0):
    y_name = dataset0_train.columns[-1] ## Outcome column must be last in dataframe
    X = dataset0_train.drop(y_name, axis=1).to_numpy()
    y = dataset0_train[y_name].to_numpy()
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(kf.split(X, y))
    return X, y, folds

def train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name='RF', seed=0, cs_type='signed'):
    fold_results = []
    cs_0 = []
    cs_1 = []
    
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
            
    
    return cs_0, cs_1


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
def calculate_weighted_p_values(conformity_scores, normalized_weights):
    """
    Calculate the weighted conformal p-values from conformity scores and given normalized weights 
    (i.e., enforce np.sum(normalized_weights) = 1).
    """
    ## Ensuring that weights are normalized, with slight flexibility to account for float-point issues
    if (np.abs(1 - np.sum(normalized_weights)) > 0.00001):
        raise Exception("normalized_weights must sum to 1")
    
    n = len(conformity_scores)
    p_values = np.array([(np.sum(normalized_weights[conformity_scores[:i] < conformity_scores[i]]) + \
                         np.random.uniform() * np.sum(normalized_weights[conformity_scores[:i] == conformity_scores[i]])) \
                         / (i + 1) for i in range(n)])
    return p_values


def ville_procedure(p_values, threshold=100):
    """
    Implements the Ville procedure. Raises an alarm when the martingale exceeds the threshold.
    """
    martingale = 1.0  # Start with initial capital of 1
    for i, p in enumerate(p_values):
        # This implies that the martingale grows if the p-value is small (indicating that the observation is unlikely under the null hypothesis)
        # and shrinks if the p-value is large.
        martingale *= (1 / p)
        if martingale >= threshold:
            print(f"Alarm raised at observation {i + 1} with martingale value = {martingale}")
            # break
    return martingale

def cusum_procedure(S, alpha):
    """
    Implements the CUSUM statistic.
    """
    gamma = np.zeros(len(S))
    threshold = np.percentile(S, 100 * alpha)
    for n in range(1, len(S)):
        gamma[n] = max(S[n] / S[i] for i in range(n))
        if gamma[n] >= threshold:
            print(f"Alarm raised at observation {n} with gamma={gamma[n]}")
            # return True, gamma
    return False, gamma

def shiryaev_roberts_procedure(S, c):
    """
    Implements the Shiryaev-Roberts statistic.
    """
    sigma = np.zeros(len(S))
    for n in range(1, len(S)):
        sigma[n] = sum(S[n] / S[i] for i in range(n))
        if sigma[n] >= c:
            print(f"Alarm raised at observation {n} with sigma={sigma[n]}")
            # return True, sigma
    return False, sigma

def simple_jumper_martingale(p_values, J=0.01, threshold=100):
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

        if C >= threshold:
            print(f"Alarm raised at observation {i} with martingale value={C}")
            # return True, np.array(martingale_values)
    
    return False, np.array(martingale_values)

def retrain_count(conformity_score, training_schedule, sr_threshold, cu_confidence):
    p_values = calculate_p_values(conformity_score)
    retrain_m, martingale_value = simple_jumper_martingale(p_values)

    if training_schedule == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold)
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence)
    
    return retrain_m, retrain_s, martingale_value, sigma



def training_function(dataset0, dataset0_name, dataset1=None, training_schedule='variable', \
                      sr_threshold=1e6, cu_confidence=0.99, muh_fun_name='RF', test0_size=1599/4898, \
                      dataset0_shift_type='none', cov_shift_bias=1.0, plot_errors=False, seed=0, cs_type='signed'):
    
    dataset0_train, dataset0_test_0 = split_and_shift_dataset0(dataset0, dataset0_name, test0_size=test0_size, \
                                                               dataset0_shift_type=dataset0_shift_type, \
                                                               cov_shift_bias = cov_shift_bias, seed=seed)
    X, y, folds = split_into_folds(dataset0_train, seed=seed)

    cs_0, cs_1 = train_and_evaluate(X, y, folds, dataset0_test_0, dataset1, muh_fun_name, seed=seed, cs_type=cs_type)

    fold_martingales_0, fold_martingales_1 = [], []
    sigmas_0, sigmas_1 = [], []
    retrain_m_count_0, retrain_s_count_0 = 0, 0
    retrain_m_count_1, retrain_s_count_1 = 0, 0
    
    for score_0 in cs_0:
        m_0, s_0, martingale_value_0, sigma_0 = retrain_count(score_0, training_schedule, sr_threshold, cu_confidence)
        if m_0:
            retrain_m_count_0 += 1
        if s_0:
            retrain_s_count_0 += 1
        fold_martingales_0.append(martingale_value_0)
        sigmas_0.append(sigma_0)
        
        
    for score_1 in cs_1:
        m_1, s_1, martingale_value_1, sigma_1 = retrain_count(score_1, training_schedule, sr_threshold, cu_confidence)
        if m_1:
            retrain_m_count_1 += 1
        if s_1:
            retrain_s_count_1 += 1
        fold_martingales_1.append(martingale_value_1)
        sigmas_1.append(sigma_1)
    
        
    ## Note: Moved plotting to outside training function
#     plot_martingale_paths(
#         dataset0_paths=sigmas_0, 
#         dataset0_name=dataset0_name,
#         dataset1_paths=sigmas_1, 
#         cs_0=cs_0,
#         cs_1=cs_1,
#         change_point_index=len(X)/3,
#         title="Paths of Shiryaev-Roberts Procedure",
#         ylabel="Shiryaev-Roberts Statistics",
#         file_name="shiryaev_roberts",
#         dataset0_shift_type=dataset0_shift_type,
#         cov_shift_bias=cov_shift_bias,
#         plot_errors=plot_errors
#     )
    
    

    # Decide to retrain based on two out of three martingales exceeding the threshold
    if retrain_m_count_0 >= 2 or retrain_s_count_0 >= 2:
        retrain_decision_0 = True
    else:
        retrain_decision_0 = False

    if retrain_m_count_1 >= 2 or retrain_s_count_1 >= 2:
        retrain_decision_1 = True
    else:
        retrain_decision_1 = False

    if retrain_decision_0:
        print("Retraining the model for normal white wine...")
    else:
        print("No retraining needed for normal white wine.")

    if retrain_decision_1:
        print("Retraining the model for red wine...")
    else:
        print("No retraining needed for red wine.")
        
        
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
    parser.add_argument('--d0_shift_type', type=str, default='none', help='Shift type to induce in dataset0.')
    parser.add_argument('--bias', type=float, default=0.0, help='Scalar bias magnitude parameter lmbda for exponential tilting covariate shift.')
    parser.add_argument('--plot_errors', type=bool, default=False, help='Whether to also plot absolute errors.')
    parser.add_argument('--schedule', type=str, default='variable', help='Training schedule: variable or fixed.')
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to run experiments on.')
    parser.add_argument('--errs_window', type=int, default=50, help='Num observations to average for plotting errors.')
    parser.add_argument('--cs_type', type=str, default='signed', help="Nonconformity score type: 'abs' or 'signed' ")
    
    
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
    print("cov_shift_bias: ", cov_shift_bias)
    
    
    ## Load datasets into dataframes
    dataset0 = eval(f'get_{dataset0_name}_data()')
    if (dataset1_name is not None):
        dataset1 = eval(f'get_{dataset1_name}_data()')
    else:
        dataset1 = None
    
    
    paths_all = pd.DataFrame()

    
    for seed in range(0, n_seeds):
        # training_schedule = ['variable', 'fix']
        paths_curr = training_function(dataset0, dataset0_name, dataset1, training_schedule=training_schedule, muh_fun_name=muh_fun_name, test0_size = test0_size, dataset0_shift_type=dataset0_shift_type, cov_shift_bias=cov_shift_bias, plot_errors=plot_errors, seed=seed, cs_type=cs_type)
        
        paths_all = pd.concat([paths_all, paths_curr], ignore_index=True)
        
        
    paths_all.to_csv(f'../results/path_results_{dataset0_name}_{muh_fun_name}_{dataset0_shift_type}shift_bias{cov_shift_bias}_nseeds{n_seeds}.csv')
    
    
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
            cs_abs_0_stderr_fold.append(paths_all_abs_sub['cs_0_'+str(i)].std()/ np.sqrt(n_seeds*errs_window))
        
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
        cs_abs_0_means=cs_abs_0_means,
        cs_abs_1_means=cs_abs_1_means,
        cs_abs_0_stderr=cs_abs_0_stderr,
        cs_abs_1_stderr=cs_abs_1_stderr,
        errs_window=errs_window,
        change_point_index= len(dataset0)*(1-test0_size)/3,
        title="Average paths of Shiryaev-Roberts Procedure",
        ylabel="Shiryaev-Roberts Statistics",
        file_name="shiryaev_roberts",
        dataset0_shift_type=dataset0_shift_type,
        cov_shift_bias=cov_shift_bias,
        plot_errors=plot_errors,
        n_seeds=n_seeds,
        cs_type=cs_type
    )
