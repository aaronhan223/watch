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


def load_wine_quality_data():
    white_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
    red_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
    return white_wine, red_wine

def split_white_wine_data(white_wine):
    white_wine_train, white_wine_test_0 = train_test_split(white_wine, test_size=1599/4898, shuffle=True, random_state=42)
    return white_wine_train, white_wine_test_0

def split_into_folds(white_wine_train):
    X = white_wine_train.drop('quality', axis=1).to_numpy()
    y = white_wine_train['quality'].to_numpy()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(kf.split(X, y))
    return X, y, folds

def train_and_evaluate(X, y, folds, white_wine_test_0, red_wine):
    fold_results = []
    cs_0 = []
    cs_1 = []
    
    for i, (train_index, cal_index) in enumerate(folds):
        if i == 2:  # Adjust the last fold to have 1099 in training
            train_index, cal_index = train_index[:-1], cal_index
        X_train, X_cal = X[train_index], X[cal_index]
        y_train, y_cal = y[train_index], y[cal_index]
        
        # Train the model on the training set proper
        model = Pipeline([
            ('scaler', StandardScaler()),  # Normalize the data
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(X_train, y_train)
        
        # Evaluate using the calibration set + test set 0 (Scenario 0)
        X_test_0 = np.concatenate((X_cal, white_wine_test_0.drop('quality', axis=1).to_numpy()), axis=0)
        y_test_0 = np.concatenate((y_cal, white_wine_test_0['quality'].to_numpy()), axis=0)
        y_pred_0 = model.predict(X_test_0)
        
        # Evaluate using the calibration set + test set 1 (Scenario 1)
        X_test_1 = np.concatenate((X_cal, red_wine.drop('quality', axis=1)), axis=0)
        y_test_1 = np.concatenate((y_cal, red_wine['quality']), axis=0)
        y_pred_1 = model.predict(X_test_1)
        
        np.set_printoptions(threshold=np.inf)
        conformity_scores_0 = y_test_0 - y_pred_0
        conformity_scores_1 = y_test_1 - y_pred_1

        # Store results for each fold
        fold_results.append({
            'fold': i + 1,
            'scenario_0_predictions': y_pred_0,
            'scenario_1_predictions': y_pred_1
        })
        cs_0.append(conformity_scores_0)
        cs_1.append(conformity_scores_1)
    
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

def retrain_count(conformity_score, method, sr_threshold, cu_confidence):
    p_values = calculate_p_values(conformity_score)
    retrain_m, martingale_value = simple_jumper_martingale(p_values)

    if method == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold)
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence)
    
    return retrain_m, retrain_s, martingale_value, sigma

def training_function(white_wine, red_wine, method, sr_threshold=1e6, cu_confidence=0.99):
    
    white_wine_train, white_wine_test_0 = split_white_wine_data(white_wine)
    X, y, folds = split_into_folds(white_wine_train)

    cs_0, cs_1 = train_and_evaluate(X, y, folds, white_wine_test_0, red_wine)

    fold_martingales_0, fold_martingales_1 = [], []
    sigmas_0, sigmas_1 = [], []
    retrain_m_count_0, retrain_s_count_0 = 0, 0
    retrain_m_count_1, retrain_s_count_1 = 0, 0
    for score_0, score_1 in zip(cs_0, cs_1):
        m_0, s_0, martingale_value_0, sigma_0 = retrain_count(score_0, method, sr_threshold, cu_confidence)
        m_1, s_1, martingale_value_1, sigma_1 = retrain_count(score_1, method, sr_threshold, cu_confidence)
        if m_0:
            retrain_m_count_0 += 1
        if s_0:
            retrain_s_count_0 += 1
        if m_1:
            retrain_m_count_1 += 1
        if s_1:
            retrain_s_count_1 += 1
        fold_martingales_0.append(martingale_value_0)
        fold_martingales_1.append(martingale_value_1)
        sigmas_0.append(sigma_0)
        sigmas_1.append(sigma_1)

    plot_martingale_paths(
        white_wine_paths=sigmas_0, 
        red_wine_paths=sigmas_1, 
        change_point_index=1100,
        title="Paths of Shiryaev-Roberts Procedure",
        ylabel="Shiryaev-Roberts Statistics",
        file_name="shiryaev_roberts"
    )

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

white_wine, red_wine = load_wine_quality_data()
# method = ['variable', 'fix']
training_function(white_wine, red_wine, 'variable')
