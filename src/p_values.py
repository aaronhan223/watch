import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity
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

class CIFAR10Discriminator(nn.Module):
    """
    A discriminator model to distinguish between source (uncorrupted) and target (corrupted) CIFAR-10 data.
    The model is designed to achieve both good accuracy and calibration.
    """
    def __init__(self, dropout_rate=0.3):
        super(CIFAR10Discriminator, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        
        # Output layer with temperature scaling for better calibration
        self.fc3 = nn.Linear(128, 2)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Temperature parameter for calibration
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        
        # Output with temperature scaling for better calibration
        logits = self.fc3(x)
        return logits / self.temperature
    
    def predict_proba(self, x):
        """
        Return calibrated probabilities for source vs target classification
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class MNISTDiscriminator(nn.Module):
    """
    A discriminator model to distinguish between source (uncorrupted) and target (corrupted) MNIST data.
    The model is designed to achieve both good accuracy and calibration.
    """
    def __init__(self, dropout_rate=0.3):
        super(MNISTDiscriminator, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer with temperature scaling for better calibration
        self.fc2 = nn.Linear(128, 2)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Temperature parameter for calibration
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer with temperature scaling for better calibration
        x = self.fc2(x)
        return x / self.temperature
    
    def predict_proba(self, x):
        """
        Returns calibrated probabilities for binary classification
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    

def offline_lik_ratio_estimates_images(cal_test_w_est_loader, test_loader, dataset0_name = 'mnist', device=None, 
                                       setting='', epochs=80, lr=1e-3, epsilon=1e-9, classifier='MLP'):

     # Train smaller MLP model to estimate source/target probabilities
    if dataset0_name == 'mnist':
        # model = MLP(input_size=784, hidden_size=32, num_classes=2).to(device)
        model = MNISTDiscriminator(dropout_rate=0.3).to(device)
    elif dataset0_name == 'cifar10':
        model = CIFAR10Discriminator(dropout_rate=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ## Fit prob classifier offline
    fit(model, epochs, cal_test_w_est_loader, optimizer, setting, device)
    ## Evaluate probability estimiates
    cal_test_prob_est, _, _, _ = eval_loss_prob(model, device, setting, cal_test_w_est_loader, test_loader, binary_classifier_probs=True)

    return cal_test_prob_est / (1 - cal_test_prob_est + epsilon)


def online_lik_ratio_estimates(X_cal, X_test_w_est, X_test_0_only, adapt_start=None, classifier='NN'):
    
    n_cal = len(X_cal)
    dim = np.shape(X_cal)[1]
    init_phase = len(X_test_w_est)
    n_test = len(X_test_0_only)
    T = len(X_test_0_only) + n_cal - adapt_start
    
    W_i = np.zeros((T, adapt_start + T))
    
    
    if (adapt_start is None):
        ## Begin adaptive at deployment time, ie first test point after calibration set. 
        ## Ie, if adapt_start is None, then adapt_start <- n_cal 
        adapt_start = n_cal 
    
    ## Scale data
    X_all = np.concatenate((X_cal, X_test_w_est, X_test_0_only), axis=0)
    scaler = preprocessing.StandardScaler().fit(X_all)
    X_all_scaled = scaler.transform(X_all)
    X_test_offline_online = X_all_scaled[adapt_start:]
    X_cal_test_scaled = np.concatenate((X_all_scaled[0:n_cal], X_all_scaled[-n_test:]), axis=0)
    
   
    class_labels_all = np.concatenate((np.zeros(adapt_start), np.ones(init_phase + T)), axis=0)
    
#     idx_include_classifier = np.concatenate((np.repeat(True, adapt_start + init_phase), np.repeat(False, T)), axis=0)
    idx_include_classifier = np.concatenate(([True], np.repeat(False, adapt_start + init_phase-2), [True], np.repeat(False, T)), axis=0)
    idx_include_kde_test = np.concatenate((np.repeat(True, init_phase), np.repeat(False, T)), axis=0)  ## init_phase : num initial offline test data available; T : num test points observed online
    
    
    ## Instantiate lik-ratio model
    if (classifier=='LR'):
        lik_ratio_model = LogisticRegression(warm_start=True)
        
    elif (classifier=='RF'):
#         n_estimators = max(20, int(dim))
        lik_ratio_model = RandomForestClassifier(n_estimators=10, criterion='entropy', min_weight_fraction_leaf=0.1,\
                                                 warm_start=True)
        
    elif (classifier=='NN'):
#         hidden_layer_size = max(5, int(dim/5))
        lik_ratio_model = MLPClassifier(hidden_layer_sizes=(20,),solver='lbfgs', alpha=0.1, activation='logistic', warm_start=True)
    
    elif (classifier=='KDE'):
        kde_cal = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde_cal.fit(X_cal)
        
#         print("len(X_test_offline_online) : ", len(X_test_offline_online))
#         print("len(idx_include_kde_test)  : ", len(idx_include_kde_test))
#         print("init_phase : ", init_phase)
#         print("n_test : ", n_test)
#         assert (len(X_test_offline_online) == len(idx_include_kde_test))
        
        kde_test = KernelDensity(kernel='gaussian', bandwidth=0.5, atol=1000, rtol=1000, leaf_size=10)
        kde_test.fit(X_test_offline_online[idx_include_kde_test])
    
    
        
    ## t=0 : Offline initialization phase (but set warm_start=True)
    ## t>0 : Online adaptation phase
    for t in range(0, T):
#         if (t % 100):
#             print(t)
        
        if (classifier=='KDE'):
            print("Running KDE")
            
            ## Kernel density estimation:
            print(f"fitting on {len(X_test_offline_online[idx_include_kde_test])} pts ")
            kde_test.fit(X_test_offline_online[idx_include_kde_test])
            est_probs_test = np.exp(kde_test.score_samples(X_cal_test_scaled))
            est_probs_cal = np.exp(kde_cal.score_samples(X_cal_test_scaled))
#             print("est_probs_test sum       : ", np.sum(est_probs_test))
#             print("est_probs_test normed sum : ", est_probs_test/np.sum(est_probs_test))
            
            idx_include_kde_test[init_phase + t] = True
            
            W_i[t] = est_probs_test / est_probs_cal
            
            
        else:
#             if (classifier == 'RF'):
#                 lik_ratio_model.n_estimators += 1
                
            ## Probabilistic classification density-ratio estimation:
            lik_ratio_model.fit(X_all_scaled[idx_include_classifier], class_labels_all[idx_include_classifier])
            est_probs = lik_ratio_model.predict_proba(X_cal_test_scaled)
#             if (t % 25==0):
#                 print(f'est_probs[{t}] {np.array(est_probs[-5:,1])}')
            idx_include_classifier[adapt_start + init_phase + t] = True
            if (t < n_cal):
                idx_include_classifier[t] = True ## Add calibration point
            
            ## Avoid divide-by zero and nan issues
            min_nonzero = min(est_probs[:,0][est_probs[:,0]>0])
#             print("min_nonzero : ", min_nonzero)
            est_probs_source = est_probs[:,0]
            est_probs_source[est_probs[:,0] == 0] = min_nonzero
            est_probs_test = np.nan_to_num(est_probs[:,1], nan=np.inf)
            
            W_i[t] = est_probs_test / est_probs_source

            
#         print(f'W_i[{t}][-10:] : ', W_i[t][-10:])
        
#         idx_include[adapt_start + init_phase + t] = True
    
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
    p_values = np.array([(np.sum(conformity_scores[:i] > conformity_scores[i]) + 
                         np.random.uniform() * np.sum(conformity_scores[:(i+1)] == conformity_scores[i])) / (i + 1)
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
        
    return p_values, q_lower, q_upper




def calculate_cp_quantiles_holdout(cal_scores, holdout_scores, alpha, cs_type='abs'):
    """
    Calculate the conformal quantiles on holdout_scores, using set constructed on cal_scores.
    """
    n = len(holdout_scores)
    q_lower = np.zeros(n)
    q_upper = np.zeros(n)
    
    ## Quantiles: Each ith quantile is computed before labels observed, with conservative validity
    if (cs_type != 'signed'):
        cal_scores_inf = np.concatenate((cal_scores, [np.inf])) ## inf in place of test pt cs for conservativeness
        
        for i in range(n):
            ## For each i, q_upper[i] := quantile(conformity_scores[:i] \cup inf, 1-alpha)
                        ## q_lower[i] := -q_upper[i]
            q_upper[i] = np.quantile(cal_scores, 1-alpha)
        q_lower = - q_upper
        
    else:
        ## Intervals for signed scores computed by (alpha/2) lower q, (1-alpha/2) upper q
        cal_scores_inf = np.concatenate((cal_scores, [np.inf]))
        cal_scores_neg_inf = np.concatenate((cal_scores, [-np.inf]))
        
        for i in range(n):
            ## For each i, q_upper[i] := quantile(conformity_scores[:i] \cup inf, 1-alpha/2)
                        ## q_lower[i] := quantile(conformity_scores[:i] \cup -inf, alpha/2)
            q_upper[i] = np.quantile(cal_scores, 1-alpha/2)
            q_lower[i] = np.quantile(cal_scores, alpha/2)
    
    ## Replace nan with appropriate inf values
    np.nan_to_num(q_lower, copy=False, nan=-np.inf, neginf=-np.inf)
    np.nan_to_num(q_upper, copy=False, nan=np.inf, posinf=np.inf)
        
    return q_lower, q_upper




def calculate_weighted_cp_quantiles_holdout(cal_propper_scores, holdout_test_scores, W_cal_holdout_test, n_cal, adapt_start, alpha, cs_type='abs'):
    '''
    cal_propper_scores  : array length int(adapt_start/2)
    holdout_test_scores : array length n_test - int(adapt_start/2)
    W_cal_holdout_test  : matrix shape (T, adapt_start + T)
    '''
    
    ## Note: W_cal_holdout_test is a matrix size (num_test, adapt_start + num_test)
    
    n_cal_propper = len(cal_propper_scores) ## int(n_cal/2)
    n_holdout = n_cal - n_cal_propper ## 
    n_holdout_test = len(holdout_test_scores)
    adapt_start_min_cal_prop = adapt_start - n_cal_propper
    T = len(W_cal_holdout_test)
    n_test = n_holdout_test - n_holdout
    wq_lower = np.zeros(n_holdout_test) ## lower weighted quantiles
    wq_upper = np.zeros(n_holdout_test) ## upper weighted quantiles
#     print("n_cal ", n_cal)
#     print("T ", T)
#     print("len(W_cal_holdout_test) ", len(W_cal_holdout_test))
#     assert(n_cal_propper + n_holdout_test == np.shape(W_cal_holdout_test)[1])
    
    ## Holdout set (ie, prior to starting adaptation) is assumed exchangeable from source distribution,
    ## so weights reduce to uniform weights
    wq_lower[:adapt_start_min_cal_prop], wq_upper[:adapt_start_min_cal_prop] = calculate_cp_quantiles_holdout(cal_propper_scores, holdout_test_scores[:adapt_start_min_cal_prop], alpha, cs_type)
    
    ## Quantiles: Each ith quantile is computed before labels observed, with conservative validity
    if (cs_type != 'signed'):
        cal_propper_scores_inf = np.concatenate((cal_propper_scores, [np.inf])) ## inf in place of test pt cs for conservativeness
        
        ## Which weights to include
        idx_include = np.concatenate((np.repeat(True, n_cal_propper), np.repeat(False, n_holdout), np.repeat(False, n_test)), axis=0) 
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## idx_include implements indices for 'fixed cal' ie, comparing to [0:adapt_start] \cup adapt_start + t_ 
            idx_include[adapt_start+t_] = True ## Move to curr test point
            if (t_ > 0):
                idx_include[adapt_start+t_] = False ## Exclude most recent test point again (except at start, is a cal point)
            W_t = W_cal_holdout_test[t_][idx_include]
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_t / np.sum(W_t) ## Should be length n_cal+1
                
            ## For each i, q_upper[i] := quantile(conformity_scores[:i] \cup inf, 1-alpha)
                        ## q_lower[i] := -q_upper[i]
            wq_upper[n_holdout+t_] = weighted_quantile(cal_propper_scores_inf, normalized_weights_t, 1-alpha)
        wq_lower = - wq_upper
        
    else:
        ## Intervals for signed scores computed by (alpha/2) lower q, (1-alpha/2) upper q
        cal_propper_scores_inf = np.concatenate((cal_propper_scores, [np.inf]))
        cal_propper_scores_neg_inf = np.concatenate((cal_propper_scores, [-np.inf]))
        
        ## Which weights to include
        idx_include = np.concatenate((np.repeat(True, n_cal_propper), np.repeat(False, T)), axis=0) 
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## idx_include implements indices for 'fixed cal' ie, comparing to [0:adapt_start] \cup adapt_start + t_ 
            idx_include[adapt_start+t_] = True ## Move to curr test point
            if (t_ > 0):
                idx_include[adapt_start+t_] = False ## Exclude most recent test point again (except at start, is a cal point)
            W_t = W_cal_holdout[t_][idx_include]
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_t / np.sum(W_t) ## Should be length n_cal+1
                
            ## For each i, q_upper[i] := quantile(conformity_scores[:i] \cup inf, 1-alpha)
                        ## q_lower[i] := -q_upper[i]
            wq_upper[n_holdout+t_] = weighted_quantile(cal_propper_scores_inf, normalized_weights_t, 1-alpha/2)
            wq_lower[n_holdout+t_] = weighted_quantile(cal_propper_scores_neg_inf, normalized_weights_t, 1-alpha/2)
            
    ## Replace nan with appropriate inf values
    np.nan_to_num(wq_lower, copy=False, nan=-np.inf, neginf=-np.inf)
    np.nan_to_num(wq_upper, copy=False, nan=np.inf, posinf=np.inf)
        
    return wq_lower, wq_upper




## Note: This is for calculating the weighted p-values once the normalized weights have already been calculated
def calculate_weighted_p_values_and_quantiles(args, conformity_scores, W_i, adapt_start, method, depth=1, cs_resampled_cal_list=None, num_test_unshifted=0):
    """
    Calculate the weighted conformal p-values from conformity scores and given normalized weights 
    (i.e., enforce np.sum(normalized_weights) = 1).
    
    W_i : List of likelihood ratio weight est. arrays, each t-th array is length (adapt_start+t)
    adapt_start : Index of first point that is assumed part of test distribution rather than cal. If method != 'fixed_cal_dyn', 
                  then  adapt_start==n_cal
    """
    print("calculating weighted p-values")
    n = len(conformity_scores)
    wp_values = np.zeros(n) ## p-values calculated with weighted conformity scores
    wq_lower = np.zeros(n) ## lower weighted quantiles
    wq_upper = np.zeros(n) ## upper weighted quantiles
    
    print("Inside calc wpvals ")
    print("adapt_start        : ", adapt_start)
    print("num_test_unshifted : ", num_test_unshifted)
    
    
    # print("len cs_resampled_cal_list : ", len(cs_resampled_cal_list))
#     ## For 0:adapt_start, compute as standard p-values and quantiles
#     if (args is None):
#         wp_values[0:adapt_start], wq_lower[0:adapt_start], wq_upper[0:adapt_start] = \
#                                         calculate_p_values_and_quantiles(conformity_scores[0:adapt_start], alpha, cs_type)
#     else:
    wp_values[0:adapt_start], wq_lower[0:adapt_start], wq_upper[0:adapt_start] = \
                                        calculate_p_values_and_quantiles(conformity_scores[0:adapt_start], args.alpha, args.cs_type)
    
    ## For computing (conservative) weighted quantiles, append infinity (which takes place of test pt score)
    conformity_scores_inf = np.concatenate((conformity_scores, [np.inf]))
    if (args.cs_type == 'signed'):
        conformity_scores_neg_inf = np.concatenate((conformity_scores, [-np.inf])) 

    if method in ['fixed_cal', 'fixed_cal_oracle', 'sliding_window', 'fixed_cal_offline', 'fixed_cal_dyn', 'resample_cal_oracle']:

        if method == 'fixed_cal':
            assert depth == 1, "Estimation depth must be 1."
        elif method == 'sliding_window':
            assert depth > 1, "Estimation depth must be greater than 1."

        T = len(conformity_scores) - adapt_start ## Number of test observations after begin adaptation
        ## indices to include in computing weighted p-values
        idx_include = np.concatenate((np.repeat(True, adapt_start), np.repeat(False, T)), axis=0) 
        
        ## indices to include in computing weighted quantiles, where last entry includes np.inf
        idx_include_inf = np.concatenate((np.repeat(True, adapt_start), np.repeat(False, T), [True]), axis=0) 
        
        print("T : ", T)
        ## Note: in loop here, t_ := t-1 for zero-indexing
        for t_ in range(0, T):
            
            ## idx_include implements indices for 'fixed cal' ie, comparing to [0:adapt_start] \cup adapt_start + t_ 
            idx_include[adapt_start+t_] = True ## Move to curr test point
#             print("t_ : ", t_)
#             print("depth : ", depth)
            if (t_ > depth - 1):
                idx_include[adapt_start+t_-depth] = False ## Exclude most recent "depth" number of test point again (except at start, is a cal point)
            
            ## Subset conformity scores and weights based on idx_include
            
            if (method == 'resample_cal_oracle'):
#                 print("len conformity_scores_t", len(cs_resampled_cal_list))
                conformity_scores_t = cs_resampled_cal_list[t_]
            else:
                conformity_scores_t = conformity_scores[idx_include]
                
#             print("conformity_scores_t : ", conformity_scores_t[0:10])
            
            
            if (method in ['fixed_cal_oracle', 'fixed_cal_offline']):
                ## 
                W_i_t = W_i[idx_include]
            else:
#                 print("W_i[t_]  : ", W_i[t_])
                W_i_t = W_i[t_][idx_include]
                
#                 if (method == 'resample_cal_oracle'):
#                     print(f'W_i_t : {t_}, {W_i[t_]}')
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
            ## Calculate weighted quantiles
            if (args.cs_type != 'signed'):
                if (method == 'resample_cal_oracle'):
                    wq_upper[adapt_start+t_] = weighted_quantile(np.concatenate((cs_resampled_cal_list[t_][:-1], [np.inf])), normalized_weights_t, 1-args.alpha)
                
                else:
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
#             wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
#                             np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
            if (np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]]) < args.alpha):
                ## If no more than (relative) weight 'alpha' put on test point score, compute exact (randomized) p-values:
                U_t = np.random.uniform()
                wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t > conformity_scores_t[-1]]) + \
                            U_t * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])
                
            else:
                ## Else: over (relative) weight 'alpha' put on test pt score, compute anticonservative (and deterministic) p-values:
                wp_values[adapt_start+t_] = np.sum(normalized_weights_t[conformity_scores_t > conformity_scores_t[-1]])
                
                # print(f'len(normalized_weights_t) : {len(normalized_weights_t)}')
                # print("Using conservative p-values at index ", t_, "p-val : ", wp_values[adapt_start+t_])
                # print(f'conformity_scores_{t_}[-1] : {conformity_scores_t[-1]}; weighted median : {weighted_quantile(conformity_scores_t, normalized_weights_t, 0.5)}')
#                 if (np.isnan(wp_values[adapt_start+t_])):
#                     print("conformity_scores_t[-1] : ", conformity_scores_t[-1])
#                     print(wp_values)
                    
#             if (wp_values[adapt_start+t_] > 0.99):
#                 print(f'large weighted p val            : {wp_values[adapt_start+t_]}')
#                 print(f'unweighted p val (conservative) : {np.sum(conformity_scores_t <= conformity_scores_t[-1])/len(conformity_scores_t)}')
#                 print(f'< sum : {np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]])}')
#                 print(f'= sum : {np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])}')
#                 print(f'conformity_scores_t[-1] : {conformity_scores_t[-1]}')
# #                 print(f'conformity_scores_t == conformity_scores_t[-1] : {conformity_scores_t == conformity_scores_t[-1]}')
#                 print(f'normalized_weights_t : {normalized_weights_t}')
#                 print(f'U_t : {U_t}')
            

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


def calculate_prediction_sets(probabilities, alpha, class_labels=None):
    """
    Calculate conformal prediction sets for classification tasks.
    
    Parameters:
    -----------
    probabilities : numpy.ndarray
        Shape (n_samples, n_classes) containing model's predicted class probabilities
    alpha : float
        Desired significance level (e.g., 0.1 for 90% confidence)
    class_labels : numpy.ndarray, optional
        The true class labels to calculate coverage
        
    Returns:
    --------
    prediction_sets : list of lists
        Each inner list contains the class indices in the prediction set
    set_sizes : numpy.ndarray
        Size (cardinality) of each prediction set
    coverage : numpy.ndarray or None
        Boolean array indicating whether true class is in the prediction set (if class_labels provided)
    """
    n_samples = probabilities.shape[0]
    prediction_sets = []
    set_sizes = np.zeros(n_samples, dtype=int)
    
    # For each sample, create prediction set by including classes in descending probability order
    # until cumulative probability exceeds 1-alpha threshold
    for i in range(n_samples):
        # Sort probabilities in descending order and get corresponding indices
        sorted_indices = np.argsort(probabilities[i])[::-1]
        sorted_probs = probabilities[i][sorted_indices]
        
        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Find minimum number of classes needed to exceed 1-alpha threshold
        cutoff_idx = np.argmax(cumulative_probs >= 1 - alpha)
        
        # Classes in prediction set
        pred_set = sorted_indices[:cutoff_idx + 1].tolist()
        prediction_sets.append(pred_set)
        set_sizes[i] = len(pred_set)
    
    # Calculate coverage if true labels are provided
    coverage = None
    if class_labels is not None:
        coverage = np.array([class_labels[i] in prediction_sets[i] for i in range(n_samples)])
    
    return prediction_sets, set_sizes, coverage
