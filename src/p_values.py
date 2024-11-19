import numpy as np
from sklearn.linear_model import SGDClassifier
import pdb

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
def calculate_weighted_p_values(conformity_scores, W_i, n_cal, weights_to_compute='fixed_cal', depth=1):
    """
    Calculate the weighted conformal p-values from conformity scores and given normalized weights 
    (i.e., enforce np.sum(normalized_weights) = 1).
    
    W_i : List of likelihood ratio weight est. arrays, each t-th array is length (n_cal+t)
    """
    init_phase = 100
    wp_values = np.zeros(len(conformity_scores))
    
    
    ## p-values for original calibration set calculated as before
    wp_values[0:(n_cal+init_phase)] = calculate_p_values(conformity_scores[0:(n_cal+init_phase)]) 
        
    if weights_to_compute in ['fixed_cal', 'fixed_cal_oracle', 'sliding_window']:

        if weights_to_compute == 'fixed_cal':
            assert depth == 1, "Estimation depth must be 1."
        elif weights_to_compute == 'sliding_window':
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
            
            if (weights_to_compute == 'fixed_cal_oracle'):
                W_i_t = W_i[idx_include]
            else:
                W_i_t = W_i[t_][idx_include]
            
            ## Normalize weights on subset of weights
            normalized_weights_t = W_i_t / np.sum(W_i_t)
            
            ## Calculate weighted p-values
            wp_values[n_cal+t_] = np.sum(normalized_weights_t[conformity_scores_t < conformity_scores_t[-1]]) + \
                            np.random.uniform() * np.sum(normalized_weights_t[conformity_scores_t == conformity_scores_t[-1]])

    elif (weights_to_compute in ['one_step_oracle', 'one_step_est', 'batch_oracle', 'multistep_oracle']):
        
        T = len(conformity_scores) - n_cal ## Number of total test observations
        
        ## Note: in loop here, t_ := t-1 for zero-indexing
            ## Note: Previously this loop started at init_phase
        for t_ in range(0, T):
            
            ## Subset conformity scores and weights based on idx_include
            conformity_scores_t = conformity_scores[:(n_cal+t_+1)]
            
            if (weights_to_compute in ['one_step_est', 'batch_oracle']):
                W_i_t = W_i[t_][:(n_cal+t_+1)]
                
            elif (weights_to_compute == 'one_step_oracle'):
                W_i_t = W_i[:(n_cal+t_+1)]
                
            elif (weights_to_compute == 'multistep_oracle'):
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