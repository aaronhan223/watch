import numpy as np
from sklearn import decomposition
from copy import deepcopy



def compute_w_ptest_split_active_replacement(cal_test_vals_mat, depth_max):
    '''
        Computes the estimated MFCS Split CP weights for calibration and test points 
        (i.e., numerator in Eq. (9) in main paper or Eq. (16) in Appendix B.2)
        
        @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
                ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
                ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1
                
        @param : depth_max          : (int) indicating the maximum recursion depth
        
        :return: Unnormalized weights on calibration and test points, computed for recursion depth depth_max
    '''
    if (depth_max < 1):
        raise ValueError('Error: depth_max should be an integer >= 1. Currently, depth_max=' + str(depth_max))
      
    if (depth_max == 1):
        ## 
        return cal_test_vals_mat[-1]
        
    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(np.array(cal_test_vals_mat[-1])).flatten()
    idx_include = np.repeat(True, n_cal_test)

    
    for i in range(n_cal_test):
#         print(i)
        idx_include[i] = False
        idx_include[i-1] = True
        summation = compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], depth_max-1)
        adjusted_vals[i] = adjusted_vals[i] * summation
    return adjusted_vals
            
        
def compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat, depth_max):
    '''
        Helper function for "compute_w_ptest_split_active_replacement". Computes a summation such as the two sums in the numerator in equation (7) in paper
        
        @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
                ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
                ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1
                
        @param : depth_max          : (int) indicating the maximum recursion depth
        
        :return: Summation such as the two sums in the numerator in equation (7) in paper
    '''
    if (depth_max == 1):
        return np.sum(cal_test_vals_mat)
    
    else:
        summation = 0
        n_cal_test = np.shape(cal_test_vals_mat)[1]
        idx_include = np.repeat(True, n_cal_test)
        for i in range(n_cal_test):
            idx_include[i] = False
            idx_include[i-1] = True
            summation += cal_test_vals_mat[-1,i]*compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], depth_max - 1) 
        return summation
    


    


def get_w(x_pca, x, dataset, bias):
    
    if (dataset == '1dim_synthetic'):
        return np.exp(x * bias).squeeze()
    
    elif (dataset == '1dim_synthetic_v2'):
        return np.exp(x * bias).squeeze()
    
    elif (dataset == '1dim_linear_synthetic'):
        return np.exp(x * bias).squeeze()
    
    elif (dataset=='airfoil'):
        return np.exp(x[:,[0,4]] @ [-bias,bias])
    
    elif(dataset=='airfoil_pca'):
        pca_1 = decomposition.PCA(n_components=1)
        pca_1.fit(x_pca)
        x_red_1 = pca_1.transform(x)
        
        return np.exp(np.abs(x_red_1) * bias).squeeze()
    
    elif(dataset == 'white_wine'):
        return np.exp(x[:,[0,10]] @ [-bias,bias])
    
    elif(dataset=='white_wine_pca'):
#         pca_1 = decomposition.PCA(n_components=1)
#         pca_1.fit(x_pca[:, 0:5])
#         x_red_1 = pca_1.transform(x[:, 0:5])
        
# #         np.random.seed(5)
#         pca_2 = decomposition.PCA(n_components=1)
#         pca_2.fit(x_pca[:, 5:10])
#         x_red_2 = pca_2.transform(x[:, 5:10])
        
#         x_red = np.c_[x_red_1, x_red_2]
        pca_1 = decomposition.PCA(n_components=1)
        pca_1.fit(x_pca)
        x_red = pca_1.transform(x)
        x_red_abs = np.abs(x_red)
        x_red_abs_normed = x_red_abs / np.max(x_red_abs, axis=0)
        # Previous version just passed x_red and did x_red @ [-bias,bias]
        
        return np.exp(x_red_abs_normed @ [bias])
        
    
    
    elif(dataset == 'red_wine'):
        return np.exp(x[:,[0,10]] @ [-bias,bias])

    
    ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['wave']):
#         np.random.seed(5)
        pca_1 = decomposition.PCA(n_components=1)
        pca_1.fit(x_pca[:, 0:32])
        x_red_1 = pca_1.transform(x[:, 0:32])
        
#         np.random.seed(5)
        pca_2 = decomposition.PCA(n_components=1)
        pca_2.fit(x_pca[:, 32:48])
        x_red_2 = pca_2.transform(x[:, 32:48])
        
        x_red = np.c_[x_red_1, x_red_2]
        
        return np.exp(x_red @ [-bias,bias])
    
        ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['superconduct']):
#         np.random.seed(5)
        pca = decomposition.PCA(n_components=1)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [bias])
    
        ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['communities']):
#         np.random.seed(5)
        pca = decomposition.PCA(n_components=2)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        return np.exp(x_red @ [-bias,bias])
    

def wsample(wts, n, d, frac=0.5):
    n = len(wts) ## n : length or num of weights
#     print("n : ", n, "d :", d)
    indices = [] ## indices : vector containing indices of the sampled data
#     print("max w ", max(wts))
    normalized_wts = wts/max(wts)
#     print("normalized weights : ", normalized_wts[0:10])
#     print("normalized weights dim : ", np.shape(normalized_wts))
    target_num_indices = int(n*frac)
#     print("target_num_indices : ", target_num_indices)
#     itr = 0 
    while(len(indices) < target_num_indices): ## Draw samples until have sampled ~25% of samples from D_test
        proposed_indices = np.where(np.random.uniform(size=n) <= normalized_wts)[0].tolist()
#         print("proposed_indices : ", proposed_indices)
        ## If (set of proposed indices that may add is less than or equal to number still needed): then add all of them
#         print("itr : ", itr)
#         itr += 1
        if (len(proposed_indices) <= target_num_indices - len(indices)):
            for j in proposed_indices:
                indices.append(j)
        else: ## Else: Only add the proposed indices that are needed to get to 25% of D_test
            for j in proposed_indices:
                if(len(indices) < target_num_indices):
                    indices.append(j)
    print("unique / total indices : ", len(np.unique(indices)) / len(indices))
    
    return(indices)

def exponential_tilting_indices(x_pca, x, dataset, bias=1):
#     x = np.matrix(x)
    (n, d) = x.shape
#     print("n : ", n, "d :", d)
    
    importance_weights = get_w(x_pca, x, dataset, bias)
#     print()
#     print("importance_weights : ", np.shape(importance_weights))
#     print("L1 squared : ", np.linalg.norm(weights, ord=1)**2)
#     print("L2 : ", np.linalg.norm(weights, ord=2)**2)
#     print("Effective sample size : ", np.linalg.norm(weights, ord=1)**2 / np.linalg.norm(weights, ord=2)**2)
    return wsample(importance_weights, n, d)