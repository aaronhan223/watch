import numpy as np
from sklearn import decomposition



def get_w(x_pca, x, dataset, bias):
    
    if (dataset=='airfoil'):
        return np.exp(x[:,[0,4]] @ [-bias,bias])
    
    elif(dataset == 'wine'):
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
    indices = [] ## indices : vector containing indices of the sampled data
    normalized_wts = wts/max(wts)
#     print(normalized_wts[0:20])
    target_num_indices = int(n*frac)
    while(len(indices) < target_num_indices): ## Draw samples until have sampled ~25% of samples from D_test
        proposed_indices = np.where(np.random.uniform(size=n) <= normalized_wts)[0].tolist()
        ## If (set of proposed indices that may add is less than or equal to number still needed): then add all of them
        if (len(proposed_indices) <= target_num_indices - len(indices)):
            for j in proposed_indices:
                indices.append(j)
        else: ## Else: Only add the proposed indices that are needed to get to 25% of D_test
            for j in proposed_indices:
                if(len(indices) < target_num_indices):
                    indices.append(j)
    return(indices)

def exponential_tilting_indices(x_pca, x, dataset, bias=1):
    (n, d) = x.shape
    importance_weights = get_w(x_pca, x, dataset, bias)
#     print("L1 squared : ", np.linalg.norm(weights, ord=1)**2)
#     print("L2 : ", np.linalg.norm(weights, ord=2)**2)
#     print("Effective sample size : ", np.linalg.norm(weights, ord=1)**2 / np.linalg.norm(weights, ord=2)**2)
    return wsample(importance_weights, n, d)