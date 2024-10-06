import numpy as np
from sklearn import decomposition
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
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


def get_w(x_pca, x, dataset, bias):
    
    if (dataset=='airfoil'):
        return np.exp(x[:,[0,4]] @ [-bias,bias])
    
    elif(dataset == 'white_wine'):
        return np.exp(x[:,[0,10]] @ [-bias,bias])
    
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
#     print("importance_weights : ", importance_weights)
#     print("L1 squared : ", np.linalg.norm(weights, ord=1)**2)
#     print("L2 : ", np.linalg.norm(weights, ord=2)**2)
#     print("Effective sample size : ", np.linalg.norm(weights, ord=1)**2 / np.linalg.norm(weights, ord=2)**2)
    return wsample(importance_weights, n, d)


def split_into_folds(dataset0_train, seed=0):
    y_name = dataset0_train.columns[-1] ## Outcome column must be last in dataframe
    X = dataset0_train.drop(y_name, axis=1).to_numpy()
    y = dataset0_train[y_name].to_numpy()
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(kf.split(X, y))
    return X, y, folds


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
        dataset0_test_0_copy = dataset0_test_0.copy()
        X_test_0 = dataset0_test_0_copy.iloc[:, :-1].values
        
        dataset0_test_0_biased_idx = exponential_tilting_indices(x_pca=X_train, x=X_test_0, dataset=dataset0_name, bias=cov_shift_bias)
        
        return dataset0_train, dataset0_test_0.iloc[dataset0_test_0_biased_idx]
    
    elif (dataset0_shift_type == 'label'):
        ## Label shift within dataset0

        if 'wine' in dataset0_name:
            # Define a threshold for 'alcohol' to identify high alcohol content wines
            alcohol_threshold = dataset0_test_0['alcohol'].quantile(label_uptick)
            # Increase the quality score by a number for wines with alcohol above the threshold
            dataset0_test_0.loc[dataset0_test_0['alcohol'] > alcohol_threshold, 'quality'] += 1
            dataset0_test_0['quality'] = dataset0_test_0['quality'].clip(lower=0, upper=10)
        elif 'airfoil' in dataset0_name:
            velocity_threshold = dataset0_test_0['Velocity'].median()
            dataset0_test_0.loc[dataset0_test_0['Velocity'] > velocity_threshold, 'Sound'] += 3

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

        elif 'airfoil' in dataset0_name:
            # Compute the median of Scaled sound pressure level
            spl_median = dataset0_test_0['Sound'].median()
            dataset0_test_0['Suction'] += np.where(
                dataset0_test_0['Sound'] >= spl_median,
                # Add positive noise for higher sound pressure levels
                np.random.normal(loc=0.0001, scale=0.00005, size=len(dataset0_test_0)),
                # Subtract noise for lower sound pressure levels
                np.random.normal(loc=-0.0001, scale=0.00005, size=len(dataset0_test_0))
            )
            # Ensure 'Suction_side_displacement_thickness' remains within valid range
            min_value = data_before_shift['Suction'].min()
            max_value = data_before_shift['Suction'].max()
            dataset0_test_0['Suction'] = dataset0_test_0['Suction'].clip(lower=min_value, upper=max_value)
        
        return dataset0_train, dataset0_test_0
