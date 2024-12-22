import numpy as np
from sklearn import decomposition
from copy import deepcopy
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
import os
import pdb


## pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 


def get_bike_sharing_data():
    # fetch dataset 
    bike_sharing_obj = fetch_ucirepo(id=275) 
    
    bike_sharing = bike_sharing_obj.data.features.iloc[:,1:]
    bike_sharing['count'] = bike_sharing_obj.data.targets
    return bike_sharing

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

def get_communities_data():
    column_names = []
    with open(os.getcwd() + '/../datasets/communities/communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    column_name = parts[1]
                    column_names.append(column_name)
    df = pd.read_csv(os.getcwd() + '/../datasets/communities/communities.data', header=None, names=column_names, na_values='?')
    non_predictive_cols = [
        'state', 'county', 'community', 'communityname', 'fold'
    ]
    df = df.drop(columns=non_predictive_cols)
    df = df.dropna(axis=1)
    # Ensure all columns are numeric
    communities_data = df.apply(pd.to_numeric)
    print("\nShape of the communities_data after preprocessing:", communities_data.shape)
    return communities_data

def get_superconduct_data():
    superconduct_data = pd.read_csv(os.getcwd() + '/../datasets/superconduct/train.csv')
    return superconduct_data


def get_wave_data():
    wave_data = pd.read_csv(
        os.getcwd() + '/../datasets/wave/WECs_DataSet/Sydney_Data.csv', 
        names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'Power_Output']
    )
    wave_data = wave_data.dropna()
    return wave_data

def get_1dim_synthetic_data(size=10000):
    high=2*np.pi
    X = np.random.uniform(low=0, high=high, size=size)
    Y = np.zeros(size)
    for i in range(0, size):
        Y[i] = np.random.normal(np.sin(X[i]), (X[i]+1)/10)

    return pd.DataFrame(np.c_[X, Y])



def get_1dim_synthetic_v2_data(size=1000):
    high=2*np.pi
    X = np.random.uniform(low=-np.pi/2, high=high, size=size)
    Y = np.zeros(size)
    for i in range(0, size):
        if (X[i] >= 0):
            Y[i] = np.random.normal(np.sin(X[i]), np.abs(X[i]+1)/10)
        else:
            Y[i] = np.random.normal(-3*np.sin(X[i]**3), np.abs(X[i])/10)

    return pd.DataFrame(np.c_[X, Y])


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
    
    
    elif (dataset == 'bike_sharing'):
        x_sub = x[:,[8,11]] ## 8:temp, 11:windspeed
        x_sub = x_sub / np.max(x[:,[8,11]]) ## normalize
        return np.exp(x_sub @ [-bias,bias])

    
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
    

# def wsample(wts, n, d, frac=0.1):
#     n = len(wts) ## n : length or num of weights
# #     print("n : ", n, "d :", d)
#     indices = [] ## indices : vector containing indices of the sampled data
# #     print("max w ", max(wts))
#     normalized_wts = wts/max(wts)
# #     print("normalized weights : ", normalized_wts[0:10])
# #     print("normalized weights dim : ", np.shape(normalized_wts))
#     target_num_indices = int(n*frac)
# #     print("target_num_indices : ", target_num_indices)
# #     itr = 0 
#     while(len(indices) < target_num_indices): ## Draw samples until have sampled ~25% of samples from D_test
#         proposed_indices = np.where(np.random.uniform(size=n) <= normalized_wts)[0].tolist()
# #         print("proposed_indices : ", proposed_indices)
#         ## If (set of proposed indices that may add is less than or equal to number still needed): then add all of them
# #         print("itr : ", itr)
# #         itr += 1
#         if (len(proposed_indices) <= target_num_indices - len(indices)):
#             for j in proposed_indices:
#                 indices.append(j)
#         else: ## Else: Only add the proposed indices that are needed to get to 25% of D_test
#             for j in proposed_indices:
#                 if(len(indices) < target_num_indices):
#                     indices.append(j)
# #     print("unique / total indices : ", len(np.unique(indices)) / len(indices))
    
#     return(indices)

def wsample(wts, n, d, frac=0.5):
    n = len(wts) ## n : length or num of weights
    indices_all = np.arange(0, n)
    normalized_wts = wts/np.sum(wts)
    target_num_indices = int(n*frac)
#     np.random.seed(seed=0) ## Added this 20241220
    indices = np.random.choice(indices_all, size=target_num_indices, p=normalized_wts)
#     print(np.shape(indices))
#     print(indices)
    return indices


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
        elif 'communities' in dataset0_name:
            youth_threshold = dataset0_test_0['agePct12t29'].median()
            # Increase ViolentCrimesPerPop by 20% for communities where agePct12t29 is above the median
            dataset0_test_0.loc[dataset0_test_0['agePct12t29'] > youth_threshold, 'ViolentCrimesPerPop'] *= 1.2
            dataset0_test_0['ViolentCrimesPerPop'] = dataset0_test_0['ViolentCrimesPerPop'].clip(lower=0, upper=1)
        elif 'superconduct' in dataset0_name:
            ea_threshold = dataset0_test_0['mean_ElectronAffinity'].quantile(0.75)
            # Increase critical_temp by 10% for materials where oxygen content is above the threshold
            dataset0_test_0.loc[dataset0_test_0['mean_ElectronAffinity'] > ea_threshold, 'critical_temp'] *= 1.1
            dataset0_test_0['critical_temp'] = dataset0_test_0['critical_temp'].clip(lower=0, upper=200)
        elif 'wave' in dataset0_name:
            x_mean_threshold = dataset0_test_0['X1'].median()
            # Increase Power_Output by 15% for instances where X_mean is above the threshold
            dataset0_test_0.loc[dataset0_test_0['X1'] > x_mean_threshold, 'Power_Output'] *= 1.15
            dataset0_test_0['Power_Output'] = dataset0_test_0['Power_Output'].clip(lower=0)
            
        elif 'bike_sharing' in dataset0_name:
            ## For 25% coldest of days, increase number of bike rentals by 10%:
            temp_threshold = dataset0_test_0['temp'].quantile(0.25) 
            dataset0_test_0.loc[dataset0_test_0['temp'] < temp_threshold, 'count'] *= 2
            

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
        
        elif 'communities' in dataset0_name:
            crime_median = dataset0_test_0['ViolentCrimesPerPop'].median()

            # Add noise to 'PctWorkMom' (percentage of moms in workforce)
            dataset0_test_0['PctWorkMom'] += np.where(
                dataset0_test_0['ViolentCrimesPerPop'] >= crime_median,
                # Add positive noise for higher crime rates
                np.random.normal(loc=1.0, scale=0.5, size=len(dataset0_test_0)),
                # Subtract noise for lower crime rates
                np.random.normal(loc=-1.0, scale=0.5, size=len(dataset0_test_0))
            )

            # Ensure 'PctWorkMom' remains within valid range (e.g., 0 to 100)
            dataset0_test_0['PctWorkMom'] = dataset0_test_0['PctWorkMom'].clip(lower=0, upper=100)

        elif 'superconduct' in dataset0_name:
            temp_median = dataset0_test_0['critical_temp'].median()
            # Add noise to 'mean_atomic_mass' based on 'critical_temp'
            dataset0_test_0['mean_atomic_mass'] += np.where(
                dataset0_test_0['critical_temp'] >= temp_median,
                # Add positive noise for higher critical temperatures
                np.random.normal(loc=5.0, scale=2.0, size=len(dataset0_test_0)),
                # Subtract noise for lower critical temperatures
                np.random.normal(loc=-5.0, scale=2.0, size=len(dataset0_test_0))
            )
            dataset0_test_0['mean_atomic_mass'] = dataset0_test_0['mean_atomic_mass'].clip(lower=0)

        elif 'wave' in dataset0_name:
            power_median = dataset0_test_0['Power_Output'].median()
            # Add noise to 'Y_mean' based on 'Power_Output'
            dataset0_test_0['Y1'] += np.where(
                dataset0_test_0['Power_Output'] >= power_median,
                # Add positive noise for higher Power_Output
                np.random.normal(loc=0.5, scale=0.2, size=len(dataset0_test_0)),
                # Subtract noise for lower Power_Output
                np.random.normal(loc=-0.5, scale=0.2, size=len(dataset0_test_0))
            )
            dataset0_test_0['Y1'] = dataset0_test_0['Y1'].clip(lower=data_before_shift['Y1'].min(), upper=data_before_shift['Y1'].max())

        return dataset0_train, dataset0_test_0
