import numpy as np
from sklearn import decomposition
from copy import deepcopy
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset
import pdb


## pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 


def get_bike_sharing_data(n_sample=None):
    # fetch dataset 
    bike_sharing_obj = fetch_ucirepo(id=275) 
    
    bike_sharing = bike_sharing_obj.data.features.iloc[:,1:]
    bike_sharing['count'] = bike_sharing_obj.data.targets
    if n_sample is not None:
        bike_sharing = bike_sharing.sample(n=n_sample, random_state=0)
    return bike_sharing

def get_meps_data(n_sample=None):
    meps_data = pd.read_csv('../datasets/meps/meps_data.txt', sep=" ", header=None)
    meps_data.columns = \
    ['AGE53X','EDUCYR','HIDEG','FAMINC16','RTHLTH53','MNHLTH53','NOINSTM','REGION53_-1','REGION53_1','REGION53_2','REGION53_3',\
     'REGION53_4','RACEV2X_1','RACEV2X_2','RACEV2X_3','RACEV2X_4','RACEV2X_5','RACEV2X_6','RACEV2X_10','RACEV2X_12','HISPANX_1',\
     'HISPANX_2','MARRY53X_-1','MARRY53X_1','MARRY53X_2','MARRY53X_3','MARRY53X_4','MARRY53X_5','MARRY53X_6','MARRY53X_7',\
     'MARRY53X_8','MARRY53X_9','MARRY53X_10','ACTDTY53_-1','ACTDTY53_1','ACTDTY53_2','ACTDTY53_3','ACTDTY53_4','HONRDC53_-1',\
     'HONRDC53_1','HONRDC53_2','HONRDC53_3','HONRDC53_4','LANGSPK_-1','LANGSPK_1','LANGSPK_2','FILEDR16_-1','FILEDR16_1',\
     'FILEDR16_2','PREGNT53_-1','PREGNT53_1','PREGNT53_2','WLKLIM53_-1','WLKLIM53_1','WLKLIM53_2','WLKDIF53_-1','WLKDIF53_1',\
     'WLKDIF53_2','WLKDIF53_3','WLKDIF53_4','AIDHLP53_-1','AIDHLP53_1','AIDHLP53_2','SOCLIM53_-1','SOCLIM53_1','SOCLIM53_2',\
     'COGLIM53_-1','COGLIM53_1','COGLIM53_2','WRGLAS42_-1','WRGLAS42_1','WRGLAS42_2','EMPST53_-1','EMPST53_1','EMPST53_2',\
     'EMPST53_3','EMPST53_4','MORJOB53_-1','MORJOB53_1','MORJOB53_2','OCCCT53H_-1','OCCCT53H_1','OCCCT53H_2','OCCCT53H_3',\
     'OCCCT53H_4','OCCCT53H_5','OCCCT53H_6','OCCCT53H_7','OCCCT53H_8','OCCCT53H_9','OCCCT53H_11','INDCT53H_-1','INDCT53H_1',\
     'INDCT53H_2','INDCT53H_3','INDCT53H_4','INDCT53H_5','INDCT53H_6','INDCT53H_7','INDCT53H_8','INDCT53H_9','INDCT53H_10',\
     'INDCT53H_11','INDCT53H_12','INDCT53H_13','INDCT53H_14','INDCT53H_15','UTILIZATION']
    if n_sample is not None:
        meps_data = meps_data.sample(n=n_sample, random_state=0)
    return meps_data

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

def get_superconduct_data(n_sample=None):
    superconduct_data = pd.read_csv(os.getcwd() + '/../datasets/superconduct/train.csv')
    if n_sample is not None:
        superconduct_data = superconduct_data.sample(n=n_sample, random_state=0)
    return superconduct_data

def get_wave_data():
    wave_data = pd.read_csv(
        os.getcwd() + '/../datasets/wave/WECs_DataSet/Sydney_Data.csv', 
        names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'Power_Output']
    )
    wave_data = wave_data.dropna()
    wave_data = wave_data.sample(n=30000, random_state=0)
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


class NpyDataset(Dataset):
    """
    A custom Dataset that loads .npy image data and corresponding labels.
    """
    def __init__(self, images_path, labels_path, transform=None):
        """
        Args:
            images_path (str): Path to the .npy file containing image data.
            labels_path (str): Path to the .npy file containing labels.
            transform (callable, optional): Optional transform to be applied
                on each image.
        """
        self.images = np.load(images_path)     # shape typically (N, H, W) or (N, H, W, C)
        self.targets = np.load(labels_path)     # shape (N,)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    

class NpyCIFAR10CDataset(Dataset):
    """
    A custom dataset for CIFAR-10-C if data is stored in .npy files.
    Expects shape (N, 32, 32, 3) for images and shape (N,) for labels.
    """
    def __init__(self, images_path, labels_path, severity, transform=None):
        super().__init__()
        self.images = np.load(images_path)[(severity - 1)*10000: severity*10000]   # shape: (N, 32, 32, 3)
        self.targets = np.load(labels_path)[(severity - 1)*10000: severity*10000]  # shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]    # shape: (32, 32, 3)
        label = self.targets[idx]

        # Convert to float tensor
        img_tensor = torch.from_numpy(img).float()  # shape: (32, 32, 3)

        # Permute to (C, H, W) => (3, 32, 32)
        img_tensor = img_tensor.permute(2, 0, 1)

        # If there's a transform, apply it (e.g., normalization)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
    

class MixtureDataset(Dataset):
    """
    A dataset that returns a mixture of clean MNIST/CIFAR-10 and corrupted MNIST/CIFAR-10 samples.
    Each time you request an item, it randomly decides whether to return
    from the corrupted set or from the clean set.

    Args:
        clean_dataset (Dataset): A PyTorch dataset for standard MNIST/CIFAR-10.
        corrupted_dataset (Dataset): A PyTorch dataset for MNIST-C/CIFAR-10-C (or any corrupted version).
        mixture_ratio (float): Probability of drawing a corrupted sample (0.0 to 1.0).
        total_size (int, optional): How many total samples to emulate in this mixture.
            If None, defaults to the maximum size of the two input datasets.
        transform (callable, optional): An optional transform to apply on the image.
    """
    def __init__(self, 
                 clean_dataset, 
                 corrupted_dataset,
                 dataset_name, 
                 mixture_ratio=0.5, 
                 total_size=None,
                 transform=None):
        super().__init__()
        self.clean_dataset = clean_dataset
        self.corrupted_dataset = corrupted_dataset
        self.mixture_ratio = mixture_ratio
        self.transform = transform
        self.dataset_name = dataset_name
        
        # By default, we pick the max length so that you can iterate for many samples.
        if total_size is None:
            self.total_size = max(len(self.clean_dataset), len(self.corrupted_dataset))
        else:
            self.total_size = total_size

        # # Inherit indices and targets from the clean dataset
        # self.indices = getattr(clean_dataset, 'indices', None)
        # self.targets = getattr(clean_dataset, 'targets', None)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        """
        Randomly choose either a corrupted sample or a clean sample
        according to the mixture_ratio.
        """
        # Decide: do we pick corrupted or clean?
        if np.random.rand() < self.mixture_ratio:
            # Pick from corrupted dataset
            img, label = self.corrupted_dataset[idx]
        else:
            # Pick from the clean dataset
            img, label = self.clean_dataset[idx]

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def cal_test_mixture(args, fulltrainset, corrupted_dataset):
    train_dataset, val_dataset = random_split(fulltrainset, [len(fulltrainset) - args.val_set_size, args.val_set_size])
    val_dataset_indices = np.array(val_dataset.indices)
    assert args.val_set_size * args.mixture_ratio_val < len(corrupted_dataset) * args.mixture_ratio_test, "Not enough data to sample from"
    
    _, original_val = random_split(val_dataset, [int(args.mixture_ratio_val*len(val_dataset)), len(val_dataset) - int(args.mixture_ratio_val*len(val_dataset))])
    original_val_indices = np.array(original_val.indices)
    
    mix_test, _ = random_split(train_dataset, [int(args.mixture_ratio_test*len(corrupted_dataset)), len(train_dataset) - int(args.mixture_ratio_test*len(corrupted_dataset))])
    remaining_test, original_test = random_split(corrupted_dataset, [int(args.mixture_ratio_test*len(corrupted_dataset)), len(corrupted_dataset) - int(args.mixture_ratio_test*len(corrupted_dataset))])
    original_test_indices = np.array(original_test.indices)
    remaining_test_indices = np.array(remaining_test.indices)

    mix_val, _ = random_split(remaining_test, [int(args.mixture_ratio_val*len(val_dataset)), len(remaining_test_indices) - int(args.mixture_ratio_val*len(val_dataset))])
    mix_val_indices = np.array(mix_val.indices)

    test_split = len(original_test) - args.init_phase
    test_w_est, rest_test_dataset = random_split(original_test, [args.init_phase, test_split])
    test_w_est_indices = np.array(test_w_est.indices)
    
    if len(fulltrainset) == 60000:
        fulltrainset.targets[val_dataset_indices[original_val_indices]] = torch.zeros(len(original_val_indices)).long()
        corrupted_dataset.targets[remaining_test_indices[mix_val_indices]] = torch.zeros(len(mix_val_indices)).long()
        corrupted_dataset.targets[original_test_indices[test_w_est_indices]] = torch.ones(len(test_w_est_indices)).long()
    else:
        for idx in val_dataset_indices[original_val_indices]:
            fulltrainset.targets[idx] = 0
        for idx in remaining_test_indices[mix_val_indices]:
            corrupted_dataset.targets[idx] = 0
        for idx in original_test_indices[test_w_est_indices]:
            corrupted_dataset.targets[idx] = 1

    cal_test_w_est_dataset = ConcatDataset([original_val, mix_val, test_w_est])
    cal_test_w_est_loader = DataLoader(
        dataset=cal_test_w_est_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False
    )
    test_dataset = ConcatDataset([rest_test_dataset, mix_test])
    test_loader_mixed = DataLoader(
        dataset=test_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False
    )
    return cal_test_w_est_loader, test_loader_mixed


def get_mnist_data(args):
    """
    Load the MNIST dataset with optional normalization.

    Args:
        batch_size (int): Number of samples per batch in the DataLoader.
        normalize (bool): Whether to apply standard MNIST normalization.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Common MNIST mean/std
    ])

    # Load train & test sets
    full_train_dataset = torchvision.datasets.MNIST(
        root=os.path.join(os.path.dirname(os.getcwd()), 'data'),       # Directory to store the MNIST data
        train=True,
        transform=transform #,
        #download=True
    )
    train_dataset, val_dataset = random_split(full_train_dataset, [60000 - args.val_set_size, args.val_set_size])
    full_test_dataset = torchvision.datasets.MNIST(
        root=os.path.join(os.path.dirname(os.getcwd()), 'data'),
        train=False,
        transform=transform #,
        #download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    if args.train_val_test_split_only:
        test_loader = DataLoader(
            full_test_dataset,
            batch_size=args.bs,
            shuffle=True
        )
        return train_loader, val_loader, test_loader
    else:
        test_split = len(full_test_dataset) - args.init_phase
        test_w_est, test_dataset = random_split(full_test_dataset, [args.init_phase, test_split])
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.bs,
            shuffle=True
        )
        
        ## Label val vs test data for like-ratio estimation
        val_indices = val_dataset.indices
        test_indices = test_w_est.indices
        val_dataset.dataset.targets[val_indices] = torch.zeros(len(val_indices)).long() ## Source data
        test_w_est.dataset.targets[test_indices] = torch.ones(len(test_indices)).long() ## Target data
        cal_test_w_est = ConcatDataset([val_dataset, test_w_est])
        #cal_test_w_est.transform=transform

        cal_test_w_est_loader = DataLoader(
            cal_test_w_est,
            batch_size=args.bs,
            shuffle=True
        )
        return train_loader, cal_test_w_est_loader, test_loader


def get_mnist_c_data(args):
    mnist_c_path = os.path.join(os.path.dirname(os.getcwd()), 'data/mnist_c', args.corruption_type)

    # Define a transform to convert the images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    corrupted_dataset = NpyDataset(os.path.join(mnist_c_path, 'test_images.npy'), 
                        os.path.join(mnist_c_path, 'test_labels.npy'), transform=transform)
    if args.train_val_test_split_only:
        test_loader_c = DataLoader(corrupted_dataset, batch_size=args.bs, shuffle=True)
        return test_loader_c
    else:
        full_train_dataset = torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(os.getcwd()), 'data'),
            train=True,
            transform=transform,
            download=True
        )
        return cal_test_mixture(args, full_train_dataset, corrupted_dataset)


def get_cifar10_data(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    fulltrainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    fulltestset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    train_dataset, val_dataset = random_split(fulltrainset, [50000 - args.val_set_size, args.val_set_size])
    trainloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    if args.train_val_test_split_only:
        testloader = DataLoader(fulltestset, batch_size=args.bs, shuffle=True, num_workers=2)
        return trainloader, valloader, testloader
    else:
        test_split = len(fulltestset) - args.init_phase
        test_w_est, testset = random_split(fulltestset, [args.init_phase, test_split])
        testloader = DataLoader(testset, batch_size=args.bs, shuffle=True, num_workers=2)

        val_indices = val_dataset.indices
        test_indices = test_w_est.indices
        for i in val_indices:
            val_dataset.dataset.targets[i] = 0
        for i in test_indices:
            test_w_est.dataset.targets[i] = 1

        cal_test_w_est = ConcatDataset([val_dataset, test_w_est])
        cal_test_w_est_loader = DataLoader(
            dataset=cal_test_w_est,
            batch_size=args.bs,
            shuffle=True,
            drop_last=False
        )
        return trainloader, cal_test_w_est_loader, testloader


def get_cifar10_c_data(args):
    """
    Create a DataLoader for CIFAR-10-C (single corruption or combined),
    assuming .npy files for images and labels.
    """
    transform_clean = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform_corrupted = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    corrupted_dataset = NpyCIFAR10CDataset(
        images_path=os.path.join(os.path.dirname(os.getcwd()), f'data/CIFAR-10-C/{args.corruption_type}.npy'),
        labels_path=os.path.join(os.path.dirname(os.getcwd()), 'data/CIFAR-10-C/labels.npy'),
        severity=args.severity,
        transform=transform_corrupted
    )
    if args.train_val_test_split_only:
        loader = DataLoader(corrupted_dataset, batch_size=args.bs, shuffle=True)
        return loader
    else:
        fulltrainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_clean)
        return cal_test_mixture(args, fulltrainset, corrupted_dataset)


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
        return cal_test_vals_mat[-1]
        
    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(np.array(cal_test_vals_mat[-1])).flatten()
    idx_include = np.repeat(True, n_cal_test)
    
    for i in range(n_cal_test):
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
        x_red_scaled = (x_red - np.min(x_red, axis=0)) / (np.max(x_red, axis=0) - np.min(x_red, axis=0))
        return np.exp(x_red_scaled @ [-bias,bias])
    
        ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['superconduct']):
#         np.random.seed(5)
        pca = decomposition.PCA(n_components=1)
        pca.fit(x_pca)
        x_red = pca.transform(x)
        x_red_scaled = (x_red - np.min(x_red, axis=0)) / (np.max(x_red, axis=0) - np.min(x_red, axis=0))
        return np.exp(x_red_scaled @ [bias])
    
        ## For communities dataset use top 2 PCs as tilting vars
    elif (dataset in ['communities', 'meps']): #, 'meps'
        x_sub = x[:,[0,1,2]] ## 0:=AGE53X (age), 1:= EDUCYR (yrs education), 2:= HIDEG (yrs education)
        x_sub = x_sub / np.max(x[:,[0,1,2]])
        return np.exp(x_sub @ [-bias,bias,bias])
# #         np.random.seed(5)
#         pca = decomposition.PCA(n_components=2)
#         pca.fit(x_pca)
#         x_red = pca.transform(x)
#         x_red_scaled = (x_red - np.min(x_red, axis=0)) / (np.max(x_red, axis=0) - np.min(x_red, axis=0))
#         return np.exp(x_red_scaled @ [-bias,bias])
    
    
    

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


def split_into_folds(dataset0_train, num_folds, seed=0):
    y_name = dataset0_train.columns[-1] ## Outcome column must be last in dataframe
    X = dataset0_train.drop(y_name, axis=1).to_numpy()
    y = dataset0_train[y_name].to_numpy()
    n = len(X)
    all_inds = np.arange(n)
    
    if (num_folds == 1):
        train_indices = np.random.choice(all_inds, int(n/2), replace=False)
        cal_indices = np.setdiff1d(all_inds, train_indices)
        folds = [[train_indices, cal_indices]] ## Same structure as KFold: List length 1, entry is (train_indices, cal_indices)
        
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
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
    noise_sigma=0,
    num_test_unshifted=500
):
    
    dataset0_train, dataset0_test_0 = train_test_split(dataset0, test_size=test0_size, shuffle=True, random_state=seed)
    dataset0_train = dataset0_train.reset_index(drop=True).astype(float)
    dataset0_test_0 = dataset0_test_0.reset_index(drop=True).astype(float)
    
    dataset0_test_0_unshifted_idx = np.arange(num_test_unshifted) ## indices of unshifted test points (before changepoint)
    dataset0_test_0_post_change = dataset0_test_0.iloc[num_test_unshifted:] ## test candiate pts excluding unshifted idx
    ## Note: idx in "dataset0_test_0" == num_test_unshifted + idx in "dataset0_test_0_post_change"
    
    
    dataset0_test_0_post_change = dataset0_test_0_post_change.reset_index(drop=True)

    if (dataset0_shift_type == 'none'):
        ## No shift within dataset0    
        return dataset0_train, dataset0_test_0
    
    
    elif (dataset0_shift_type == 'covariate'):
        ## Covariate shift within dataset0
        
        
        dataset0_train_copy = dataset0_train.copy()
        X_train = dataset0_train_copy.iloc[:, :-1].values
        dataset0_test_0_post_change = dataset0_test_0_post_change.copy()
        X_test_0 = dataset0_test_0_post_change.iloc[:, :-1].values
        
        ## Get indices of biased test samples post changepoint (indices relative to dataset0_test_0 by adding num_test_unshifted)
        dataset0_test_0_biased_idx = num_test_unshifted + exponential_tilting_indices(x_pca=X_train, x=X_test_0, dataset=dataset0_name, bias=cov_shift_bias)
        
        dataset0_test_0_idx = np.concatenate((dataset0_test_0_unshifted_idx, dataset0_test_0_biased_idx))
        
        return dataset0_train, dataset0_test_0.iloc[dataset0_test_0_idx]
    
    
    elif (dataset0_shift_type == 'label'):
        ## Label shift within dataset0

        if 'wine' in dataset0_name:
            # Define a threshold for 'alcohol' to identify high alcohol content wines
            alcohol_threshold = dataset0_test_0['alcohol'].quantile(label_uptick)
            # Increase the quality score by a number for wines with alcohol above the threshold
            indices_to_shift = num_test_unshifted + np.where(dataset0_test_0_post_change['alcohol'] > alcohol_threshold)[0]
            dataset0_test_0.loc[indices_to_shift, 'quality'] += 1
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
            indices_to_shift = num_test_unshifted + np.where(dataset0_test_0_post_change['mean_ElectronAffinity'] > ea_threshold)[0]
            dataset0_test_0.loc[indices_to_shift, 'critical_temp'] *= label_uptick #2 #1.1
#             dataset0_test_0['critical_temp'] = dataset0_test_0['critical_temp'].clip(lower=0, upper=200)
            
        elif 'wave' in dataset0_name:
            x_mean_threshold = dataset0_test_0['X1'].median()
            # Increase Power_Output by 15% for instances where X_mean is above the threshold
            dataset0_test_0.loc[dataset0_test_0['X1'] > x_mean_threshold, 'Power_Output'] *= 1.15
            dataset0_test_0['Power_Output'] = dataset0_test_0['Power_Output'].clip(lower=0)
            
        elif 'bike_sharing' in dataset0_name:
#             ## For 25% coldest of days, increase number of bike rentals by 10%:
            temp_threshold = dataset0_test_0['temp'].quantile(0.25) 
            indices_to_shift = num_test_unshifted + np.where(dataset0_test_0_post_change['temp'] > temp_threshold)[0]
            dataset0_test_0.loc[indices_to_shift, 'count'] *= label_uptick #2
#             temp_mean = dataset0_test_0['temp'].mean() 
# #             indices_to_shift = num_test_unshifted + np.where(dataset0_test_0_post_change['temp'] > temp_threshold)[0]
#             dataset0_test_0.loc[num_test_unshifted:, 'count'] = dataset0_test_0.loc[num_test_unshifted:, 'count']**2 / temp_mean #2

        elif 'meps' in dataset0_name:
            age_threshold = dataset0_test_0['AGE53X'].quantile(0.25) 
            indices_to_shift = num_test_unshifted + np.where(dataset0_test_0_post_change['AGE53X'] > age_threshold)[0]
            dataset0_test_0.loc[indices_to_shift, 'UTILIZATION'] *= label_uptick #1.5
            

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
    
    
    
def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]
    
    return [v_sorted, w_sorted]
    

def weighted_quantile(v, w_normalized, q):
    if (len(v) != len(w_normalized)):
        raise ValueError('Error: v is length ' + str(len(v)) + ', but w_normalized is length ' + str(len(w_normalized)))
        
    if (np.sum(w_normalized) > 1.01 or np.sum(w_normalized) < 0.99):
        raise ValueError('Error: w_normalized does not add to 1')
        
    if (q < 0 or 1 < q):
        raise ValueError('Error: Invalid q')

    n = len(v)
    
    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)
    
    cum_w_sum = w_sorted[0]
    i = 0
    while(cum_w_sum <= q):
            i += 1
            cum_w_sum += w_sorted[i]
            
    if (q > 0.5): ## If taking upper quantile: ceil
        return v_sorted[i]
            
    elif (q < 0.5): ## Elif taking lower quantile:
        if (i > 0):
            return v_sorted[i-1]
        else:
            return v_sorted[0]
        
    else: ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if (cum_w_sum == 0.5):
            return v_sorted[i]
        
        elif (i > 0):
            return (v_sorted[i]*w_sorted[i] + v_sorted[i-1]*w_sorted[i-1]) / (w_sorted[i] + w_sorted[i-1])
        
        else:
            return v_sorted[0]
