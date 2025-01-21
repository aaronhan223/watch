import numpy as np
import pandas as pd
from plot import plot_martingale_paths
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import pdb

from utils import *
from martingales import *
from p_values import *
import argparse
import random


def set_seed(seed: int):
    """
    Set the random seed for PyTorch, NumPy, and Python 'random' library.
    This helps to ensure reproducible results in experiments.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    """
    A simple 3-layer MLP for MNIST/CIFAR-10 classification.
    Input: (N, 1, 28, 28)
    Output: (N, 10) for 10 classes (digits 0..9)
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (N, 1, 28, 28)
        # Flatten to (N, 784)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train_one_epoch(model, device, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Compute accuracy
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, device, data_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_loss_prob(model, device, setting, loader_0, loader_1, binary_classifier_probs = False):
    model.load_state_dict(torch.load(os.getcwd() + '/../pkl_files/best_model_' + setting + '.pth'))
    model.eval()
    all_preds = []
    all_losses = []

    with torch.no_grad():
        # Evaluate on validation loader
        for images, labels in loader_0:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            if binary_classifier_probs:
                ## If want probs for lik ratio estimation (binary classifier prob est)
                source_target_labels = torch.ones(len(labels), dtype=torch.int64).to(device)
                correct_class_probs = probabilities.gather(1, source_target_labels.view(-1, 1)).squeeze()
                
            else:
                ## Default case for class probs (not for binary classification for lik ratio est)
                correct_class_probs = probabilities.gather(1, labels.view(-1, 1)).squeeze()
            all_preds.extend(correct_class_probs.cpu().numpy())
            
            # Calculate cross entropy loss for each sample
            loss = F.cross_entropy(outputs, labels, reduction='none')
            all_losses.append(loss.cpu().numpy())

        # Evaluate on test loader
        for images, labels in loader_1:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            if binary_classifier_probs:
                ## If want probs for lik ratio estimation (binary classifier prob est)
                source_target_labels = torch.ones(len(labels), dtype=torch.int64).to(device)
                correct_class_probs = probabilities.gather(1, source_target_labels.view(-1, 1)).squeeze()
            else:
                ## Default case for class probs (not for binary classification for lik ratio est)
                correct_class_probs = probabilities.gather(1, labels.view(-1, 1)).squeeze()
            all_preds.extend(correct_class_probs.cpu().numpy())
            
            # Calculate cross entropy loss for each sample
            loss = F.cross_entropy(outputs, labels, reduction='none')
            all_losses.append(loss.cpu().numpy())
        all_losses = np.concatenate(all_losses)
    
    return np.array(all_preds), all_losses



def fit(model, epochs, train_loader, val_loader_0, test_loader, optimizer, setting, device):
    """
    Train the model on the training set.
    """
#     best_clean_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

#         # Evaluate on both clean and corrupted to see performance
#         clean_loss, clean_acc = evaluate(model, device, test_loader)
#         print(f"   Clean Test Loss: {clean_loss:.4f}, Clean Acc: {clean_acc*100:.2f}%")

#         # Save checkpoint if clean accuracy improves
#         if clean_acc > best_clean_acc:
#         best_clean_acc = clean_acc
    torch.save(model.state_dict(), os.getcwd() + '/../pkl_files/best_model_' + setting + '.pth')
#         print("Checkpoint saved for model with clean accuracy: {:.2f}%".format(best_clean_acc * 100))




def train_and_evaluate(train_loader_0, val_loader_0, test_loader_0, dataset0_name, epochs, device, lr, setting, loader_1=None,
                       cal_test_w_est_loader_0=None, cal_test_w_est_loader_1=None, val_loader_mixed=None, test_loader_mixed=None, verbose=False, 
                       methods=['baseline'], init_phase=500, epsilon=1e-9):
    '''
    baseline uniform weights:
    - train on clean data, eval on clean data: train_loader_0 + val_loader_0 + test_loader_0
    - train on clean data, eval on corrupted data: train_loader_0 + val_loader_0 + loader_1
    WCTMs:
    - train on clean data, eval on clean data: train_loader_0 + val_loader_0 + test_loader_0
    - train on clean data, eval on corrupted data: train_loader_0 + val_loader_mixed + test_loader_mixed
    '''
    cs_0 = []
    cs_1 = []
    W_0_dict = {}
    W_1_dict = {}

    #for images, labels in train_loader_0:
    #    print("train loader[0] shape : ", np.shape(images))
    #    print("train loader[1] shape : ", np.shape(labels))

    # Train the model on the training set proper
    if dataset0_name == 'mnist':
        model = MLP(input_size=784, hidden_size=256, num_classes=10).to(device)
    elif dataset0_name == 'cifar10':
        model = MLP(input_size=3*32*32, hidden_size=1024, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    fit(model, epochs, train_loader_0, val_loader_0, test_loader_0, optimizer, setting, device)
    
    clean_pred, clean_loss = eval_loss_prob(model, device, setting, val_loader_0, test_loader_0)
    if loader_1 is not None:
        # CTMs
        corrupt_pred, corrupt_loss = eval_loss_prob(model, device, setting, val_loader_0, loader_1)
    else:
        # WCTMs
        ## TODO: 
        breakpoint()
        corrupt_pred, corrupt_loss = eval_loss_prob(model, device, setting, val_loader_mixed, test_loader_mixed)

    ### val + test conformity scores: cs_0 is clean val + test, cs_1 is corrupted val + test
    if cs_type == 'probability':
        cs_0 = 1 - clean_pred
        cs_1 = 1 - corrupt_pred
    elif cs_type == 'neg_log':
        cs_0 = -np.log(clean_pred + epsilon)
        cs_1 = -np.log(corrupt_pred + epsilon)
    #pdb.set_trace()
    
    #### Computing (unnormalized) weights
    # TODO: @Drew, I removed the weight computation module since it requires special design for image data
    # the implementation above is for regular CTMs, and the cs computation for WCTMs
    # val_loader_mixed and test_loader_mixed are validation and test set that are mixed with certain ratio of corrupted data
    # test_w_est_0 and test_w_est_1 are test data used to initialize the density ratio estimator from clean and corrupted dataset
    
    ### NOTE: 'fixed_cal_offline' is the primary method implemented for now.
    for method in methods:

        if (method in ['fixed_cal_offline']):
            ###
            print("device : ", device)
            W_0_dict[method] = offline_lik_ratio_estimates_images(cal_test_w_est_loader_0, val_loader_0, test_loader_0, dataset0_name, device=device, setting=setting)
            W_1_dict[method] = offline_lik_ratio_estimates_images(cal_test_w_est_loader_1, val_loader_mixed, test_loader_mixed, dataset0_name, device=device, setting=setting)

        elif (method in ['fixed_cal', 'one_step_est']):
            ## Estimating likelihood ratios for each cal, test point
            ## np.shape(W_i) = (T, n_cal + T)
            raise Exception("Method not yet implemented")
#             W_i = online_lik_ratio_estimates_images(X_cal, X_test_w_est, X_test_0_only, adapt_start=n_cal)

        elif (method in ['fixed_cal_dyn']):
            ## fixed_cal except with dynamically/automatically determined start to adaptation
#             W_i = online_lik_ratio_estimates_images(X_cal, X_test_w_est, X_test_0_only, adapt_start=adapt_starts[i])
            raise Exception("Method not yet implemented")


        elif (method in ['fixed_cal_oracle']):
            ## Oracle one-step likelihood ratios
            ## np.shape(W_i) = (n_cal + T, )
            raise Exception("Method not yet implemented")

#             X_full = np.concatenate((X_train, X_cal_test_0), axis = 0)

#             W_i = get_w(x_pca=X_train, x=X_cal_test_0, dataset=dataset0_name, bias=cov_shift_bias) 

        else:
            ## Else: Unweighted / uniform-weighted CTM
            W_0_dict[method] = np.ones(len(cal_test_w_est_loader_0.dataset))
            W_1_dict[method] = np.ones(len(cal_test_w_est_loader_1.dataset))



    return cs_0, cs_1, clean_loss, corrupt_loss, W_0_dict, W_1_dict


def retrain_count(conformity_score, training_schedule, sr_threshold=1e6, cu_confidence=0.99, W=None, n_cal=None, verbose=False, method='baseline'):
    p_values = calculate_p_values(conformity_score)
    
    if (method in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle', 'fixed_cal_offline']):
        p_values = calculate_weighted_p_values(conformity_score, W, n_cal, method)
    
    retrain_m, martingale_value = composite_jumper_martingale(p_values, verbose=verbose)

    if training_schedule == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
        
    elif (training_schedule == 'basic'):
        print("plotting martingale (wealth) values directly")
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, sr_threshold, verbose)
        sigma = martingale_value
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence, verbose)
    return retrain_m, retrain_s, martingale_value, sigma, p_values


def training_function(train_loader_0, val_loader_0, test_loader_0, dataset0_name, epochs, device, lr, setting, loader_1=None,
                      schedule='variable', cal_test_w_est_loader_0=None, cal_test_w_est_loader_1=None, val_loader_mixed=None, test_loader_mixed=None, 
                      verbose=False, methods=['baseline'], init_phase=500):
    
    if cal_test_w_est_loader_0 is None:
        cs_0, cs_1, test_loss, corrupt_loss = train_and_evaluate(
            train_loader_0=train_loader_0,
            val_loader_0=val_loader_0,
            test_loader_0=test_loader_0,
            dataset0_name=dataset0_name,
            loader_1=loader_1,
            epochs=epochs,
            device=device,
            lr=lr,
            setting=setting,
            verbose=verbose,
            methods=methods
        )
    else:
        # should return weights here
        cs_0, cs_1, clean_loss, corrupt_loss, W_0_dict, W_1_dict = train_and_evaluate(
            train_loader_0=train_loader_0,
            val_loader_0=val_loader_0,
            test_loader_0=test_loader_0,
            cal_test_w_est_loader_0=cal_test_w_est_loader_0,
            cal_test_w_est_loader_1=cal_test_w_est_loader_1,
            val_loader_mixed=val_loader_mixed,
            test_loader_mixed=test_loader_mixed,
            dataset0_name=dataset0_name,
            epochs=epochs,
            device=device,
            lr=lr,
            setting=setting,
            verbose=verbose,
            methods=methods,
            init_phase=init_phase
        )

    martingales_0_dict, martingales_1_dict = {}, {}
    sigmas_0_dict, sigmas_1_dict = {}, {}
    retrain_m_count_0_dict, retrain_s_count_0_dict = {}, {}
    retrain_m_count_1_dict, retrain_s_count_1_dict = {}, {}
    p_values_0_dict = {}
    coverage_0_dict = {}
    
    for method in methods:
        martingales_0_dict[method], martingales_1_dict[method] = [], []
        sigmas_0_dict[method], sigmas_1_dict[method] = [], []
        retrain_m_count_0_dict[method], retrain_s_count_0_dict[method] = [], []
        retrain_m_count_1_dict[method], retrain_s_count_1_dict[method] = [], []
        p_values_0_dict[method] = []
        coverage_0_dict[method] = []
        
    for method in methods:
        if (method in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle', 'fixed_cal_offline']):
            m_0, s_0, martingale_value_0, sigma_0, p_vals = retrain_count(conformity_score=cs_0, training_schedule=schedule,W=W_dict[method],n_cal=len(val_loader_0.dataset), verbose=verbose, method=method)
        
        else:
            ## Run baseline with uniform weights
            m_0, s_0, martingale_value_0, sigma_0, p_vals = retrain_count(
                conformity_score=cs_0,
                training_schedule=schedule,
                verbose=verbose,
                method=methods
            )

        if m_0:
            retrain_m_count_0_dict[method] += 1
        if s_0:
            retrain_s_count_0_dict[method] += 1
            
        martingales_0_dict[method].append(martingale_value_0)
        sigmas_0_dict[method].append(sigma_0)

        ## Storing p-values
        p_values_0_dict[method].append(p_vals)
        coverage_0_dict[method].append(p_vals <= 0.9)

    for method in methods:
        m_1, s_1, martingale_value_1, sigma_1, p_vals = retrain_count(
            conformity_score=cs_1,
            training_schedule=schedule,
            verbose=verbose,
            method=methods
        )

        if m_1:
            retrain_m_count_1_dict[method] += 1
        if s_1:
            retrain_s_count_1_dict[method] += 1
        martingales_1_dict[method].append(martingale_value_1)
        sigmas_1_dict[method].append(sigma_1)
        
    ## min_len : Smallest fold length, for clipping longer ones to all same length
    min_len = np.min([len(sigmas_0_dict[method][i]) for i in range(0, len(sigmas_0_dict[method]))])
    paths_dict = {}
    for method in methods:
    
        paths = pd.DataFrame(np.c_[np.repeat(seed, min_len), np.arange(0, min_len)], columns = ['itrial', 'obs_idx'])
        sigmas_0 = sigmas_0_dict[method]
        sigmas_1 = sigmas_1_dict[method]
        for k in range(0, len(sigmas_0_dict[method])):
            paths['sigmas_0_'+str(k)] = sigmas_0_dict[method][k][0:min_len]
#             paths['cs_0_'+str(k)] = cs_0[k][0:min_len]
            paths['losses_0_'+str(k)] = test_loss[0:min_len]
            paths['pvals_0_'+str(k)] = p_values_0_dict[method][k][0:min_len]
            paths['coverage_0_'+str(k)] = coverage_0_dict[method][k][0:min_len]
        for k in range(0, len(sigmas_1)):
            paths['sigmas_1_'+str(k)] = sigmas_1[k][0:min_len]
#             paths['cs_1_'+str(k)] = cs_1[k][0:min_len]
        paths_dict[method] = paths
    
    return paths_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run WTR experiments for images.')
    parser.add_argument('--dataset0', type=str, default='mnist', \
                        help='Training/cal dataset for expts; Shifted split of dataset0 used for test set 0.')
    parser.add_argument('--dataset1', type=str, default=None, \
                        help='Dataset for test set 1; Test dataset which may differ from dataset0.')
    parser.add_argument('--verbose', action='store_true', help="Whether to print out alarm raising info.")
    parser.add_argument('--methods', nargs='+', help='Names of methods to try (weight types)', required = True)
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds to run experiments on.')
    parser.add_argument('--cs_type', type=str, default=1, help='Nonconformity score type.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the MLP model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of MLP')
    parser.add_argument('--bs', type=int, default=64, help='Batch size for training')
    parser.add_argument('--train_val_test_split_only', type=bool, default=False, help='Only split data into train/test sets or train/validation/test sets.')
    parser.add_argument('--corruption_type', type=str, default='fog', help='Type of corruption to apply to MNIST/CIFAR dataset.')
    parser.add_argument('--severity', type=int, default=5, help='Level of corruption to apply to MNIST/CIFAR dataset.')
    parser.add_argument('--init_phase', type=int, default=500, help="Num test pts that pre-trained density-ratio estimator has access to")
    parser.add_argument('--schedule', type=str, default='variable', help='Training schedule: variable or fixed.')
    parser.add_argument('--errs_window', type=int, default=50, help='Num observations to average for plotting errors.')
    parser.add_argument('--plot_errors', type=bool, default=True, help='Whether to also plot absolute errors.')
    parser.add_argument('--mixture_ratio_val', type=float, default=0.1, help='Mixture ratio of corruption for validation set.')
    parser.add_argument('--mixture_ratio_test', type=float, default=0.9, help='Mixture ratio of corruption for test set.')
    parser.add_argument('--val_set_size', type=int, default=10000, help='Validation set size.')

    args = parser.parse_args()
    dataset0_name = args.dataset0
    dataset1_name = args.dataset1
    n_seeds = args.n_seeds
    init_phase = args.init_phase
    methods = args.methods
    verbose = args.verbose
    cs_type = args.cs_type
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    #train_val_test_split_only = args.train_val_test_split_only
    train_val_test_split_only = False
    corruption_type = args.corruption_type
    severity = args.severity
    schedule = args.schedule
    errs_window = args.errs_window
    plot_errors = args.plot_errors
    mixture_ratio_val = args.mixture_ratio_val
    mixture_ratio_test = args.mixture_ratio_test
    val_set_size = args.val_set_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths_dict_all = {}
    for method in methods:
        paths_dict_all[method] = pd.DataFrame()


    methods_all = "_".join(methods)
    setting = '{}-{}-{}-{}-nseeds{}-epochs{}-lr{}-bs{}-severity{}-methods{}-mix_val{}-mix_test{}-val_set{}'.format(
        dataset0_name,
        dataset1_name,
        corruption_type,
        cs_type,
        n_seeds,
        epochs,
        lr,
        bs,
        severity,
        methods_all,
        mixture_ratio_val,
        mixture_ratio_test,
        val_set_size
    )

    print(f"Running experiments for {n_seeds} random seeds.")
    print(f"Training dataset: {dataset0_name}")
    print(f"Test dataset: {dataset1_name}")
    
    for seed in tqdm(range(0, n_seeds)):
        set_seed(seed)
        if dataset0_name == 'mnist':
            loaders = get_mnist_data(batch_size=bs, init_phase=init_phase, train_val_test_split_only=train_val_test_split_only, val_set_size=val_set_size)
            loader_1 = get_mnist_c_data(batch_size=bs, corruption_type=corruption_type, train_val_test_split_only=train_val_test_split_only,
                                        mixture_ratio_val=mixture_ratio_val, mixture_ratio_test=mixture_ratio_test, init_phase=init_phase,
                                        val_set_size=val_set_size)
        else:
            loaders = get_cifar10_data(batch_size=bs, init_phase=init_phase, train_val_test_split_only=train_val_test_split_only, val_set_size=val_set_size)
            loader_1 = get_cifar10_c_data(batch_size=bs, corruption_type=corruption_type, severity=severity, 
                                          train_val_test_split_only=train_val_test_split_only, mixture_ratio_val=mixture_ratio_val, 
                                          mixture_ratio_test=mixture_ratio_test, init_phase=init_phase, val_set_size=val_set_size)
        if train_val_test_split_only:
            train_loader_0, val_loader_0, test_loader_0 = loaders
            paths_dict_curr = training_function(
                train_loader_0=train_loader_0, 
                val_loader_0=val_loader_0,
                test_loader_0=test_loader_0, 
                dataset0_name=dataset0_name, 
                loader_1=loader_1,
                epochs=epochs,
                device=device,
                lr=lr,
                setting=setting,
                verbose=verbose, 
                methods=methods,
                schedule=schedule
            )
        else:
            train_loader_0, val_loader_0, test_loader_0, cal_test_w_est_loader_0 = loaders
            val_loader_mixed, test_loader_mixed, cal_test_w_est_loader_1 = loader_1
            paths_dict_curr = training_function(
                train_loader_0=train_loader_0, 
                val_loader_0=val_loader_0,
                test_loader_0=test_loader_0, 
                val_loader_mixed=val_loader_mixed,
                test_loader_mixed=test_loader_mixed,
                cal_test_w_est_loader_0=cal_test_w_est_loader_0,
                cal_test_w_est_loader_1=cal_test_w_est_loader_1,
                dataset0_name=dataset0_name, 
                epochs=epochs,
                device=device,
                lr=lr,
                setting=setting,
                verbose=verbose, 
                methods=methods,
                init_phase=init_phase
            )

        for method in methods:
            paths_dict_all[method] = pd.concat([paths_dict_all[method], paths_dict_curr[method]], ignore_index=True)
    
    sigmas_0_means_dict, sigmas_1_means_dict = {}, {}
    sigmas_0_stderr_dict, sigmas_1_stderr_dict = {}, {}
    errors_0_means_dict, errors_1_means_dict = {}, {}
    errors_0_stderr_dict, errors_1_stderr_dict = {}, {}
    coverage_0_means_dict = {}
    coverage_0_stderr_dict = {}
    pvals_0_means_dict = {}
    pvals_0_stderr_dict = {}
    p_vals_cal_dict = {}
    p_vals_test_dict = {}

    changepoint_index = val_set_size

    for method in methods:
        paths_dict_all[method].to_csv(f'../results/' + setting + '.csv')
    
        ## Compute average and stderr values for plotting
        paths_all = paths_dict_all[method]
        num_obs = paths_all['obs_idx'].max() + 1

        sigmas_0_means, sigmas_1_means = [], []
        sigmas_0_stderr, sigmas_1_stderr = [], []
        errors_0_means, errors_1_means = [], []
        errors_0_stderr, errors_1_stderr = [], []
        coverage_0_means = []
        coverage_0_stderr = []
        pvals_0_means = []
        pvals_0_stderr = []

        ## Compute average martingale values over trials
        sigmas_0_means.append(paths_all[['sigmas_0_0', 'obs_idx']].groupby('obs_idx').mean())
        sigmas_1_means.append(paths_all[['sigmas_1_0', 'obs_idx']].groupby('obs_idx').mean())

        ## Compute average and stderr absolute score (residual) values over window, trials
        errors_0_means_fold = []
        errors_0_stderr_fold = []
        coverage_0_means_fold = []
        coverage_0_stderr_fold = []
        pvals_0_means_fold = []
        pvals_0_stderr_fold = []

        for j in range(0, int(num_obs / errs_window)):
            ## Subset dataframe by window
            paths_all_sub = paths_all[paths_all['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]

            ## Averages and stderrs for that window
            errors_0_means_fold.append(paths_all_sub['losses_0_0'].mean())
            errors_0_stderr_fold.append(paths_all_sub['losses_0_0'].std() / np.sqrt(n_seeds*errs_window))

            ## Coverages for window
            coverage_0_means_fold.append(paths_all_sub['coverage_0_0'].mean())
            coverage_0_stderr_fold.append(paths_all_sub['coverage_0_0'].std() / np.sqrt(n_seeds*errs_window))

            ## P values for window
            pvals_0_means_fold.append(paths_all_sub['pvals_0_0'].mean())
            pvals_0_stderr_fold.append(paths_all_sub['pvals_0_0'].std() / np.sqrt(n_seeds*errs_window))

        ## Averages and stderrs for that fold
        errors_0_means.append(errors_0_means_fold)
        errors_0_stderr.append(errors_0_stderr_fold)

        ## Average coverages for fold
        coverage_0_means.append(coverage_0_means_fold)
        coverage_0_stderr.append(coverage_0_stderr_fold)

        ## Average pvals for fold
        pvals_0_means.append(pvals_0_means_fold)
        pvals_0_stderr.append(pvals_0_stderr_fold)     

        sigmas_0_means_dict[method], sigmas_1_means_dict[method] = sigmas_0_means, sigmas_1_means
        sigmas_0_stderr_dict[method], sigmas_1_stderr_dict[method] = sigmas_0_stderr, sigmas_1_stderr
        errors_0_means_dict[method], errors_1_means_dict[method] = errors_0_means, errors_1_means
        errors_0_stderr_dict[method], errors_1_stderr_dict[method] = errors_0_stderr, errors_1_stderr
        coverage_0_means_dict[method] = coverage_0_means
        coverage_0_stderr_dict[method] = coverage_0_stderr
        pvals_0_means_dict[method] = pvals_0_means
        pvals_0_stderr_dict[method] = pvals_0_stderr
        
        ## Plotting p-values for debugging
        paths_cal = paths_all[paths_all['obs_idx'] < changepoint_index]
        paths_test = paths_all[paths_all['obs_idx'] >= changepoint_index]
        p_vals_cal = np.array(paths_cal['pvals_0_0'])
        p_vals_test = np.array(paths_test['pvals_0_0'])
        p_vals_cal_dict[method] = p_vals_cal
        p_vals_test_dict[method] = p_vals_test

    plot_martingale_paths(
        dataset0_paths_dict=sigmas_0_means_dict,
        dataset0_name=dataset0_name,
        dataset1_paths_dict=sigmas_1_means_dict, 
        dataset1_name=dataset1_name,
        errors_0_means_dict=errors_0_means_dict,
        errors_1_means_dict=errors_1_means_dict,
        errors_0_stderr_dict=errors_0_stderr_dict,
        errors_1_stderr_dict=errors_1_stderr_dict,
        p_vals_cal_dict=p_vals_cal_dict,
        p_vals_test_dict=p_vals_test_dict,
        errs_window=errs_window,
        change_point_index=changepoint_index,
        title="Average paths of Shiryaev-Roberts Procedure",
        ylabel="Shiryaev-Roberts Statistics",
        martingale="shiryaev_roberts",
        dataset0_shift_type=corruption_type,
        noise_mu=None,
        noise_sigma=None,
        plot_errors=plot_errors,
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting,
        coverage_0_means_dict=coverage_0_means_dict,
        coverage_0_stderr_dict=coverage_0_stderr_dict,
        pvals_0_means_dict=pvals_0_means_dict,
        pvals_0_stderr_dict=pvals_0_stderr_dict,
        methods=methods,
        severity=severity
    )
    print('\nProgram done!')
