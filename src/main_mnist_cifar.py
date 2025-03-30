import numpy as np
import pandas as pd
from plot_separated import *
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import pdb

from utils import *
from martingales import *
from p_values import *
import argparse
import random
from podkopaev_ramdas.baseline_alg import podkopaev_ramdas_algorithm1, podkopaev_ramdas_changepoint_detection
from resnet import ResNet20, ResNet32
import time
from datetime import date


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


class RegularizedMNISTModel(nn.Module):
    """
    A CNN model for MNIST classification with regularization techniques.
    This model includes dropout and batch normalization to prevent overfitting
    and improve generalization, especially for corrupted images.
    
    Input: (N, 1, 28, 28)
    Output: (N, 10) for 10 classes (digits 0-9)
    """
    def __init__(self, dropout_rate=0.3):
        super(RegularizedMNISTModel, self).__init__()
        
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
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 10)
        
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


def eval_loss_prob(model, device, setting, loader_0, loader_1, binary_classifier_probs=False):
    model.load_state_dict(torch.load(os.getcwd() + '/../pkl_files/best_model_' + setting + '.pth'))
    model.eval()
    all_preds = []
    all_losses = []

    with torch.no_grad():
        # Evaluate on validation loader
        correct = 0
        total = 0
        for images, labels in loader_0:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            if binary_classifier_probs:
                ## If want probs for lik ratio estimation (binary classifier prob est)
                source_target_labels = torch.ones(len(labels), dtype=torch.int64).to(device) ## Estimate target probs p(Y_i=1) 
                class_probs = probabilities.gather(1, source_target_labels.view(-1, 1)).squeeze()
                
            else:
                ## Default case for class probs (not for binary classification for lik ratio est)
                class_probs = probabilities.gather(1, labels.view(-1, 1)).squeeze()
            all_preds.extend(class_probs.cpu().numpy())
            
            if not binary_classifier_probs:
                # Calculate cross entropy loss for each sample
                loss = F.cross_entropy(outputs, labels, reduction='none')
                all_losses.append(loss.cpu().numpy())
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f'Accuracy on cal loader: {accuracy * 100:.2f}%')
        
        # Evaluate on test loader
        for images, labels in loader_1:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            if binary_classifier_probs:
                ## If want probs for lik ratio estimation (binary classifier prob est)
                source_target_labels = torch.ones(len(labels), dtype=torch.int64).to(device) ## Estimate target probs p(Y_i=1) 
                class_probs = probabilities.gather(1, source_target_labels.view(-1, 1)).squeeze()
            else:
                ## Default case for class probs (not for binary classification for lik ratio est)
                class_probs = probabilities.gather(1, labels.view(-1, 1)).squeeze()
            all_preds.extend(class_probs.cpu().numpy())
            
            if not binary_classifier_probs:
                # Calculate cross entropy loss for each sample
                loss = F.cross_entropy(outputs, labels, reduction='none')
                all_losses.append(loss.cpu().numpy())

            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        print(f'Accuracy on test loader: {accuracy * 100:.2f}%')
        if not binary_classifier_probs:
            all_losses = np.concatenate(all_losses)
    return np.array(all_preds), all_losses


def fit(model, epochs, train_loader, optimizer, setting, device):
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



def train_and_evaluate(args, train_loader_0, test_loader_0, test_loader_s, device, setting, loader_1=None, val_loader_0=None,
                       cal_test_w_est_loader_0=None, cal_test_w_est_loader_1=None, test_loader_mixed=None, epsilon=1e-9,
                       cal_test_w_est_loader_binary_0=None, cal_test_w_est_loader_binary_1=None, test_loader_mixed_binary=None,
                       test_loader_s_binary=None):
    '''
    baseline uniform weights:
    - train on clean data, eval on clean data: train_loader_0 + val_loader_0 + test_loader_0
    - train on clean data, eval on corrupted data: train_loader_0 + val_loader_0 + loader_1
    WCTMs:
    - train on clean data, eval on clean data: train_loader_0 + val_loader_0 + test_loader_0
    - train on clean data, eval on corrupted data: train_loader_0 + val_loader_mixed + test_loader_mixed
    '''
    cs_0 = {}
    cs_1 = {}
    clean_loss_dict = {}
    corrupt_loss_dict = {}
    W_0_dict = {}
    W_1_dict = {}

    # Train the model on the training set proper
    if dataset0_name == 'mnist':
        # model = MLP(input_size=784, hidden_size=256, num_classes=10).to(device)
        model = RegularizedMNISTModel(dropout_rate=0.3).to(device)
    elif dataset0_name == 'cifar10':
        model = ResNet20().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"\nTraining classification models for {args.epochs} epochs")
    fit(model, args.epochs, train_loader_0, optimizer, setting, device)

    for method in args.methods:
        if method == 'fixed_cal_offline':
            print(f"\nEvaluating {method} on clean datasets")
            clean_pred, clean_loss = eval_loss_prob(model, device, setting, cal_test_w_est_loader_0, test_loader_s)
            print(f"\nEvaluating {method} on corrupted datasets")
            corrupt_pred, corrupt_loss = eval_loss_prob(model, device, setting, cal_test_w_est_loader_1, test_loader_mixed)
            clean_loss_dict[method] = clean_loss
            corrupt_loss_dict[method] = corrupt_loss
        else:
            print(f"\nEvaluating {method} on clean datasets")
            clean_pred, clean_loss = eval_loss_prob(model, device, setting, val_loader_0, test_loader_0)
            print(f"\nEvaluating {method} on corrupted datasets")
            corrupt_pred, corrupt_loss = eval_loss_prob(model, device, setting, val_loader_0, loader_1)
            clean_loss_dict[method] = clean_loss
            corrupt_loss_dict[method] = corrupt_loss

        if cs_type == 'probability':
            cs_0[method] = 1 - clean_pred
            cs_1[method] = 1 - corrupt_pred
        elif cs_type == 'neg_log':
            cs_0[method] = -np.log(clean_pred + epsilon)
            cs_1[method] = -np.log(corrupt_pred + epsilon)
    
    #### Computing (unnormalized) weights
    ### NOTE: 'fixed_cal_offline' is the primary method implemented for now.
    for method in args.methods:

        if (method in ['fixed_cal_offline']):
            print(f"\nEstimating weights for {method} on clean dataset")
            W_0_dict[method] = offline_lik_ratio_estimates_images(cal_test_w_est_loader_binary_0, test_loader_s_binary, args.dataset0, device=device, setting=setting, epochs=args.weight_epoch)
            print(f"\nEstimating weights for {method} on corrupted dataset")
            W_1_dict[method] = offline_lik_ratio_estimates_images(cal_test_w_est_loader_binary_1, test_loader_mixed_binary, args.dataset0, device=device, setting=setting, epochs=args.weight_epoch)
        else:
            ## Else: Unweighted / uniform-weighted CTM
            W_0_dict[method] = None
            W_1_dict[method] = None

    return cs_0, cs_1, clean_loss_dict, corrupt_loss_dict, W_0_dict, W_1_dict


def retrain_count(args, conformity_score, method, cu_confidence=0.99, W=None):
    p_values = calculate_p_values(conformity_score)
    
    if (method in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle', 'fixed_cal_offline']):
        p_values, q_lower, q_upper = calculate_weighted_p_values_and_quantiles(args, conformity_score, W, args.val_set_size, method)
    else:
        p_values, q_lower, q_upper = calculate_p_values_and_quantiles(conformity_score, args.alpha, args.cs_type)
    
    # num_samples = min(10000 + args.val_set_size, args.num_samples_vis + args.init_clean)
    if args.init_ctm_on_cal_set:
        retrain_m, martingale_value = composite_jumper_martingale(p_values[:args.num_samples_vis], verbose=args.verbose, threshold=args.mt_threshold)
    else:
        retrain_m, martingale_value = composite_jumper_martingale(p_values[args.val_set_size:], verbose=args.verbose, threshold=args.mt_threshold)

    if args.schedule == 'variable':
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, args.sr_threshold, args.verbose)
    elif (args.schedule == 'basic'):
        print("plotting martingale (wealth) values directly")
        retrain_s, sigma = shiryaev_roberts_procedure(martingale_value, args.sr_threshold, args.verbose)
        sigma = martingale_value
    else:
        retrain_s, sigma = cusum_procedure(martingale_value, cu_confidence, args.verbose)
    
    # Append the last value of sigma to itself to ensure consistent length
    if len(sigma) > 0:
        sigma = np.append(sigma, sigma[-1])
    return retrain_m, retrain_s, martingale_value, sigma, p_values[:args.num_samples_vis], q_lower[:args.num_samples_vis], q_upper[:args.num_samples_vis]


def training_function(args, train_loader_0, test_loader_0, test_loader_s, device, setting, val_loader_0=None, loader_1=None, 
                      cal_test_w_est_loader_0=None, cal_test_w_est_loader_1=None, test_loader_mixed=None,
                      cal_test_w_est_loader_binary_0=None, cal_test_w_est_loader_binary_1=None, test_loader_mixed_binary=None,
                      test_loader_s_binary=None):
    
    cs_0, cs_1, clean_loss_dict, corrupt_loss_dict, W_0_dict, W_1_dict = train_and_evaluate(
        args=args,
        train_loader_0=train_loader_0,
        val_loader_0=val_loader_0,
        test_loader_0=test_loader_0,
        test_loader_s=test_loader_s,
        test_loader_s_binary=test_loader_s_binary,
        loader_1=loader_1,
        cal_test_w_est_loader_0=cal_test_w_est_loader_0,
        cal_test_w_est_loader_1=cal_test_w_est_loader_1,
        cal_test_w_est_loader_binary_0=cal_test_w_est_loader_binary_0,
        cal_test_w_est_loader_binary_1=cal_test_w_est_loader_binary_1,
        test_loader_mixed=test_loader_mixed,
        test_loader_mixed_binary=test_loader_mixed_binary,
        device=device,
        setting=setting
    )

    martingales_0_dict, martingales_1_dict = {}, {}
    sigmas_0_dict, sigmas_1_dict = {}, {}
    # retrain_m_count_0_dict, retrain_s_count_0_dict = {}, {}
    # retrain_m_count_1_dict, retrain_s_count_1_dict = {}, {}
    p_values_0_dict, p_values_1_dict = {}, {}
    coverage_0_dict, coverage_1_dict = {}, {}
    widths_0_dict, widths_1_dict = {}, {}

    if args.run_PR_ST:
        ## Podkopaev Ramdas sequential testing method
        PR_ST_alarm_0_dict, PR_ST_alarm_1_dict = {}, {}
        PR_ST_source_UCB_tols_0_dict, PR_ST_source_UCB_tols_1_dict = {}, {}
        PR_ST_target_LCBs_0_dict, PR_ST_target_LCBs_1_dict = {}, {}
        
    if args.run_PR_CD:
        ## Podkopaev Ramdas changepoint detection method
        PR_CD_alarm_0_dict, PR_CD_alarm_1_dict = {}, {}
        PR_CD_source_UCB_tols_0_dict, PR_CD_source_UCB_tols_1_dict = {}, {}
        PR_CD_target_LCBs_0_dict, PR_CD_target_LCBs_1_dict = {}, {}

    for method in args.methods:
        martingales_0_dict[method], martingales_1_dict[method] = [], []
        sigmas_0_dict[method], sigmas_1_dict[method] = [], []
        # retrain_m_count_0_dict[method], retrain_s_count_0_dict[method] = [], []
        # retrain_m_count_1_dict[method], retrain_s_count_1_dict[method] = [], []
        p_values_0_dict[method], p_values_1_dict[method] = [], []
        coverage_0_dict[method], coverage_1_dict[method] = [], []
        widths_0_dict[method], widths_1_dict[method] = [], []

        if args.run_PR_ST:
            PR_ST_alarm_0_dict['PR_ST_cp_'+method], PR_ST_alarm_1_dict['PR_ST_cp_'+method] = [], []
            PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method], PR_ST_source_UCB_tols_1_dict['PR_ST_cp_'+method] = [], []
            PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method], PR_ST_target_LCBs_1_dict['PR_ST_cp_'+method] = [], []
            
        if args.run_PR_CD:
            PR_CD_alarm_0_dict['PR_CD_cp_'+method], PR_CD_alarm_1_dict['PR_CD_cp_'+method] = [], []
            PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method], PR_CD_source_UCB_tols_1_dict['PR_CD_cp_'+method] = [], []
            PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method], PR_CD_target_LCBs_1_dict['PR_CD_cp_'+method] = [], []

    for method in args.methods:
        if (method in ['fixed_cal', 'fixed_cal_oracle', 'one_step_est', 'one_step_oracle', 'batch_oracle', 'multistep_oracle', 'fixed_cal_offline']):
            print('Clean dataset:   ', method)
            m_0, s_0, martingale_value_0, sigma_0, p_vals_0, q_lower_0, q_upper_0 = retrain_count(args=args, conformity_score=cs_0[method], W=W_0_dict[method], method=method)
            print(f'Corrupted dataset {args.corruption_type} severity {args.severity}:   ', method)
            m_1, s_1, martingale_value_1, sigma_1, p_vals_1, q_lower_1, q_upper_1 = retrain_count(args=args, conformity_score=cs_1[method], W=W_1_dict[method], method=method)
        else:
            ## Run baseline with uniform weights
            print('Clean dataset:   ', method)
            m_0, s_0, martingale_value_0, sigma_0, p_vals_0, q_lower_0, q_upper_0 = retrain_count(args=args, conformity_score=cs_0[method], method=method, W=W_0_dict[method])
            print(f'Corrupted dataset {args.corruption_type} severity {args.severity}:   ', method)
            m_1, s_1, martingale_value_1, sigma_1, p_vals_1, q_lower_1, q_upper_1 = retrain_count(args=args, conformity_score=cs_1[method], method=method, W=W_1_dict[method])
            
        martingales_0_dict[method].append(martingale_value_0)
        sigmas_0_dict[method].append(sigma_0)
        martingales_1_dict[method].append(martingale_value_1)
        sigmas_1_dict[method].append(sigma_1)

        ## Storing p-values
        p_values_0_dict[method].append(p_vals_0)
        p_values_1_dict[method].append(p_vals_1)
        coverage_0_dict[method].append(p_vals_0 <= 1 - args.alpha)
        coverage_1_dict[method].append(p_vals_1 <= 1 - args.alpha)
        coverage_vals_0 = ((q_lower_0 <= cs_0[method][:args.num_samples_vis])&(q_upper_0 >= cs_0[method][:args.num_samples_vis]))
        coverage_vals_1 = ((q_lower_1 <= cs_1[method][:args.num_samples_vis])&(q_upper_1 >= cs_1[method][:args.num_samples_vis]))
        width_vals_0 = q_upper_0 - q_lower_0
        width_vals_1 = q_upper_1 - q_lower_1

        if args.run_PR_ST:
            miscoverage_losses = 1 - coverage_vals
            
            ## Run Podkopaev Ramdas baseline on miscoverage losses for corresponding CP method
            print("Running PodRam algorithm 1")
            start_time = time.time()
            PR_ST_alarm_test_idx, PR_ST_source_UCB_tol, PR_ST_target_LCBs = podkopaev_ramdas_algorithm1(\
                                                                                    miscoverage_losses[:args.val_set_size], \
                                                                                    miscoverage_losses[args.val_set_size:], \
                                                                                    source_conc_type=pr_source_conc_type, \
                                                                                    target_conc_type=pr_target_conc_type, \
                                                                                    eps_tol=pr_st_eps_tol, \
                                                                                    source_delta=pr_st_source_delta, \
                                                                                    target_delta=pr_st_target_delta,\
                                                                                    stop_criterion=pr_st_stop_criterion)
            print("Completed PodRam algorithm 1; runtime in min = ", (time.time()-start_time)/60)
            
            ## Record results for PodRam sequential testing baseline
            if (PR_ST_alarm_test_idx is None):
                PR_ST_alarm_0_dict['PR_ST_cp_'+method].append(None)
            else:
                PR_ST_alarm_0_dict['PR_ST_cp_'+method].append(PR_ST_alarm_test_idx)
                
            PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method].append(PR_ST_source_UCB_tol)
            PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method].append(PR_ST_target_LCBs)

        if args.run_PR_CD:

            print("Running PodRam changepoint detection algo")
            start_time = time.time()
            PR_CD_alarm_test_idx, PR_CD_source_UCB_tol, PR_CD_target_LCBs = podkopaev_ramdas_changepoint_detection(\
                                                                                    miscoverage_losses[:args.val_set_size], \
                                                                                    miscoverage_losses[args.val_set_size:], \
                                                                                    source_conc_type=pr_source_conc_type, \
                                                                                    target_conc_type=pr_target_conc_type, \
                                                                                    eps_tol=pr_cd_eps_tol,\
                                                                                    source_delta=pr_cd_source_delta, \
                                                                                    target_delta=pr_cd_target_delta,\
                                                                                    stop_criterion=pr_cd_stop_criterion)
        
            print("Completed PodRam changepoint detection algo; runtime in min = ", (time.time()-start_time)/60)
            
            ## Record results for PodRam changepoint detection method
            if (PR_CD_alarm_test_idx is None):
                PR_CD_alarm_0_dict['PR_CD_cp_'+method].append(None)
            else:
                PR_CD_alarm_0_dict['PR_CD_cp_'+method].append(PR_CD_alarm_test_idx)
                
            PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method].append(PR_CD_source_UCB_tol)
            PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method].append(PR_CD_target_LCBs)

        if not args.init_ctm_on_cal_set:
            p_vals_0 = p_vals_0[args.val_set_size:]
            coverage_vals = coverage_vals[args.val_set_size:]
            width_vals = width_vals[args.val_set_size:]

        p_values_0_dict[method].append(p_vals_0)
        coverage_0_dict[method].append(coverage_vals_0)
        widths_0_dict[method].append(width_vals_0)
        p_values_1_dict[method].append(p_vals_1)
        coverage_1_dict[method].append(coverage_vals_1)
        widths_1_dict[method].append(width_vals_1)
        
    ## min_len : Smallest fold length, for clipping longer ones to all same length
    # min_len_0 = np.min([len(sigmas_0_dict[method][i]) for i in range(0, len(sigmas_0_dict[method]))])
    # min_len_1 = np.min([len(sigmas_1_dict[method][i]) for i in range(0, len(sigmas_1_dict[method]))])
    paths_dict = {}
    # paths_dict_1 = {}
    PR_ST_paths_dict = {}
    PR_ST_paths=None
    
    PR_CD_paths_dict = {}
    PR_CD_paths=None

    for method in methods:
    
        paths = pd.DataFrame(np.c_[np.repeat(seed, args.num_samples_vis), np.arange(0, args.num_samples_vis)], columns = ['itrial', 'obs_idx'])
        # paths_1 = pd.DataFrame(np.c_[np.repeat(seed, min_len_1), np.arange(0, min_len_1)], columns = ['itrial', 'obs_idx'])
        sigmas_0 = sigmas_0_dict[method]
        sigmas_1 = sigmas_1_dict[method]
        for k in range(0, len(sigmas_0)):
            paths['sigmas_0_'+str(k)] = sigmas_0_dict[method][k]
            paths['martingales_0_'+str(k)] = martingales_0_dict[method][k]
            paths['pvals_0_'+str(k)] = p_values_0_dict[method][k]
            paths['coverage_0_'+str(k)] = coverage_0_dict[method][k]
            paths['widths_0_'+str(k)] = widths_0_dict[method][k]
            paths['losses_0'] = clean_loss_dict[method][:args.num_samples_vis]
            
        for k in range(0, len(sigmas_1)):
            paths['sigmas_1_'+str(k)] = sigmas_1[k]
            paths['martingales_1_'+str(k)] = martingales_1_dict[method][k]
            paths['pvals_1_'+str(k)] = p_values_1_dict[method][k]
            paths['coverage_1_'+str(k)] = coverage_1_dict[method][k]
            paths['widths_1_'+str(k)] = widths_1_dict[method][k]
            paths['losses_1'] = corrupt_loss_dict[method][:args.num_samples_vis]
        paths_dict[method] = paths
        # paths_dict_1[method] = paths_1

        if args.run_PR_ST:
            ## PR_ST_cp baseline method:
            PR_ST_min_len = len(PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method][0])
            PR_ST_paths = pd.DataFrame(np.c_[np.repeat(seed, PR_ST_min_len), np.arange(0, PR_ST_min_len)], columns = ['itrial', 'obs_idx'])
            for k in range(0, len(PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method])):
                PR_ST_paths['PR_ST_alarm_0_'+str(k)] = PR_ST_alarm_0_dict['PR_ST_cp_'+method][k]
                PR_ST_paths['PR_ST_UCBtol_0_'+str(k)] = PR_ST_source_UCB_tols_0_dict['PR_ST_cp_'+method][k]
                PR_ST_paths['PR_ST_LCB_0_'+str(k)] = PR_ST_target_LCBs_0_dict['PR_ST_cp_'+method][k][0:PR_ST_min_len]
                
            PR_ST_paths_dict['PR_ST_cp_'+method] = PR_ST_paths
            
        if args.run_PR_CD:
            ## PR_CD_cp baseline method:
            PR_CD_min_len = len(PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method][0])
            PR_CD_paths = pd.DataFrame(np.c_[np.repeat(seed, PR_CD_min_len), np.arange(0, PR_CD_min_len)], columns = ['itrial', 'obs_idx'])
            for k in range(0, len(PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method])):
                PR_CD_paths['PR_CD_alarm_0_'+str(k)] = PR_CD_alarm_0_dict['PR_CD_cp_'+method][k]
                PR_CD_paths['PR_CD_UCBtol_0_'+str(k)] = PR_CD_source_UCB_tols_0_dict['PR_CD_cp_'+method][k]
                PR_CD_paths['PR_CD_LCB_0_'+str(k)] = PR_CD_target_LCBs_0_dict['PR_CD_cp_'+method][k][0:PR_CD_min_len]
                
            PR_CD_paths_dict['PR_CD_cp_'+method] = PR_CD_paths
    return paths_dict, PR_ST_paths_dict, PR_CD_paths_dict


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
    parser.add_argument('--corruption_type', type=str, default='fog', help='Type of corruption to apply to MNIST/CIFAR dataset.')
    parser.add_argument('--severity', type=int, default=5, help='Level of corruption to apply to MNIST/CIFAR dataset.')
    parser.add_argument('--init_clean', type=int, default=500, help="Num target pts that pre-trained density-ratio estimator has access to")
    parser.add_argument('--init_corrupt', type=int, default=500, help="Num target pts that pre-trained density-ratio estimator has access to")

    parser.add_argument('--schedule', type=str, default='variable', help='Training schedule: variable or fixed.')
    parser.add_argument('--errs_window', type=int, default=50, help='Num observations to average for plotting errors.')
    parser.add_argument('--mixture_ratio_val', type=float, default=0.1, help='Mixture ratio of corruption for validation set.')
    parser.add_argument('--mixture_ratio_test', type=float, default=0.9, help='Mixture ratio of corruption for test set.')
    parser.add_argument('--val_set_size', type=int, default=10000, help='Validation set size.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Pre-specified miscoverage rate.')
    parser.add_argument('--sr_threshold', type=float, default=1e20, help='Threshold for shiryaev roberts procedure.')
    parser.add_argument('--mt_threshold', type=float, default=1e20, help='Martingale threshold.')
    parser.add_argument('--num_samples_vis', type=int, default=1000, help='Number of samples to visualize.')
    parser.add_argument('--weight_epoch', type=int, default=80, help='Number of epoch for training weight estimator.')

    ## PodRam baseline params:
    parser.add_argument('--run_PR_ST', dest='run_PR_ST', action='store_true', help="Whether to run PodkopaevRamdas sequential testing (their algorithm 1) baseline.")
    parser.add_argument('--run_PR_CD', dest='run_PR_CD', action='store_true', help="Whether to run PodkopaevRamdas changepoint detection baseline (runs algorithm 1 many times, can be slow).")
    parser.add_argument('--pr_source_conc_type', type=str, default='betting', help="Concentration type used for source data in PodRam baselines (both sequential testing and changepoint detection).")
    parser.add_argument('--pr_target_conc_type', type=str, default='betting', help="Concentration type used for target data in PodRam baselines (both sequential testing and changepoint detection).")
    parser.add_argument('--pr_st_eps_tol', type=float, default=0.0, help="PodRam ST epsilon tolerance.")
    parser.add_argument('--pr_st_source_delta', type=float, default=1/200, help="PodRam ST source delta.")
    parser.add_argument('--pr_st_target_delta', type=float, default=1/200, help="PodRam ST target delta.")
    parser.add_argument('--pr_st_stop_criterion', type=str, default='fixed_length', help="Stopping criterion for PodRam ST Algorithm 1 baseline.")
    parser.add_argument('--pr_cd_eps_tol', type=float, default=0.0, help="PodRam CD epsilon tolerance.")
    parser.add_argument('--pr_cd_source_delta', type=float, default=1/20000, help="PodRam CD source delta.")
    parser.add_argument('--pr_cd_target_delta', type=float, default=1/20000, help="PodRam CD target delta.")
    parser.add_argument('--pr_cd_stop_criterion', type=str, default='fixed_length', help="Stopping criterion for PodRam changepoint detection baseline.")
#     parser.add_argument('--init_ctm_on_cal_set', type=bool, default=True, help="Whether to initialize conformal martingales on the calibration set (as in Vovk et al); false := initialize at deployment time instead for comparison with Ramdas")
    parser.add_argument('--init_on_cal', dest='init_ctm_on_cal_set', action='store_true',
                    help='Set the init_ctm_on_cal_set flag value to True.')
    parser.add_argument('--init_on_test', dest='init_ctm_on_cal_set', action='store_false',
                    help='Set the init_ctm_on_cal_set flag value to False.')
    parser.set_defaults(init_ctm_on_cal_set=True, run_PR_ST=False, run_PR_CD=False)

    args = parser.parse_args()
    dataset0_name = args.dataset0
    dataset1_name = args.dataset1
    n_seeds = args.n_seeds
    methods = args.methods
    verbose = args.verbose
    cs_type = args.cs_type
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    corruption_type = args.corruption_type
    severity = args.severity
    schedule = args.schedule
    errs_window = args.errs_window
    mixture_ratio_val = args.mixture_ratio_val
    mixture_ratio_test = args.mixture_ratio_test
    val_set_size = args.val_set_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_PR_ST = args.run_PR_ST
    run_PR_CD = args.run_PR_CD
    ## PodRam params for both ST and CD baselines
    pr_source_conc_type=args.pr_source_conc_type
    pr_target_conc_type=args.pr_target_conc_type
    
    ## PodRam ST baseline params
    pr_st_eps_tol=args.pr_st_eps_tol
    pr_st_source_delta=args.pr_st_source_delta
    pr_st_target_delta=args.pr_st_target_delta
    pr_st_stop_criterion=args.pr_st_stop_criterion
    
    ## PodRam CD baseline params
    pr_cd_eps_tol=args.pr_cd_eps_tol
    pr_cd_source_delta=args.pr_cd_source_delta
    pr_cd_target_delta=args.pr_cd_target_delta
    pr_cd_stop_criterion=args.pr_cd_stop_criterion

    paths_dict_all = {}
    PR_ST_paths_dict_all = {}
    PR_CD_paths_dict_all = {}
    for method in args.methods:
        paths_dict_all[method] = pd.DataFrame()

        if run_PR_ST:
            PR_ST_paths_dict_all['PR_ST_cp_'+method] = pd.DataFrame()
        if run_PR_CD:
            PR_CD_paths_dict_all['PR_CD_cp_'+method] = pd.DataFrame()

    methods_all = "_".join(args.methods)
    setting = '{}-{}-{}-{}-nseeds{}-epochs{}-lr{}-bs{}-severity{}-methods{}-mix_val{}-mix_test{}-val_set{}-init{}-num_samples_vis{}'.format(
        args.dataset0,
        args.dataset1,
        args.corruption_type,
        args.cs_type,
        args.n_seeds,
        args.epochs,
        args.lr,
        args.bs,
        args.severity,
        methods_all,
        args.mixture_ratio_val,
        args.mixture_ratio_test,
        args.val_set_size,
        args.init_clean,
        args.num_samples_vis
    )

    if run_PR_ST:
        PR_ST_setting = 'sConc{}-tConc{}-eTol{}-sDelta{}-tDelta{}-stop{}'.format(
            pr_source_conc_type,
            pr_target_conc_type,
            pr_st_eps_tol, 
            pr_st_source_delta, 
            pr_st_target_delta,
            pr_st_stop_criterion
        )
    if run_PR_CD:
        PR_CD_setting = 'sConc{}-tConc{}-eTol{}-sDelta{}-tDelta{}-stop{}'.format(
            pr_source_conc_type, 
            pr_target_conc_type,
            pr_cd_eps_tol, 
            pr_cd_source_delta, 
            pr_cd_target_delta,
            pr_cd_stop_criterion
        )

    print(f"Running experiments for {args.n_seeds} random seeds.")
    print(f"Training dataset: {args.dataset0}")
    print(f"Test dataset: {args.dataset1}")
    
    for seed in tqdm(range(0, args.n_seeds)):
        set_seed(seed)
        if dataset0_name == 'mnist':
            loaders = get_mnist_data(args)
            loader_1 = get_mnist_c_data(args)
        else:
            loaders = get_cifar10_data(args)
            loader_1 = get_cifar10_c_data(args)

        train_loader_0, val_loader_0, test_loader_0, cal_test_w_est_loader_binary_0, cal_test_w_est_loader_0, test_loader_s, test_loader_s_binary = loaders
        loader_1, cal_test_w_est_loader_binary_1, cal_test_w_est_loader_1, test_loader_mixed, test_loader_mixed_binary = loader_1
        paths_dict_curr, PR_ST_paths_dict_curr, PR_CD_paths_dict_curr = training_function(
            args=args,
            train_loader_0=train_loader_0, 
            val_loader_0=val_loader_0,
            test_loader_0=test_loader_0, 
            test_loader_s=test_loader_s,
            test_loader_s_binary=test_loader_s_binary,
            loader_1=loader_1,
            test_loader_mixed=test_loader_mixed,
            test_loader_mixed_binary=test_loader_mixed_binary,
            cal_test_w_est_loader_0=cal_test_w_est_loader_0,
            cal_test_w_est_loader_1=cal_test_w_est_loader_1,
            cal_test_w_est_loader_binary_0=cal_test_w_est_loader_binary_0,
            cal_test_w_est_loader_binary_1=cal_test_w_est_loader_binary_1,
            device=device,
            setting=setting
        )

        for method in methods:
            paths_dict_all[method] = pd.concat([paths_dict_all[method], paths_dict_curr[method]], ignore_index=True)
            if run_PR_ST:
                PR_ST_paths_dict_all['PR_ST_cp_'+method] = pd.concat([PR_ST_paths_dict_all['PR_ST_cp_'+method], \
                                                                      PR_ST_paths_dict_curr['PR_ST_cp_'+method]],\
                                                                     ignore_index=True)
            if run_PR_CD:
                PR_CD_paths_dict_all['PR_CD_cp_'+method] = pd.concat([PR_CD_paths_dict_all['PR_CD_cp_'+method], \
                                                                      PR_CD_paths_dict_curr['PR_CD_cp_'+method]],\
                                                                     ignore_index=True)

    ## Save all results together
    results_all = paths_dict_all[methods[0]]
    results_all['method'] = methods[0]
    
    if run_PR_ST:
        PR_ST_results_all = PR_ST_paths_dict_all['PR_ST_cp_'+methods[0]]
        PR_ST_results_all['method'] = 'PR_ST_cp_'+methods[0]
    if run_PR_CD:
        PR_CD_results_all = PR_CD_paths_dict_all['PR_CD_cp_'+methods[0]]
        PR_CD_results_all['method'] = 'PR_CD_cp_'+methods[0]
    
    for method in methods[1:]:
        paths_dict_all[method]['method'] = method
        results_all = pd.concat([results_all, paths_dict_all[method]], ignore_index=True)

        if run_PR_ST:
            PR_ST_paths_dict_all['PR_ST_cp_'+method]['method'] = 'PR_ST_cp_'+method
            PR_ST_results_all = pd.concat([PR_ST_results_all, PR_ST_paths_dict_all['PR_ST_cp_'+method]], ignore_index=True)
        if run_PR_CD:
            PR_CD_paths_dict_all['PR_CD_cp_'+method]['method'] = 'PR_CD_cp_'+method
            PR_CD_results_all = pd.concat([PR_CD_results_all, PR_CD_paths_dict_all['PR_CD_cp_'+method]], ignore_index=True)
        
    results_all.to_csv(f'../results/{setting}.csv')
    
    if run_PR_ST:
        PR_ST_results_all.to_csv(f'../results/{setting}_PR_ST-{PR_ST_setting}.csv')
    if run_PR_CD:
        PR_CD_results_all.to_csv(f'../results/{setting}_PR_CD-{PR_CD_setting}.csv')

    sigmas_0_means_dict, sigmas_1_means_dict = {}, {}
    sigmas_0_stderr_dict, sigmas_1_stderr_dict = {}, {}
    martingales_0_means_dict, martingales_1_means_dict = {}, {}
    martingales_0_stderr_dict, martingales_1_stderr_dict = {}, {}
    errors_0_means_dict, errors_1_means_dict = {}, {}
    errors_0_stderr_dict, errors_1_stderr_dict = {}, {}
    coverage_0_means_dict = {}
    coverage_0_stderr_dict = {}
    widths_0_medians_dict = {}
    widths_0_lower_q_dict = {}
    widths_0_upper_q_dict = {}
    pvals_0_means_dict = {}
    pvals_0_stderr_dict = {}
    p_vals_pre_change_dict = {}
    p_vals_post_change_dict = {}
    changepoint_index = args.val_set_size

    for method in methods:
        paths_dict_all[method].to_csv(f'../results/' + setting + '.csv')
        
        ## Compute average and stderr values for plotting
        paths_all = paths_dict_all[method]
        num_obs = paths_all['obs_idx'].max() + 1

        sigmas_0_means, sigmas_1_means = [], []
        sigmas_0_stderr, sigmas_1_stderr = [], []
        martingales_0_means, martingales_1_means = [], []
        martingales_0_stderr, martingales_1_stderr = [], []
        errors_0_means, errors_1_means = [], []
        errors_0_stderr, errors_1_stderr = [], []
        coverage_0_means = []
        coverage_0_stderr = []
        widths_0_medians = []
        widths_0_lower_q = []
        widths_0_upper_q = []
        pvals_0_means = []
        pvals_0_stderr = []

        ## Compute average martingale values over trials
        sigmas_0_means.append(paths_all[['sigmas_0_0', 'obs_idx']].groupby('obs_idx').mean())
        sigmas_0_stderr.append(paths_all[['sigmas_0_0', 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))
        sigmas_1_means.append(paths_all[['sigmas_1_0', 'obs_idx']].groupby('obs_idx').mean())
        sigmas_1_stderr.append(paths_all[['sigmas_1_0', 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))

        martingales_0_means.append(paths_all[['martingales_0_0', 'obs_idx']].groupby('obs_idx').mean())
        martingales_0_stderr.append(paths_all[['martingales_0_0', 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))
        martingales_1_means.append(paths_all[['martingales_1_0', 'obs_idx']].groupby('obs_idx').mean())
        martingales_1_stderr.append(paths_all[['martingales_1_0', 'obs_idx']].groupby('obs_idx').std() / np.sqrt(n_seeds))

        ## Compute average and stderr absolute score (residual) values over window, trials
        errors_0_means_fold = []
        errors_0_stderr_fold = []
        coverage_0_means_fold = []
        coverage_0_stderr_fold = []
        widths_0_medians_fold = []
        widths_0_lower_q_fold = []
        widths_0_upper_q_fold = []
        pvals_0_means_fold = []
        pvals_0_stderr_fold = []

        for j in range(0, int(num_obs / errs_window)):
            ## Subset dataframe by window
            paths_all_sub = paths_all[paths_all['obs_idx'].isin(np.arange(j*errs_window,(j+1)*errs_window))]

            ## Averages and stderrs for that window
            errors_0_means_fold.append(paths_all_sub['losses_0'].mean())
            errors_0_stderr_fold.append(paths_all_sub['losses_0'].std() / np.sqrt(n_seeds*errs_window))

            ## Coverages for window
            coverage_0_means_fold.append(paths_all_sub['coverage_0_0'].mean())
            coverage_0_stderr_fold.append(paths_all_sub['coverage_0_0'].std() / np.sqrt(n_seeds*errs_window))
            
            ## Widths for window
            wid_med = paths_all_sub['widths_0_0'].median()
            widths_0_medians_fold.append(wid_med)
            widths_0_lower_q_fold.append(paths_all_sub['widths_0_0'].quantile(0.25))
            widths_0_upper_q_fold.append(paths_all_sub['widths_0_0'].quantile(0.75))

            ## P values for window
            pvals_0_means_fold.append(paths_all_sub['pvals_0_0'].mean())
            pvals_0_stderr_fold.append(paths_all_sub['pvals_0_0'].std() / np.sqrt(n_seeds*errs_window))

        ## Averages and stderrs for that fold
        errors_0_means.append(errors_0_means_fold)
        errors_0_stderr.append(errors_0_stderr_fold)

        ## Average coverages for fold
        coverage_0_means.append(coverage_0_means_fold)
        coverage_0_stderr.append(coverage_0_stderr_fold)
        
        ## Median widths for fold
        widths_0_medians.append(widths_0_medians_fold)
        widths_0_lower_q.append(widths_0_lower_q_fold)
        widths_0_upper_q.append(widths_0_upper_q_fold)

        ## Average pvals for fold
        pvals_0_means.append(pvals_0_means_fold)
        pvals_0_stderr.append(pvals_0_stderr_fold)  

        sigmas_0_means_dict[method], sigmas_1_means_dict[method] = sigmas_0_means, sigmas_1_means
        sigmas_0_stderr_dict[method], sigmas_1_stderr_dict[method] = sigmas_0_stderr, sigmas_1_stderr
        martingales_0_means_dict[method], martingales_1_means_dict[method] = martingales_0_means, martingales_1_means
        martingales_0_stderr_dict[method], martingales_1_stderr_dict[method] = martingales_0_stderr, martingales_1_stderr
        errors_0_means_dict[method], errors_1_means_dict[method] = errors_0_means, errors_1_means
        errors_0_stderr_dict[method], errors_1_stderr_dict[method] = errors_0_stderr, errors_1_stderr
        coverage_0_means_dict[method] = coverage_0_means
        coverage_0_stderr_dict[method] = coverage_0_stderr
        pvals_0_means_dict[method] = pvals_0_means
        pvals_0_stderr_dict[method] = pvals_0_stderr
        widths_0_medians_dict[method] = widths_0_medians
        widths_0_lower_q_dict[method] = widths_0_lower_q
        widths_0_upper_q_dict[method] = widths_0_upper_q

        ## Saving p-values together for histograms
        paths_pre_change = paths_all[paths_all['obs_idx'] < changepoint_index]
        paths_post_change = paths_all[paths_all['obs_idx'] >= changepoint_index]
        p_vals_pre_change = paths_pre_change['pvals_0_0']
        p_vals_post_change = paths_post_change['pvals_0_0']
        p_vals_pre_change = np.concatenate((p_vals_pre_change, paths_pre_change['pvals_0_0']))
        p_vals_post_change = np.concatenate((p_vals_post_change, paths_post_change['pvals_0_0']))
        p_vals_pre_change_dict[method] = p_vals_pre_change
        p_vals_post_change_dict[method] = p_vals_post_change

    plot_martingale_paths(
        dataset0_paths_dict=sigmas_0_means_dict,
        dataset0_paths_stderr_dict=sigmas_0_stderr_dict,
        martingales_0_dict=martingales_0_means_dict,
        martingales_0_stderr_dict=martingales_0_stderr_dict,
        dataset1_paths_dict=sigmas_1_means_dict,
        dataset1_paths_stderr_dict=sigmas_1_stderr_dict,
        change_point_index=changepoint_index,
        martingales_1_dict=martingales_1_means_dict,
        martingales_1_stderr_dict=martingales_1_stderr_dict,
        dataset0_name=dataset0_name,
        dataset0_shift_type=corruption_type,
        martingale=["Shiryaev-Roberts", "Martingale"],
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting,
        methods=methods,
        severity=severity
    )
    plot_errors(
        errors_0_means_dict=errors_0_means_dict,
        errors_0_stderr_dict=errors_0_stderr_dict,
        errs_window=errs_window,
        change_point_index=changepoint_index,
        dataset0_name=dataset0_name,
        dataset0_shift_type=corruption_type,
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting,
        methods=methods,
        severity=severity
    )
    plot_coverage(
        coverage_0_means_dict=coverage_0_means_dict,
        coverage_0_stderr_dict=coverage_0_stderr_dict,
        errs_window=errs_window,
        change_point_index=changepoint_index,
        dataset0_name=dataset0_name,
        dataset0_shift_type=corruption_type,
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting,
        methods=methods,
        severity=severity
    )
    plot_widths(
        widths_0_medians_dict=widths_0_medians_dict,
        widths_0_lower_q_dict=widths_0_lower_q_dict,
        widths_0_upper_q_dict=widths_0_upper_q_dict,
        errs_window=errs_window,
        change_point_index=changepoint_index,
        dataset0_name=dataset0_name,
        dataset0_shift_type=corruption_type,
        n_seeds=n_seeds,
        cs_type=cs_type,
        setting=setting,
        methods=methods,
        severity=severity
    )
    plot_p_vals(
        p_vals_pre_change_dict=p_vals_pre_change_dict,
        p_vals_post_change_dict=p_vals_post_change_dict,
        dataset0_name=dataset0_name,
        setting=setting,
        methods=methods
    )
    
    print('\nProgram done!')
