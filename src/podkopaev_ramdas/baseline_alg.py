from podkopaev_ramdas.concentrations import *
from podkopaev_ramdas.tests import *
from podkopaev_ramdas.set_valued_prediction import Set_valued_predictor_wrapper
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

import time



def podkopaev_ramdas_algorithm1(cal_losses, test_losses, source_conc_type='betting', target_conc_type='betting', \
                                verbose=False, eps_tol=0.0, source_delta=0.005, target_delta = 0.005,\
                                stop_criterion='fixed_length', max_length=2000):
    """
    Implementation of Podkopaev & Ramdas *sequential testing* baseline, i.e., algorithm 1 in that paper. 

    Parameters
    ----------
    cal_losses       : Array of losses for calibration (holdout) set; for evaluating set losses, should be 
                      *mis*coverage indicators; (if evaluating point losses, would be conformity scores)
    test_losses      : Array of losses for test (deployment) set; for evaluating set losses, should be 
                      *mis*coverage indicators; (if evaluating point losses, would be conformity scores)
    source_conc_type : Concentration used for source UCB
    target_conc_type : Concentration used for target LCB
    eps_tol          : Epsilon tolerance
    stop_criterion   : Criterion for when to stop running the algorithm. in ['first_alarm', 'full_path', 'fixed_length']
    fixed_length     : Max num test points to process when stop_criterion=='fixed_length'

    Returns
    ------- 
    alarm_idx        : The index of the test point where alarm is first raised
    source_upper_bound_plus_tol : Estimate of UCB on source risk plus tolerance: \hat{U}_S(f) + \epsilon
    target_lower_bounds : Array, estimates of LCB on target risk at each timestep \hat{L}_T^{t}(f)
    """
    
    start_time = time.time()
    
    ## Index in test set of first alarm
    alarm_idx = None
    elapsed_time_min = None
    
    ## Set up Drop_tester for computer UCB on source risk and LCB on target risk
    tester = Drop_tester()
    tester.eps_tol = eps_tol
    tester.source_conc_type = source_conc_type
    tester.target_conc_type = target_conc_type
    tester.change_type = 'absolute'
    tester.source_delta = target_delta
    tester.target_delta = target_delta
    
    ## Estimate source risk UCB
    tester.estimate_risk_source(cal_losses)
    source_upper_bound_plus_tol = tester.source_rejection_threshold
    if (verbose):
        print(f'source_upper_bound_plus_tol (\hatU_S(f) + \epsilon) : {source_upper_bound_plus_tol}\n')
    
    ## Sequentially estimate target risk LCB for each testpoint
    T = len(test_losses)
    target_lower_bounds = [] # np.zeros(T)
    
    for t in range(T):
#     for t in tqdm(range(T)):
        tester.estimate_risk_target(test_losses[:(t+1)])
        target_lower_bounds.append(tester.target_risk_lower_bound)
        
        if (verbose and t % 50 == 0):
            print(f'target_lower_bounds[t={t}] (\hatL_T^{t}(f)): {target_lower_bounds[t]}')
    
        if (target_lower_bounds[t] > source_upper_bound_plus_tol and alarm_idx is None):
            alarm_idx = t
            elapsed_time_min = (time.time() - start_time) / 60
                        
            if (verbose):
                print(f'podkopaev_ramdas_algorithm1 alarm raised at test point {t}!\n')
                
            if (stop_criterion == 'first_alarm'):
                return alarm_idx, source_upper_bound_plus_tol, target_lower_bounds, elapsed_time_min
            
        if (t > max_length): #stop_criterion == 'fixed_length' and 
            return alarm_idx, source_upper_bound_plus_tol, target_lower_bounds, elapsed_time_min
           
    
    return alarm_idx, source_upper_bound_plus_tol, np.array(target_lower_bounds), elapsed_time_min





def podkopaev_ramdas_changepoint_detection(cal_losses, test_losses, source_conc_type='betting', target_conc_type='betting', \
                                verbose=False, eps_tol=0.0, source_delta=0.000005, target_delta = 0.000005,\
                                stop_criterion='first_alarm', max_length=500, batch_size=50):
    """
    Implementation of Podkopaev & Ramdas *changepoint detection* baseline,
    i.e., see "From sequential testing to changepoint detection" in that paper.
    
    Runs algorithm1 of that paper at each test timepoint and returns the earliest stopping time.
    
    NOTE: Currently this implementation is TOO SLOW (due to calling podkopaev_ramdas_algorithm1 n_test times)
          to compare to in an online data (ie, non minibatch) setting.

    Parameters
    ----------
    cal_losses       : Array of losses for calibration (holdout) set; for evaluating set losses, should be 
                      *mis*coverage indicators; (if evaluating point losses, would be conformity scores)
    test_losses      : Array of losses for test (deployment) set; for evaluating set losses, should be 
                      *mis*coverage indicators; (if evaluating point losses, would be conformity scores)
    source_conc_type : Concentration used for source UCB
    target_conc_type : Concentration used for target LCB
    eps_tol          : Epsilon tolerance
    stop_criterion   : Criterion for when to stop running the algorithm. in ['first_alarm', 'full_path', 'fixed_length']
    fixed_length     : Max num test points to process when stop_criterion=='fixed_length'

    Returns
    ------- 
    alarm_idx        : The index of the test point where alarm is first raised
    source_UCB_tol   : Estimate of UCB on source risk plus tolerance: \hat{U}_S(f) + \epsilon
    target_max_LCBs  : Array, estimates of LCB on target risk at each timestep \hat{L}_T^{t}(f)
    """
    start_time = time.time()
    
    T = len(test_losses) ## num_test
    testers_all = []
    ## For each separate t-th run of algorithm 1, record the following:
#     alarm_times = [] ## Alarm times for t-th run
    alarm_idx = None
    elapsed_time_min = None
    source_UCB_tol = None ## Source UCB+tolerance for t-th run
    target_LCBs_all = [] ## Array of LCBs for t-th run 
                     ## Note: target_LCBs[t] will be an array for timesteps t:(T-1) (of length T-t) 
    target_max_LCBs = []
    
    ## Runs algorithm1 of that paper at each test timepoint and return the earliest stopping time.
    alarm_min = T+1
    
    for t in range(int(T/batch_size)):
#         print(t)

#     for t in tqdm(range(int(T/batch_size))):
        ## Initiate new sequential testing object
        ## Set up Drop_tester for computer UCB on source risk and LCB on target risk
        testers_all.append(Drop_tester())
        testers_all[t].eps_tol = eps_tol
        testers_all[t].source_conc_type = source_conc_type
        testers_all[t].target_conc_type = target_conc_type
        testers_all[t].change_type = 'absolute'
        testers_all[t].source_delta = target_delta
        testers_all[t].target_delta = target_delta

        ## Estimate source risk UCB, which will be the same for all testers
        if (t == 0):
            testers_all[t].estimate_risk_source(cal_losses)
            source_UCB_tol = testers_all[t].source_rejection_threshold
            if (verbose):
                print(f'source_UCB_tol (\hatU_S(f) + \epsilon) : {source_UCB_tol}\n')
        else:
            testers_all[t].source_rejection_threshold = testers_all[0].source_rejection_threshold
                    

        ## Sequentially estimate target risk LCB for each testpoint
        target_LCBs_all.append([]) ## Add new list for storing new test's LCBs
        
        ## Update LCB_t estimate for each i-th tester, i \in {0, ..., t}
        ## Ie, at each time t, LCB for tester i is computed on points i:t
        for i in range(len(testers_all)):
#             print(f't = {t}, i = {i}, (i*batch_size) = {(i*batch_size)}, (t+1)*batch_size = {(t+1)*batch_size}')
#             print(f'len(test_losses) {len(test_losses)}')
            testers_all[i].estimate_risk_target(test_losses[(i*batch_size):((t+1)*batch_size)])
            target_LCBs_all[i].append(testers_all[i].target_risk_lower_bound)
            
            ## Check alarm threshold
            if (testers_all[i].target_risk_lower_bound > source_UCB_tol and alarm_idx is None):
                alarm_idx = (t+1)*batch_size-1
                elapsed_time_min = (time.time() - start_time) / 60

        
        ## Record max target LCB for time t
        target_LCBs_0_t = target_LCBs_all[:(t+1)] ## list of target_LCBs arrays for times 0, ..., t
        target_LCBs_t = [target_LCBs_0_t[i][t-i] for i in range(len(target_LCBs_0_t))] ## list of LCB values at time t
        target_max_LCBs.append(max(target_LCBs_t)) ## Maximum LCB value at time t
        if (verbose and ((t % 10 == 0) or batch_size >= 10)):
            print(f'target_max_LCBs[(t+1)*batch_size={(t+1)*batch_size}] (max(\hatL_T^t(f))): {target_max_LCBs[t]}')

        
        ## If alarm was raised at time t then can stop testing
        if (alarm_idx is not None and stop_criterion=='first_alarm'):
            break
        
        if (stop_criterion=='fixed_length' and (alarm_idx is not None and (t+1)*batch_size >= max_length)):
            break
            
        ## max length for any method
        if ((t+1)*batch_size >= max_length):
            break
    
    return alarm_idx, source_UCB_tol, np.array(target_max_LCBs), elapsed_time_min




def pod_ram_mnist_cifar(model, ds_clean, ds_corrupted, tester, device, num_of_repeats=50, plot=True, 
                  plot_batch_size=50, plot_num_of_batches=40, setting=None, corruption_type='fog'):
    """
    DRAFT implementation of Podkopaev & Ramdas baseline, i.e., algorithm 1 in that paper. 

    Parameters
    ----------
    model             : The neural network model to be evaluated.
    ds_clean          : The clean dataset.
    ds_corrupted      : The corrupted dataset.
    tester            : The tester object used to estimate risks.
    device            : The device to run the model on (e.g., 'cpu' or 'cuda').
    num_of_repeats    : Number of repetitions for the algorithm.
    plot              : Boolean flag to indicate if the results should be plotted.
    plot_batch_size   : Batch size for plotting.
    plot_num_of_batches: Number of batches for plotting.
    setting           : The setting for the experiment (used for saving plots).
    corruption_type   : The type of corruption applied to the dataset.

    Returns
    ------- 
    """
    clean_lower_bounds = list()
    corrupted_lower_bounds = list()
    source_upper_bounds = list()
    print('Running Podkopaev & Ramdas Algorithms...\n')

    for cur_run in tqdm(range(num_of_repeats)):

        # here we only use test set of clean and corrupted data
        # both of them have 10000 samples for mnist
        # follow the pod&ram paper, they used 1000 samples to compute conc ineq for source risk
        # the remaining 9000 samples are used to compute conc ineq for target risk
        # but they didn't end up with using all 9000 samples, they gradually increased the sample sizes with 50 samples step-size
        indices = np.arange(10000)
        np.random.shuffle(indices)
        risk_source_indices = indices[:1000]
        risk_target_indices = indices[1000:]

        ###### Source risk upper bound
        risk_source = Subset(ds_clean, risk_source_indices.tolist())
        loader_source = DataLoader(risk_source, batch_size=1000, shuffle=False)
        dataiter = iter(loader_source)
        images, labels = next(dataiter)
        images = images.to(device)
        outputs = model(images).argmax(axis=1)
        ind_loss_source = misclas_losses(labels.numpy(), outputs.cpu().numpy())

        tester.estimate_risk_source(ind_loss_source)
        source_upper_bounds += [tester.source_rejection_threshold]

        ###### Target risk lower bound
        clean_lower_bounds += [[]]
        corrupted_lower_bounds += [[]]

        ###### clean
        np.random.shuffle(risk_target_indices)
        risk_target_clean = Subset(ds_clean, risk_target_indices[:plot_num_of_batches * plot_batch_size].tolist())
        loader_target_clean = DataLoader(risk_target_clean, batch_size=plot_num_of_batches * plot_batch_size, shuffle=False)
        dataiter = iter(loader_target_clean)
        images, labels = next(dataiter)
        images = images.to(device)
        outputs = model(images).argmax(axis=1)
        all_losses_clean = misclas_losses(labels.numpy(), outputs.cpu().numpy()).astype(int)

        for cur_batch in range(plot_num_of_batches):
            cur_losses = all_losses_clean[:(cur_batch + 1) * plot_batch_size]
            tester.estimate_risk_target(cur_losses)
            clean_lower_bounds[cur_run] += [tester.target_risk_lower_bound]

        ###### corrupted
        np.random.shuffle(risk_target_indices)
        risk_target_corrupted = Subset(ds_corrupted, risk_target_indices[:plot_num_of_batches * plot_batch_size].tolist())
        loader_target_corrupted = DataLoader(risk_target_corrupted, batch_size=plot_num_of_batches * plot_batch_size, shuffle=False)
        dataiter = iter(loader_target_corrupted)
        images, labels = next(dataiter)
        images = images.to(device)
        outputs = model(images).argmax(axis=1)
        all_losses_corrupted = misclas_losses(labels.numpy(), outputs.cpu().numpy()).astype(int)

        for cur_batch in range(plot_num_of_batches):
            cur_losses = all_losses_corrupted[:(cur_batch + 1) * plot_batch_size]
            tester.estimate_risk_target(cur_losses)
            corrupted_lower_bounds[cur_run] += [tester.target_risk_lower_bound]
    
    ### mean and std across 10 repeats
    clean_mean = np.mean(clean_lower_bounds, axis=0)
    corrupted_mean = np.mean(corrupted_lower_bounds, axis=0)
    clean_std = np.std(clean_lower_bounds, axis=0)
    corrupted_std = np.std(corrupted_lower_bounds, axis=0)

    ### compute how fast it raises alarm here: (averaged) target lower bound - source upper bound
    mask = corrupted_mean - np.mean(source_upper_bounds) > 0
    pos_indices = np.where(mask)[0]
    if len(pos_indices) == 0:
        print('No alarm is raised by Podkopaev & Ramdas algorithm\n')
    else:
        first_positive_index = pos_indices[0]
        print(f"Alarm raised within the range of [{first_positive_index * plot_batch_size} ~ {(first_positive_index + 1) * plot_batch_size}] samples\n")

    if plot:
        print('Plotting Podkopaev & Ramdas bounds...')
        plot_func(plot_num_of_batches, plot_batch_size, num_of_repeats, clean_mean, clean_std, corrupted_mean, 
                  corrupted_std, source_upper_bounds, corruption_type, setting)


def pod_ram_set_value(model, ds_clean, ds_corrupted, tester, device, num_of_repeats=50, plot=True, 
                      plot_batch_size=50, plot_num_of_batches=40, setting=None, corruption_type='fog'):
    """
    DRAFT implementation of Podkopaev & Ramdas baseline, i.e., algorithm 1 in that paper. 

    Parameters
    ----------
    model             : The neural network model to be evaluated.
    ds_clean          : The clean dataset.
    ds_corrupted      : The corrupted dataset.
    tester            : The tester object used to estimate risks.
    device            : The device to run the model on (e.g., 'cpu' or 'cuda').
    num_of_repeats    : Number of repetitions for the algorithm.
    plot              : Boolean flag to indicate if the results should be plotted.
    plot_batch_size   : Batch size for plotting.
    plot_num_of_batches: Number of batches for plotting.
    setting           : The setting for the experiment (used for saving plots).
    corruption_type   : The type of corruption applied to the dataset.

    Returns
    ------- 
    """
    clean_lower_bounds = list()
    corrupted_lower_bounds = list()
    source_upper_bounds = list()
    wrap = Set_valued_predictor_wrapper()
    wrap.base_predictor = model
    wrap.delta = 0.05
    wrap.alpha=0.05
    print('Running Podkopaev & Ramdas Algorithms...\n')

    for cur_run in tqdm(range(num_of_repeats)):

        # learn a wrapper on the calibration set with 1000 samples
        indices = np.arange(10000)
        np.random.shuffle(indices)
        cal_indices = indices[:1000]
        risk_source_indices = indices[1000:2000]
        risk_target_indices = indices[2000:]

        X_cal = ds_clean.data[cal_indices]
        ###### Source risk upper bound
        risk_source = Subset(ds_clean, risk_source_indices.tolist())
        loader_source = DataLoader(risk_source, batch_size=1000, shuffle=False)
        dataiter = iter(loader_source)
        images, labels = next(dataiter)
        images = images.to(device)
        outputs = model(images).argmax(axis=1)
        ind_loss_source = misclas_losses(labels.numpy(), outputs.cpu().numpy())

        tester.estimate_risk_source(ind_loss_source)
        source_upper_bounds += [tester.source_rejection_threshold]

        ###### Target risk lower bound
        clean_lower_bounds += [[]]
        corrupted_lower_bounds += [[]]

        ###### clean
        np.random.shuffle(risk_target_indices)
        risk_target_clean = Subset(ds_clean, risk_target_indices[:plot_num_of_batches * plot_batch_size].tolist())
        loader_target_clean = DataLoader(risk_target_clean, batch_size=plot_num_of_batches * plot_batch_size, shuffle=False)
        dataiter = iter(loader_target_clean)
        images, labels = next(dataiter)
        images = images.to(device)
        outputs = model(images).argmax(axis=1)
        all_losses_clean = misclas_losses(labels.numpy(), outputs.cpu().numpy()).astype(int)

        for cur_batch in range(plot_num_of_batches):
            cur_losses = all_losses_clean[:(cur_batch + 1) * plot_batch_size]
            tester.estimate_risk_target(cur_losses)
            clean_lower_bounds[cur_run] += [tester.target_risk_lower_bound]

        ###### corrupted
        np.random.shuffle(risk_target_indices)
        risk_target_corrupted = Subset(ds_corrupted, risk_target_indices[:plot_num_of_batches * plot_batch_size].tolist())
        loader_target_corrupted = DataLoader(risk_target_corrupted, batch_size=plot_num_of_batches * plot_batch_size, shuffle=False)
        dataiter = iter(loader_target_corrupted)
        images, labels = next(dataiter)
        images = images.to(device)
        outputs = model(images).argmax(axis=1)
        all_losses_corrupted = misclas_losses(labels.numpy(), outputs.cpu().numpy()).astype(int)

        for cur_batch in range(plot_num_of_batches):
            cur_losses = all_losses_corrupted[:(cur_batch + 1) * plot_batch_size]
            tester.estimate_risk_target(cur_losses)
            corrupted_lower_bounds[cur_run] += [tester.target_risk_lower_bound]
    
    ### mean and std across 10 repeats
    clean_mean = np.mean(clean_lower_bounds, axis=0)
    corrupted_mean = np.mean(corrupted_lower_bounds, axis=0)
    clean_std = np.std(clean_lower_bounds, axis=0)
    corrupted_std = np.std(corrupted_lower_bounds, axis=0)

    ### compute how fast it raises alarm here: (averaged) target lower bound - source upper bound
    mask = corrupted_mean - np.mean(source_upper_bounds) > 0
    pos_indices = np.where(mask)[0]
    if len(pos_indices) == 0:
        print('No alarm is raised by Podkopaev & Ramdas algorithm\n')
    else:
        first_positive_index = pos_indices[0]
        print(f"Alarm raised within the range of [{first_positive_index * plot_batch_size} ~ {(first_positive_index + 1) * plot_batch_size}] samples\n")

    if plot:
        print('Plotting Podkopaev & Ramdas bounds...')
        plot_func(plot_num_of_batches, plot_batch_size, num_of_repeats, clean_mean, clean_std, corrupted_mean, 
                  corrupted_std, source_upper_bounds, corruption_type, setting)
        

def plot_func(num_of_batches, batch_size, num_of_repeats, clean_mean, clean_std, corrupted_mean, 
              corrupted_std, source_upper_bounds, corruption_type, setting):

    l1, = plt.plot((np.arange(num_of_batches) + 1) * batch_size,
               clean_mean,
               color='dimgray',
               marker='*',
               label='Clean')

    plt.fill_between((np.arange(num_of_batches) + 1) * batch_size,
                    y1=clean_mean - 2 * clean_std / np.sqrt(num_of_repeats),
                    y2=clean_mean + 2 * clean_std / np.sqrt(num_of_repeats),
                    color='dimgray',
                    alpha=0.5)

    l2, = plt.plot((np.arange(num_of_batches) + 1) * batch_size,
                corrupted_mean,
                color='indianred',
                marker='*',
                label='Fog')

    plt.fill_between((np.arange(num_of_batches) + 1) * batch_size,
                    y1=corrupted_mean - 2 * corrupted_std / np.sqrt(num_of_repeats),
                    y2=corrupted_mean + 2 * corrupted_std / np.sqrt(num_of_repeats),
                    color='indianred',
                    alpha=0.5)

    l3, = plt.plot((np.arange(num_of_batches) + 1) * batch_size,
                np.repeat(np.mean(source_upper_bounds), num_of_batches),
                linestyle='dashed',
                c='goldenrod')

    plt.fill_between((np.arange(num_of_batches) + 1) * batch_size,
                    y1=np.mean(source_upper_bounds) -
                    2 * np.std(source_upper_bounds) / np.sqrt(num_of_repeats),
                    y2=np.mean(source_upper_bounds) +
                    2 * np.std(source_upper_bounds) / np.sqrt(num_of_repeats),
                    color='goldenrod',
                    alpha=0.5)

    plt.xlabel('Number of samples from the target domain', fontsize=23)
    plt.ylabel('Misclassification risk', fontsize=23)

    # plt.legend(loc='best', markerscale=1.5, prop={'size': 20})

    p = plt.plot([0.15], marker='None', linestyle='None', label='dummy-tophead')

    plt.ylabel('Misclassification risk', fontsize=23)
    plt.xlabel('Number of samples from the target domain', fontsize=23)

    categories = [
        'Rejection threshold: ' +
        r'$\widehat{U}_S(f) + \varepsilon_{\mathrm{tol}}$',
        r'$\textbf{LCB on the target risk for:}$ ', 'Clean', f'{corruption_type}'
    ]

    leg4 = plt.legend([l3, p, l1, l2],
                    categories,
                    loc=1,
                    ncol=1,
                    prop={'size': 18})  # Two columns, horizontal group labels
    plt.savefig(f'/cis/home/xhan56/code/wtr/figs/{setting}.pdf', bbox_inches='tight')