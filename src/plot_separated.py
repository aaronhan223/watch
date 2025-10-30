import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import os
import pdb
from datetime import date


def plot_martingale_paths(dataset0_paths_dict, dataset0_paths_stderr_dict, martingales_0_dict,
                          martingales_0_stderr_dict, dataset1_paths_dict, dataset1_paths_stderr_dict, 
                          dataset0_name, change_point_index, 
                          martingales_1_dict=None, martingales_1_stderr_dict=None, title="Martingale Paths",
                          xlabel="Test (Deployment) Datapoint Index $t$", martingale="martingale_paths", 
                          dataset0_shift_type='none', n_seeds=1, cs_type='signed', 
                          setting=None, methods=['none'], severity=None, plot_image_data='mnist_15000_',
                          title_size=28, x_label_size=25, y_label_size=25, 
                          legend_size=20, x_tick_size=18, y_tick_size=18):
    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    stat_validities = {'Shiryaev-Roberts' : 'Scheduled', 'Martingale' : 'Anytime-Valid'}
    stat_formal = {'Shiryaev-Roberts' : '($\sum_{i=0}^{t-1} M_t / M_i$)', 'Martingale' : '($M_t / M_0$)'}

    paths_0_dicts_all = [dataset0_paths_dict, martingales_0_dict]
    paths_0_stderr_dicts_all = [dataset0_paths_stderr_dict, martingales_0_stderr_dict]
    paths_1_dicts_all = [dataset1_paths_dict, martingales_1_dict]
    paths_1_stderr_dicts_all = [dataset1_paths_stderr_dict, martingales_1_stderr_dict]
    thresholds = [10**6, 10**6]
    for p_i, paths_0_dict in enumerate(paths_0_dicts_all):
        paths_0_stderr_dict = paths_0_stderr_dicts_all[p_i]
        statistic_name = martingale[p_i]
        stat_validity = stat_validities[statistic_name]
        plt.figure(figsize=(10, 5))

        # Plot dataset0 group with dashed lines
        for m_i, method in enumerate(methods):
            mean_path = paths_0_dict[method][0]
            stderr_path = paths_0_stderr_dict[method][0]
            
            plt.plot(mean_path, label=method_name_dict[method], linestyle='-', color=f'C{m_i}')
            # Add error bands
            plt.fill_between(
                range(len(mean_path)),
                np.squeeze((mean_path - stderr_path).to_numpy()),
                np.squeeze((mean_path + stderr_path).to_numpy()),
                alpha=0.3,
                color=f'C{m_i}'
            )

        plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')
        plt.axhline(y=thresholds[p_i], color='red', linestyle='--', label='Alarm threshold', linewidth=3)

        plt.yscale('log')  # Use logarithmic scale for the y-axis
        plt.xlabel(xlabel, fontsize=x_label_size)
        plt.ylabel(fr'{statistic_name} Value {stat_formal[statistic_name]}', fontsize=y_label_size)
        plt.title(f'Minor Benign Corruption \n ({stat_validity} Monitoring Criterion)', fontsize=title_size)

        plt.legend(fontsize=legend_size)
        plt.grid(True, which="both", ls="--")
        plt.xticks(fontsize=x_tick_size)        
        plt.yticks(fontsize=y_tick_size)
        plt.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_{statistic_name}_clean_{setting}.pdf', bbox_inches='tight')

    for p_i, paths_1_dict in enumerate(paths_1_dicts_all):
        paths_1_stderr_dict = paths_1_stderr_dicts_all[p_i]
        statistic_name = martingale[p_i]
        stat_validity = stat_validities[statistic_name]
        plt.figure(figsize=(10, 5))

        # Plot dataset1 group with dashed lines
        for m_i, method in enumerate(methods):
            mean_path = paths_1_dict[method][0]
            stderr_path = paths_1_stderr_dict[method][0]
            
            plt.plot(mean_path, label=method_name_dict[method], linestyle='-', color=f'C{m_i}')
            # Add error bands
            plt.fill_between(
                range(len(mean_path)),
                np.squeeze((mean_path - stderr_path).to_numpy()),
                np.squeeze((mean_path + stderr_path).to_numpy()),
                alpha=0.3,
                color=f'C{m_i}'
            )

        plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')
        plt.axhline(y=thresholds[p_i], color='red', linestyle='--', label='Alarm threshold', linewidth=3)

        plt.yscale('log')  # Use logarithmic scale for the y-axis
        plt.xlabel(xlabel, fontsize=x_label_size)
        plt.ylabel(fr'{statistic_name} Value {stat_formal[statistic_name]}', fontsize=y_label_size)
        plt.title(f'Minor Benign Corruption \n ({stat_validity} Monitoring Criterion)', fontsize=title_size)

        plt.legend(fontsize=legend_size)
        plt.grid(True, which="both", ls="--")
        plt.xticks(fontsize=x_tick_size)        
        plt.yticks(fontsize=y_tick_size)
        plt.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_{statistic_name}_corrupted_{setting}.pdf', bbox_inches='tight')


def plot_errors(errors_0_means_dict, errors_0_stderr_dict, 
                errs_window, dataset0_name, change_point_index,
                xlabel="Test (Deployment) Datapoint Index $t$", 
                dataset0_shift_type='none', cov_shift_bias=0.0, n_seeds=1, cs_type='signed', 
                setting=None, label_shift_bias=1, noise_mu=0, noise_sigma=0, methods=['none'], severity=None, 
                plot_image_data='mnist_15000_',
                title_size=28, x_label_size=25, y_label_size=25, 
                legend_size=20, x_tick_size=18, y_tick_size=18):
    
    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}

    plt.figure(figsize=(10, 8))
    
    for m_i, method in enumerate(methods):
        for i, errs in enumerate(errors_0_means_dict[method]):
            plt.plot(np.arange(0, len(errs)*errs_window, errs_window), errs, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)

            plt.fill_between(np.arange(0, len(errs)*errs_window, errs_window), \
                                (errs-np.array(errors_0_stderr_dict[method][i])).flatten(), \
                                (errs+np.array(errors_0_stderr_dict[method][i])).flatten(), alpha=0.5, color=f'C{m_i}')

    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')

    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.ylabel(r'Absolute Error ($\leftarrow$)', fontsize=y_label_size)
    if dataset0_shift_type == 'covariate':
        plt.title(f'Error paths, {dataset0_shift_type} shift, \n bias={str(cov_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=title_size)
    elif dataset0_shift_type == 'label':
        plt.title(f'Error paths, {dataset0_shift_type} shift, \n label shift={str(label_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=title_size)
    elif dataset0_shift_type == 'noise':
        plt.title(f'Error paths, {dataset0_shift_type} shift, \n mean var={str(noise_mu)} {str(noise_sigma)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=title_size)
    else:
        plt.title(f'Error paths, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=title_size)

    plt.legend(fontsize=legend_size)
    plt.grid(True, which="both", ls="--")
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size)     
    
    if (dataset0_shift_type == 'none'):
        plt.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_{dataset0_name}_AbsoluteErrors_{setting}.pdf', bbox_inches='tight')
    else:
        plt.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_error_{setting}.pdf', bbox_inches='tight')


def plot_p_vals(p_vals_pre_change_dict, p_vals_post_change_dict, dataset0_name, 
                setting=None, methods=['none'], plot_image_data='mnist_15000_',
                title_size=28, x_label_size=25, x_tick_size=18):

    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}

    fig, ax = plt.subplots(1, 2,figsize=(10, 8))
    # ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    # ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    for m_i, method in enumerate(methods):
        ax[0].hist(p_vals_pre_change_dict[method], label=method_name_dict[method], color=f'C{m_i}', alpha=0.5) #row=0, col=0
        ax[0].set_title('Before Changepoint', fontsize=x_label_size)

        ax[1].hist(p_vals_post_change_dict[method], label=method_name_dict[method], color=f'C{m_i}', alpha=0.5) #row=1, col=0
        ax[1].set_title('After Changepoint', fontsize=x_label_size)
    
    ax[0].tick_params(axis='both', which='major', labelsize=x_tick_size)
    ax[1].tick_params(axis='both', which='major', labelsize=x_tick_size)
    ax[0].set_xlabel('P-Values', fontsize=x_label_size, color='white')
    ax[1].set_xlabel('P-Values', fontsize=x_label_size, color='white')
    
#     plt.legend(fontsize=legend_size)
    fig.suptitle(f'Histograms of P-Values (Monitoring Inputs)', fontsize=title_size)
    fig.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_p_vals_hist_{setting}.pdf', bbox_inches='tight')


def plot_classification_metrics(set_sizes_dict, class_coverage_dict, 
                               errs_window, dataset0_name, change_point_index, args,
                               xlabel="Test (Deployment) Datapoint Index $t$", 
                               dataset0_shift_type='none', n_seeds=1, cs_type='signed', 
                               setting=None, methods=['none'], severity=None, plot_image_data='mnist_15000_',
                               title_size=28, x_label_size=25, y_label_size=25, 
                               legend_size=20, x_tick_size=18, y_tick_size=18):
    """
    Plot classification-specific metrics: prediction set size and class coverage rate.
    
    Parameters:
    -----------
    set_sizes_dict : dict
        Dictionary mapping methods to lists of prediction set sizes
    class_coverage_dict : dict
        Dictionary mapping methods to lists of class coverage rates
    errs_window : int
        Window size for averaging metrics
    dataset0_name : str
        Name of the dataset (e.g., 'mnist', 'cifar10')
    change_point_index : int
        Index in the dataset where the distribution shift occurs
    plot_image_data : str
        Path prefix for saving the image files
    """
    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    
    # Plot prediction set sizes
    plt.figure(figsize=(10, 8))
    
    for m_i, method in enumerate(methods):
        # Process all seeds for this method
        all_set_sizes = set_sizes_dict[method]
        n_actual_seeds = len(all_set_sizes)
        if n_actual_seeds > 1:
            # Multiple seeds case - compute mean and standard error across seeds
            
            # First, ensure all seeds have the same length
            min_length = min(len(sizes) for sizes in all_set_sizes)
            truncated_sizes = [sizes[:min_length] for sizes in all_set_sizes]
            
            # Calculate windowed means for each seed
            windowed_means_by_seed = []
            for seed_sizes in truncated_sizes:
                window_avgs = []
                for j in range(0, len(seed_sizes) - errs_window + 1, errs_window):
                    window = seed_sizes[j:j+errs_window]
                    window_avgs.append(np.mean(window))
                windowed_means_by_seed.append(window_avgs)
            
            # Convert to numpy arrays for easier calculations
            windowed_means_array = np.array(windowed_means_by_seed)
            
            # Calculate mean and stderr across seeds for each time point
            mean_across_seeds = np.mean(windowed_means_array, axis=0)
            stderr_across_seeds = np.std(windowed_means_array, axis=0) / np.sqrt(n_actual_seeds)
            
            # Plot means and error bands
            x_vals = np.arange(0, len(mean_across_seeds) * errs_window, errs_window)
            plt.plot(x_vals, mean_across_seeds, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)
            
            # Add error bands from multiple seeds
            plt.fill_between(x_vals, 
                            mean_across_seeds - stderr_across_seeds,
                            mean_across_seeds + stderr_across_seeds, 
                            alpha=0.5, color=f'C{m_i}')
        else:
            # Single seed case - just do windowed averaging for smoothing
            set_sizes = all_set_sizes[0]
            
            # Apply windowed averaging for smoother plots
            window_avgs = []
            for j in range(0, len(set_sizes) - errs_window + 1, errs_window):
                window = set_sizes[j:j+errs_window]
                window_avgs.append(np.mean(window))
            
            # Plot the windowed averages
            x_vals = np.arange(0, len(window_avgs) * errs_window, errs_window)
            plt.plot(x_vals, window_avgs, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)

    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')
    plt.title(f'Prediction Set Size \n (Number of Classes in Set)', fontsize=title_size)
    plt.ylabel('Average Set Size ($\\leftarrow$)', fontsize=y_label_size)
    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=legend_size)
    plt.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_set_sizes_{setting}.pdf', bbox_inches='tight')
    
    # Plot class coverage rate
    plt.figure(figsize=(10, 8))
    
    for m_i, method in enumerate(methods):
        # Process all seeds for this method
        all_coverage = class_coverage_dict[method]
        n_actual_seeds = len(all_coverage)
        
        if n_actual_seeds > 1:
            # Multiple seeds case - compute mean and standard error across seeds
            
            # First, ensure all seeds have the same length
            min_length = min(len(cov) for cov in all_coverage)
            truncated_coverage = [cov[:min_length] for cov in all_coverage]
            
            # Calculate windowed means for each seed
            windowed_means_by_seed = []
            for seed_coverage in truncated_coverage:
                window_avgs = []
                for j in range(0, len(seed_coverage) - errs_window + 1, errs_window):
                    window = seed_coverage[j:j+errs_window]
                    window_avgs.append(np.mean(window))
                windowed_means_by_seed.append(window_avgs)
            
            # Convert to numpy arrays for easier calculations
            windowed_means_array = np.array(windowed_means_by_seed)
            
            # Calculate mean and stderr across seeds for each time point
            mean_across_seeds = np.mean(windowed_means_array, axis=0)
            stderr_across_seeds = np.std(windowed_means_array, axis=0) / np.sqrt(n_actual_seeds)
            
            # Plot means and error bands
            x_vals = np.arange(0, len(mean_across_seeds) * errs_window, errs_window)
            plt.plot(x_vals, mean_across_seeds, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)
            
            # Add error bands from multiple seeds
            plt.fill_between(x_vals, 
                            mean_across_seeds - stderr_across_seeds,
                            mean_across_seeds + stderr_across_seeds, 
                            alpha=0.5, color=f'C{m_i}')
        else:
            # Single seed case - just do windowed averaging for smoothing
            coverage = all_coverage[0]
            
            # Apply windowed averaging for smoother plots
            window_avgs = []
            for j in range(0, len(coverage) - errs_window + 1, errs_window):
                window = coverage[j:j+errs_window]
                window_avgs.append(np.mean(window))
            
            # Plot the windowed averages
            x_vals = np.arange(0, len(window_avgs) * errs_window, errs_window)
            plt.plot(x_vals, window_avgs, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)

    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')
    # if hasattr(args, 'alpha'):
    #     plt.axhline(y=1-args.alpha, color='r', linestyle='--', linewidth=2, label=f'Target ({1-args.alpha:.2f})')
    plt.title(f'Class Coverage Rate \n (Proportion of True Classes in Prediction Sets)', fontsize=title_size)
    plt.ylabel('Coverage Rate ($\\rightarrow$)', fontsize=y_label_size)
    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size)
    plt.ylim([0, 1.0])  # Adjust as needed
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=legend_size)
    plt.savefig(os.getcwd() + f'/../image_results/{plot_image_data}figs/{date.today()}_class_coverage_{setting}.pdf', bbox_inches='tight')