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
                          setting=None, methods=['none'], severity=None, 
                          title_size=28, x_label_size=25, y_label_size=25, 
                          legend_size=20, x_tick_size=18, y_tick_size=18):
    plot_image_data = ''
    if (dataset0_name in ['mnist', 'cifar10']):
        plot_image_data = 'mnist_cifar_'
    
    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    stat_validities = {'Shiryaev-Roberts' : 'Scheduled', 'Martingale' : 'Anytime-Valid'}
    stat_formal = {'Shiryaev-Roberts' : '($\sum_{i=0}^{t-1} M_t / M_i$)', 'Martingale' : '($M_t / M_0$)'}

    paths_0_dicts_all = [dataset0_paths_dict, martingales_0_dict]
    paths_0_stderr_dicts_all = [dataset0_paths_stderr_dict, martingales_0_stderr_dict]
    paths_1_dicts_all = [dataset1_paths_dict, martingales_1_dict]
    paths_1_stderr_dicts_all = [dataset1_paths_stderr_dict, martingales_1_stderr_dict]
    thresholds = [10**6, 10**6]
    for p_i, paths_0_dict in enumerate(paths_0_dicts_all):
        # paths_0_stderr_dict = paths_0_stderr_dicts_all[p_i]
        statistic_name = martingale[p_i]
        stat_validity = stat_validities[statistic_name]
        plt.figure(figsize=(10, 5))

        # Plot dataset0 group with dashed lines
        for m_i, method in enumerate(methods):
            
            plt.plot(paths_0_dict[method][0], label=method_name_dict[method], linestyle='-', color=f'C{m_i}')

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
        plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_{statistic_name}_clean_{setting}.pdf', bbox_inches='tight')

    for p_i, paths_1_dict in enumerate(paths_1_dicts_all):
        # paths_1_stderr_dict = paths_1_stderr_dicts_all[p_i]
        statistic_name = martingale[p_i]
        stat_validity = stat_validities[statistic_name]
        plt.figure(figsize=(10, 5))

        # Plot dataset0 group with dashed lines
        for m_i, method in enumerate(methods):
            
            plt.plot(paths_1_dict[method][0], label=method_name_dict[method], linestyle='-', color=f'C{m_i}')
            # plt.plot(dataset1_paths_dict[method][0], label=method_name_dict[method], linestyle='-', color=f'C{m_i+1}')

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
        plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_{statistic_name}_corrupted_{setting}.pdf', bbox_inches='tight')


def plot_errors(errors_0_means_dict, errors_0_stderr_dict, 
                errs_window, dataset0_name, change_point_index,
                xlabel="Test (Deployment) Datapoint Index $t$", 
                dataset0_shift_type='none', cov_shift_bias=0.0, n_seeds=1, cs_type='signed', 
                setting=None, label_shift_bias=1, noise_mu=0, noise_sigma=0, methods=['none'], severity=None, 
                title_size=28, x_label_size=25, y_label_size=25, 
                legend_size=20, x_tick_size=18, y_tick_size=18):
    
    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    if (dataset0_name in ['mnist', 'cifar10']):
        plot_image_data = 'mnist_cifar_'

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
        plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_{dataset0_name}_AbsoluteErrors.pdf', bbox_inches='tight')
    else:
        plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_error_{setting}.pdf', bbox_inches='tight')


def plot_coverage(coverage_0_means_dict, coverage_0_stderr_dict, errs_window, dataset0_name, change_point_index, 
                  xlabel="Test (Deployment) Datapoint Index $t$",
                  dataset0_shift_type='none', n_seeds=1, cs_type='signed', 
                  setting=None, methods=['none'], severity=None, title_size=28, x_label_size=25, y_label_size=25, 
                  legend_size=20, x_tick_size=18, y_tick_size=18):

    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    if (dataset0_name in ['mnist', 'cifar10']):
        plot_image_data = 'mnist_cifar_'
    plt.figure(figsize=(10, 8))
    
    for m_i, method in enumerate(methods):
        for i, coverage in enumerate(coverage_0_means_dict[method]):
            plt.plot(np.arange(0, len(coverage)*errs_window, errs_window), coverage, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)
            plt.fill_between(np.arange(0, len(coverage)*errs_window, errs_window), \
                             (coverage-np.array(coverage_0_stderr_dict[method][i])).flatten(), \
                                 (coverage+np.array(coverage_0_stderr_dict[method][i])).flatten(), alpha=0.5,\
                             color=f'C{m_i}')
    plt.axhline(y=0.9, color='k', linestyle='--', linewidth=3, label='Target coverage')
    plt.ylim([0.85, 1.0])
    plt.yticks(np.arange(0.85, 1.01, 0.05))

    if severity is not None:
        plt.title(f'Coverage, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=title_size)
    else:
        plt.title(f'Coverage \n (Prediction Safety)', fontsize=title_size)
    plt.ylabel(r'Mean Coverage ($\rightarrow$)', fontsize=y_label_size)
    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')
    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size) 
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=legend_size)
    plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_coverage_{setting}.pdf', bbox_inches='tight')


def plot_widths(widths_0_medians_dict, dataset0_name, errs_window, change_point_index,
                widths_0_lower_q_dict=[], widths_0_upper_q_dict=[],
                xlabel="Test (Deployment) Datapoint Index $t$", 
                dataset0_shift_type='none', n_seeds=1, cs_type='signed', 
                setting=None, methods=['none'], severity=None, title_size=28, x_label_size=25, y_label_size=25, 
                legend_size=20, x_tick_size=18, y_tick_size=18):

    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    if (dataset0_name in ['mnist', 'cifar10']):
        plot_image_data = 'mnist_cifar_'
    plt.figure(figsize=(10, 8))
    
    for m_i, method in enumerate(methods):
        for i, widths in enumerate(widths_0_medians_dict[method]):
            
            plt.plot(np.arange(0, len(widths)*errs_window, errs_window), widths, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)
            plt.fill_between(np.arange(0, len(widths)*errs_window, errs_window), \
                             (np.array(widths_0_lower_q_dict[method][i])).flatten(), \
                                 (np.array(widths_0_upper_q_dict[method][i])).flatten(), alpha=0.5,\
                             color=f'C{m_i}')
    
    if severity is not None:
        plt.title(f'Interval Widths, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=title_size)
    else:
        plt.title(f'Interval Widths \n (Prediction Informativeness)', fontsize=title_size)
    plt.ylabel(r'Median Interval Widths ($\leftarrow$)', fontsize=y_label_size)
    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Changepoint')
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size)        
    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=legend_size)
    plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_widths_{setting}.pdf', bbox_inches='tight')


def plot_p_vals(p_vals_pre_change_dict, p_vals_post_change_dict, dataset0_name, 
                setting=None, methods=['none'], title_size=28, x_label_size=25, x_tick_size=18):

    method_name_dict = {'fixed_cal_offline' : 'WCTM (proposed)', 'fixed_cal' : 'WCTM (proposed)', 'none' : 'CTM (Vovk et al., 2021)'}
    if (dataset0_name in ['mnist', 'cifar10']):
        plot_image_data = 'mnist_cifar_'

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
    fig.savefig(os.getcwd() + f'/../{plot_image_data}figs/{date.today()}_p_vals_hist_{setting}.pdf', bbox_inches='tight')