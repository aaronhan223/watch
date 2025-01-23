import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

def plot_martingale_paths(dataset0_paths_dict, dataset0_paths_stderr_dict, dataset0_name, martingales_0_dict,\
                          martingales_0_stderr_dict, dataset1_paths_dict,dataset1_paths_stderr_dict,errors_0_means_dict,\
                          errors_1_means_dict,errors_0_stderr_dict, errors_1_stderr_dict, p_vals_cal_dict, p_vals_test_dict,\
                          errs_window=100,change_point_index=None, title="Martingale Paths", xlabel="Observation Index", \
                          ylabel="Simple Jumper Martingale Value", martingale="martingale_paths", dataset0_shift_type='none',\
                          cov_shift_bias=0.0, plot_errors=False, n_seeds=1, cs_type='signed', setting=None, \
                          label_shift_bias=1,dataset1_name=None,martingales_1_dict=None,martingales_1_stderr_dict=None,\
                          noise_mu=0, noise_sigma=0, coverage_0_means_dict=[], coverage_0_stderr_dict=[],\
                          pvals_0_means_dict=[], pvals_0_stderr_dict=[], widths_0_medians_dict=[], widths_0_lower_q_dict=[],\
                          widths_0_upper_q_dict=[],methods=['none'], severity=None,schedule='variable',num_test_unshifted=1000,\
                          title_size=30, x_label_size=25, y_label_size=25, legend_size=20, x_tick_size=18, y_tick_size=18):
    """
    Plot martingale paths for red wine and white wine groups over time, similar to Figure 2 in the paper.
    
    Parameters:
    - dataset0_paths: List of arrays, where each array contains the martingale values for a path in the dataset0 group.
    - dataset0_name: Name of dataset0.
    - dataset1_paths: List of arrays, where each array contains the martingale values for a path in the dataset1 group.
    - cs_0: Conformity scores on test dataset0
    - cs_1: Conformity scores on test dataset1
    - change_point_index: The index where the change point occurs (vertical line).
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - plot_errors: Whether to also plot absolute errors.
    """
    plot_image_data = ''
    if (dataset0_name in ['mnist', 'cifar10']):
        plot_image_data = 'mnist_cifar_'
    
    method_name_dict = {'fixed_cal_dyn' : 'WCTM', 'fixed_cal' : 'WCTM', 'none' : 'CTM'}
    
    ####################
    ## Plot test statistic AND martingale paths
    ####################
    paths_0_dicts_all = [dataset0_paths_dict, martingales_0_dict]
    paths_0_stderr_dicts_all = [dataset0_paths_stderr_dict, martingales_0_stderr_dict]
    paths_1_dicts_all = [dataset1_paths_dict, martingales_1_dict]
    paths_1_stderr_dicts_all = [dataset1_paths_stderr_dict, martingales_1_stderr_dict]
    thresholds = [10**6, 10**2]
    
    for p_i, paths_0_dict in enumerate(paths_0_dicts_all):
        paths_0_stderr_dict = paths_0_stderr_dicts_all[p_i]
        statistic_name = martingale[p_i]
    
        plt.figure(figsize=(10, 8))

        # Plot dataset0 group with dashed lines
        for m_i, method in enumerate(methods):
            
            if severity is not None:
                
                plt.plot(paths_0_dicts_all[method][0], label=method_name_dict[method], linestyle='-', color=f'C{m_i}')
                plt.plot(dataset1_paths_dict[method][0], label=method_name_dict[method], linestyle='-', color=f'C{m_i+1}')

            else:
                for i, path in enumerate(paths_0_dict[method]):
                    martingale_stderrs = np.array(paths_0_stderr_dict[method][i])

                    plt.plot(path, label=method_name_dict[method], linestyle='-', color=f'C{m_i}')
                    plt.fill_between(np.arange(len(path)), \
                                     (path.to_numpy()-martingale_stderrs).flatten(), \
                                     (path.to_numpy()+martingale_stderrs).flatten(), alpha=0.5, color=f'C{m_i}')

    #         # Plot dataset1 group with solid lines
    #         for i, path in enumerate(dataset1_paths_dict[method]):
    #             plt.plot(path, label=f'Red Wine Fold {i+1}', linestyle='-', color=f'C{i+3}')

        # Add vertical line at the change point
        plt.axvline(x=change_point_index-num_test_unshifted, color='gray', linestyle='--', linewidth=3, label='Deployment time')
        plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')
        plt.axhline(y=thresholds[p_i], color='red', linestyle='--', label='Alarm threshold', linewidth=3)

        plt.yscale('log')  # Use logarithmic scale for the y-axis
        plt.xlabel(xlabel, fontsize=x_label_size)
        plt.ylabel(f'{statistic_name} values', fontsize=y_label_size)
        plt.title(f'Average {statistic_name} paths', fontsize=title_size)

#         if (dataset0_shift_type == 'none'):
#             plt.title(title, fontsize=26)
#         elif (dataset0_shift_type == 'covariate'):
#             plt.title(f'{title}, {dataset0_shift_type} shift, \n bias={str(cov_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#         elif dataset0_shift_type == 'label':
#             plt.title(f'{title}, {dataset0_shift_type} shift, \n label shift={str(label_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#         elif dataset0_shift_type == 'noise':
#             plt.title(f'{title}, {dataset0_shift_type} shift, \n mean var={str(noise_mu)} {str(noise_sigma)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#         else:
        if (plot_image_data == 'mnist_cifar_'):
            plt.title(f'{title}, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=title_size)

        plt.legend(fontsize=legend_size)
        plt.grid(True, which="both", ls="--")
        plt.xticks(fontsize=x_tick_size)        
        plt.yticks(fontsize=y_tick_size)
        
        if (dataset0_shift_type == 'none'):

            plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{dataset0_name}_{statistic_name}.pdf')
        else:
            plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{statistic_name}_{setting}.pdf')
            
    
#     plt.figure(figsize=(12, 8))
    
#     # Plot dataset0 group with dashed lines
#     for m_i, method in enumerate(methods):
# #         martingale_means = dataset0_paths_dict[method][0]
# #         martingale_stderrs = np.array(dataset0_paths_stderr_dict[method][0])
            
#         if severity is not None:

#             plt.plot(dataset0_paths_dict[method][0], label=dataset0_name + f' {method}', linestyle='-', color=f'C{m_i}')
#             plt.plot(dataset1_paths_dict[method][0], label=dataset1_name + f' {method}', linestyle='-', color=f'C{m_i+1}')

#         else:
#             for i, path in enumerate(dataset0_paths_dict[method]):
#                 martingale_stderrs = np.array(dataset0_paths_stderr_dict[method][i])
        
#                 plt.plot(path, label=dataset0_name + f' {method}, fold {i+1}', linestyle='-', color=f'C{m_i}')
#                 plt.fill_between(np.arange(len(path)), \
#                                  (path.to_numpy()-martingale_stderrs).flatten(), \
#                                  (path.to_numpy()+martingale_stderrs).flatten(), alpha=0.5, color=f'C{m_i}')

# #         # Plot dataset1 group with solid lines
# #         for i, path in enumerate(dataset1_paths_dict[method]):
# #             plt.plot(path, label=f'Red Wine Fold {i+1}', linestyle='-', color=f'C{i+3}')

#     # Add vertical line at the change point
#     plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')
#     plt.axhline(y=10**5, color='red', linestyle='--', label='Alarm threshold')

#     plt.yscale('log')  # Use logarithmic scale for the y-axis
#     plt.xlabel(xlabel, fontsize=24)
#     if (schedule == 'variable'):
#         plt.ylabel(ylabel, fontsize=24)

#     if (dataset0_shift_type == 'none'):
#         plt.title(title, fontsize=26)
#     elif (dataset0_shift_type == 'covariate'):
#         plt.title(f'{title}, {dataset0_shift_type} shift, \n bias={str(cov_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#     elif dataset0_shift_type == 'label':
#         plt.title(f'{title}, {dataset0_shift_type} shift, \n label shift={str(label_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#     elif dataset0_shift_type == 'noise':
#         plt.title(f'{title}, {dataset0_shift_type} shift, \n mean var={str(noise_mu)} {str(noise_sigma)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#     else:
#         plt.title(f'{title}, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=20)

#     plt.legend(fontsize=15)
#     plt.grid(True, which="both", ls="--")
    
#     if (dataset0_shift_type == 'none'):
            
#         plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{dataset0_name}_{martingale}.pdf')
#     else:
#         plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/sigma_{setting}.pdf')
    
    
    
    
    ## Plot absolute errors
    if (plot_errors):
        plt.figure(figsize=(10, 8))
        
        for m_i, method in enumerate(methods):
            ### Plotting errors (ie, abs(scores))
            # Plot dataset0 group with dashed lines
            for i, errs in enumerate(errors_0_means_dict[method]):
                plt.plot(np.arange(0, len(errs)*errs_window, errs_window), errs, label=method_name_dict[method], linestyle='-', color=f'C{m_i}')

                plt.fill_between(np.arange(0, len(errs)*errs_window, errs_window), \
                                 (errs-np.array(errors_0_stderr_dict[method][i])).flatten(), \
                                 (errs+np.array(errors_0_stderr_dict[method][i])).flatten(), alpha=0.5, color=f'C{m_i}')

#             # Plot dataset1 group with solid lines
#             for i, errs in enumerate(errors_1_means_dict[method]):
#                 plt.plot(np.abs(errs), label=dataset1_name + f' {method}, fold {i+1}', linestyle='-', color=f'C{m_i+i+3}')

        # Add vertical line at the change point
        plt.axvline(x=change_point_index-num_test_unshifted, color='gray', linestyle='--', linewidth=3, label='Deployment time')
        plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')

        plt.xlabel(xlabel, fontsize=x_label_size)
        plt.ylabel(r'Absolute error ($\leftarrow$)', fontsize=y_label_size)
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
            plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/{dataset0_name}_AbsoluteErrors.pdf')
        else:
            plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/error_{setting}.pdf')
        
            
    ## Plot coverage
    print("plotting coverage")
    plt.figure(figsize=(10, 8))
    
    # Plot dataset0 group with dashed lines
    for m_i, method in enumerate(methods):

        for i, coverage in enumerate(coverage_0_means_dict[method]):
            plt.plot(np.arange(0, len(coverage)*errs_window, errs_window), coverage, label=method_name_dict[method], linestyle='-', color=f'C{m_i}')
            plt.fill_between(np.arange(0, len(coverage)*errs_window, errs_window), \
                             (coverage-np.array(coverage_0_stderr_dict[method][i])).flatten(), \
                                 (coverage+np.array(coverage_0_stderr_dict[method][i])).flatten(), alpha=0.5,\
                             color=f'C{m_i}')
        plt.axhline(y=0.9, color='k', linestyle='--', linewidth=3, label='Target coverage')
    
    if severity is not None:
        plt.title(f'Coverage, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=title_size)
    else:
        plt.title(f'Mean coverage', fontsize=title_size)
    plt.ylabel(r'Coverage ($\rightarrow$)', fontsize=y_label_size)
    plt.axvline(x=change_point_index-num_test_unshifted, color='gray', linestyle='--', linewidth=3, label='Deployment time')
    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')
    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size) 
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=legend_size)
    plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/coverage_{setting}.pdf')
        
        
        
    ## Plot widths
    print("plotting widths")
    plt.figure(figsize=(10, 8))
    
    # Plot dataset0 group with dashed lines
    for m_i, method in enumerate(methods):

        for i, widths in enumerate(widths_0_medians_dict[method]):
            
            plt.plot(np.arange(0, len(widths)*errs_window, errs_window), widths, label=method_name_dict[method], linestyle='-', color=f'C{m_i}', linewidth=3)
            plt.fill_between(np.arange(0, len(widths)*errs_window, errs_window), \
                             (np.array(widths_0_lower_q_dict[method][i])).flatten(), \
                                 (np.array(widths_0_upper_q_dict[method][i])).flatten(), alpha=0.5,\
                             color=f'C{m_i}')
    
    if severity is not None:
        plt.title(f'Interval widths, {dataset0_shift_type} shift, \n {dataset0_name}, severity={severity}, n_seeds={n_seeds}, {cs_type} Scores', fontsize=title_size)
    else:
        plt.title(f'Median interval widths', fontsize=title_size)
    plt.ylabel(r'Interval widths ($\leftarrow$)', fontsize=y_label_size)
#     if (dataset0_name == 'meps'):
#         plt.ylim([0,18])
#     elif (dataset0_name == 'bike_sharing'):
#         plt.ylim([0,200])
    plt.axvline(x=change_point_index-num_test_unshifted, color='gray', linestyle='--', linewidth=3, label='Deployment time')
    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')
    plt.xticks(fontsize=x_tick_size)        
    plt.yticks(fontsize=y_tick_size)        
    plt.xlabel(xlabel, fontsize=x_label_size)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=legend_size)
    plt.savefig(os.getcwd() + f'/../{plot_image_data}figs/widths_{setting}.pdf')
        
        
        
        
    
#     ## Plot p-values sequence
#     plt.figure(figsize=(12, 8))
    
#     # Plot dataset0 group with dashed lines
#     for i, p_means in enumerate(pvals_0_means):
#         plt.plot(np.arange(0, len(p_means)*errs_window, errs_window), p_means, label=dataset0_name + f' Fold {i+1}', linestyle='-', color=f'C{i}')
#         plt.fill_between(np.arange(0, len(p_means)*errs_window, errs_window), \
#                          (p_means-np.array(pvals_0_stderr[i])).flatten(), \
#                              (p_means+np.array(pvals_0_stderr[i])).flatten(), alpha=0.5, color=f'C{i}')
        
#     plt.title(f'Average p-values, {dataset0_shift_type} shift, \n bias={str(cov_shift_bias)}, n_seeds={n_seeds}, {cs_type}Scores', fontsize=20)
#     plt.ylabel(r'Average p-values ($\rightarrow$)', fontsize=20)
#     plt.ylim([0,1])
#     plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')
#     plt.savefig(os.getcwd() + f'/../figs/pseq_' + setting + '.pdf')
    
    
    ## Plot p-values
    ## Plotting histograms of p-values
    fig, ax = plt.subplots(1, 2,figsize=(10, 8))
    
    for m_i, method in enumerate(methods):
        ax[0].hist(p_vals_cal_dict[method], label=method_name_dict[method], color=f'C{m_i}', alpha=0.5) #row=0, col=0
        ax[0].set_title('Before changepoint', fontsize=x_label_size)

        ax[1].hist(p_vals_test_dict[method], label=method_name_dict[method], color=f'C{m_i}', alpha=0.5) #row=1, col=0
        ax[1].set_title('After changepoint', fontsize=x_label_size)
    
    ax[0].tick_params(axis='both', which='major', labelsize=x_tick_size)
    ax[1].tick_params(axis='both', which='major', labelsize=x_tick_size)
#     plt.legend(fontsize=legend_size)
    fig.suptitle(f'Histograms of p-values', fontsize=title_size)
    fig.savefig(os.getcwd() + f'/../{plot_image_data}figs/p_vals_hist_{setting}.pdf')
