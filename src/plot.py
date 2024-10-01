import numpy as np
import matplotlib.pyplot as plt
import os

def plot_martingale_paths(dataset0_paths, dataset0_name, dataset1_paths, cs_0, cs_1, change_point_index=None, \
                          title="Martingale Paths", xlabel="Observation Index", ylabel="Simple Jumper Martingale Value", \
                          file_name="martingale_paths", dataset0_shift_type = 'none', cov_shift_bias=0.0, \
                          plot_errors=False):
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
    plt.figure(figsize=(12, 8))

    
    # Plot dataset0 group with dashed lines
    for i, path in enumerate(dataset0_paths):
        plt.plot(path, label=dataset0_name + f' Fold {i+1}', linestyle='-', color=f'C{i+3}')

    # Plot dataset1 group with solid lines
    for i, path in enumerate(dataset1_paths):
        plt.plot(path, label=f'Red Wine Fold {i+1}', linestyle='-', color=f'C{i}')


    # Add vertical line at the change point
    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')

    plt.yscale('log')  # Use logarithmic scale for the y-axis
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=26)
    
    if (dataset0_shift_type != 'none'):
        plt.title(title + ', ' + dataset0_shift_type + ' shift, bias=' + str(cov_shift_bias), fontsize=22)
#     plt.ylim([10.0, 100000])
    plt.legend(fontsize=15)
    plt.grid(True, which="both", ls="--")
    
    if (dataset0_shift_type == 'none'):
        plt.savefig(os.getcwd() + f'/../figs/{dataset0_name}_{file_name}.pdf')
    elif (dataset0_shift_type == 'covariate'):
        plt.savefig(os.getcwd() + f'/../figs/{dataset0_name}_{file_name}_CovShift.pdf')
    
    
    if (plot_errors):
        plt.figure(figsize=(12, 8))
        
        ### Plotting errors (ie, abs(scores))
        # Plot dataset0 group with dashed lines
        for i, cs in enumerate(cs_0):
            window=100
            cs_averaged = [np.mean(np.abs(cs[(j*window):((j+1)*window)])) for j in range(0, len(cs))]
            plt.plot(np.array(range(0, len(cs)))*window, cs_averaged, label=dataset0_name + f' Fold {i+1}', linestyle='-', color=f'C{i+3}')

        # Plot dataset1 group with solid lines
        for i, cs in enumerate(cs_1):
            plt.plot(np.abs(cs), label=f'Red Wine Fold {i+1}', linestyle='-', color=f'C{i}')

        # Add vertical line at the change point
        plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')

        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel('Absolute error', fontsize=24)
        plt.title('Error paths', fontsize=26)
        plt.legend(fontsize=15)
        plt.grid(True, which="both", ls="--")

    
        if (dataset0_shift_type == 'none'):
            plt.savefig(os.getcwd() + f'/../figs/{dataset0_name}_AbsoluteErrors.pdf')
        elif (dataset0_shift_type == 'covariate'):
            plt.savefig(os.getcwd() + f'/../figs/{dataset0_name}_AbsoluteErrors_CovShift.pdf')