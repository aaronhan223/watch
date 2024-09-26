import numpy as np
import matplotlib.pyplot as plt

def plot_martingale_paths(red_wine_paths, white_wine_paths, change_point_index=None, title="Martingale Paths", 
                        xlabel="Observation Index", ylabel="Simple Jumper Martingale Value", file_name="martingale_paths", 
                        shift_type = False):
    """
    Plot martingale paths for red wine and white wine groups over time, similar to Figure 2 in the paper.
    
    Parameters:
    - red_wine_paths: List of arrays, where each array contains the martingale values for a path in the red wine group.
    - white_wine_paths: List of arrays, where each array contains the martingale values for a path in the white wine group.
    - change_point_index: The index where the change point occurs (vertical line).
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(12, 8))

    # Plot red wine group with solid lines
    for i, path in enumerate(red_wine_paths):
        plt.plot(path, label=f'Red Wine Fold {i+1}', linestyle='-', color=f'C{i}')

    # Plot white wine group with dashed lines
    for i, path in enumerate(white_wine_paths):
        plt.plot(path, label=f'White Wine Fold {i+1}', linestyle='-', color=f'C{i+3}')

    # Add vertical line at the change point
    plt.axvline(x=change_point_index, color='k', linestyle='solid', linewidth=5, label='Change Point')

    plt.yscale('log')  # Use logarithmic scale for the y-axis
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.title(title, fontsize=26)
    plt.legend(fontsize=15)
    plt.grid(True, which="both", ls="--")
    
    if (shift_type == 'none'):
        plt.savefig(f'../figs/{file_name}.pdf')
    elif (shift_type == 'covariate'):
        plt.savefig(f'../figs/{file_name}_cov_shift.pdf')