import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_smooth_loss(file_path):
    # Lists to store epochs and loss values
    epochs = []
    losses = []
    
    # Read the file and extract data
    with open(file_path, 'r') as file:
        # Read all lines and remove duplicates while preserving order
        lines = list(dict.fromkeys(file.readlines()))
        
        for line in lines:
            # Extract epoch and loss using regular expression
            match = re.match(r'Epoch (\d+): Loss = ([^,]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2)) if match.group(2) != 'inf' else float('inf')
                
                epochs.append(epoch)
                losses.append(loss)
    
    # Convert to numpy arrays
    epochs = np.array(epochs)
    losses = np.array(losses)
    
    # Remove inf values for smoothing
    mask = np.isfinite(losses)
    epochs_finite = epochs[mask]
    losses_finite = losses[mask]
    
    # Create smooth curve
    if len(epochs_finite) > 3:  # Need at least 4 points for cubic spline
        # Create a finer x scale for smoother curve
        X_smooth = np.linspace(epochs_finite.min(), epochs_finite.max(), 300)
        
        # Create the spline function
        spl = make_interp_spline(epochs_finite, losses_finite, k=3)
        Y_smooth = spl(X_smooth)
        
        # Ensure no negative values in smoothed curve
        Y_smooth = np.maximum(Y_smooth, 0)
    else:
        X_smooth = epochs_finite
        Y_smooth = losses_finite

    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot original points
    # plt.scatter(epochs_finite, losses_finite, color='blue', alpha=0.5, 
    #            label='Original data', zorder=2)
    
    # Plot smooth curve
    plt.plot(X_smooth, Y_smooth, color='red', linewidth=2, 
             label='Loss curve', zorder=1)
    
    # Customize the plot
    plt.title('Training Loss over Epochs', fontsize=15, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Customize axis
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add padding to y-axis
    plt.margins(y=0.1)
    
    # Set y-axis limits to show only relevant range
    if len(losses_finite) > 0:
        plt.ylim(0, max(losses_finite) * 1.1)
    
    # Tight layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

# Usage
file_path = 'hw1_3.txt'  # Replace with your file path
plot_smooth_loss(file_path)